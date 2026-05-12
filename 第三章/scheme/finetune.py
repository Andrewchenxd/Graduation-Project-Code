"""
微调策略模块

实现单数据集分类微调的训练/验证循环。
支持 RMLA(11类), RMLB(10类), RMLC(11类) 三个数据集。
"""

import os
import numpy as np
import torch
import torch.nn as nn
import scipy.io as scio
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, List, Any

from dataset import SigDataSet_sgn
from tool.utils import (
    AverageMeter, CSVStats, EarlyStopping,
    load_dict, adjust_learning_rate
)
from sklearn.metrics import confusion_matrix


def acc_classes(pre: torch.Tensor, labels: torch.Tensor, batch_size: int) -> float:
    """计算分类准确率"""
    pre_y = torch.max(pre, dim=1)[1]
    train_acc = torch.eq(pre_y, labels.to(pre.device)).sum().item() / batch_size
    return train_acc


def acc_AA(pre: torch.Tensor, labels: torch.Tensor,
           acc_AA_pre: np.ndarray, acc_AA_count: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """累计每个类别的准确率"""
    pre_y = torch.max(pre, dim=1)[1]
    pre_y = pre_y.detach().cpu().numpy()
    labelclass = np.array(labels.cpu())
    for i in range(len(labelclass)):
        if pre_y[i] == labelclass[i]:
            acc_AA_pre[0, labelclass[i]] += 1
            acc_AA_count[0, labelclass[i]] += 1
        else:
            acc_AA_count[0, labelclass[i]] += 1
    return acc_AA_pre, acc_AA_count


def acc_snrs(pre: torch.Tensor, labels: torch.Tensor, snr: torch.Tensor,
             acc_snr_pre: np.ndarray, acc_snr_count: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """累计每个 SNR 的准确率"""
    pre_y = torch.max(pre, dim=1)[1]
    pre_y = pre_y.detach().cpu().numpy()
    labelclass = np.array(labels.cpu())
    snr_np = snr.detach().cpu().numpy()
    for i in range(len(labelclass)):
        if pre_y[i] == labelclass[i]:
            acc_snr_pre[0, snr_np[i]] += 1
            acc_snr_count[0, snr_np[i]] += 1
        else:
            acc_snr_count[0, snr_np[i]] += 1
    return acc_snr_pre, acc_snr_count


def calculate_accuracy(true_label: np.ndarray, pre_label: np.ndarray,
                       classesnum: int) -> Tuple[np.ndarray, float, float]:
    """计算混淆矩阵、OA、AA"""
    cm = confusion_matrix(true_label, pre_label)
    accuracy_matrix = np.zeros((1, classesnum))
    for i in range(classesnum):
        accuracy_matrix[0, i] = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
    OA = np.trace(cm) / np.sum(cm)
    AA = np.mean(accuracy_matrix)
    accuracy_matrix = accuracy_matrix.flatten()
    return accuracy_matrix, OA, AA


def train_one_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    epoch_max: int,
    config: Any,
    adsbis: bool = False
) -> Tuple[float, float]:
    """
    训练一个 epoch（分类任务）。

    Args:
        train_loader: 训练 DataLoader
        model: 模型
        criterion: 损失函数 (CrossEntropyLoss)
        optimizer: 优化器
        epoch: 当前 epoch
        epoch_max: 总 epoch 数
        config: 配置对象
        adsbis: 是否使用 ADSB 数据集

    Returns:
        (accuracy, loss)
    """
    losses_class = AverageMeter()
    acc = AverageMeter()

    if adsbis:
        acc_snr_pre = np.zeros((1, 31))
        acc_snr_count = np.zeros((1, 31))
    elif config.trans_choose in ["pwvd", "gasf", "pwvd_2018"]:
        acc_snr_pre = np.zeros((1, 26))
        acc_snr_count = np.zeros((1, 26))
    else:
        acc_snr_pre = np.zeros((1, 20))
        acc_snr_count = np.zeros((1, 20))

    scaler = GradScaler()
    model.train()
    device = next(model.parameters()).device

    with tqdm(total=len(train_loader), desc=f'Epoch{epoch}/{epoch_max}', postfix=dict, mininterval=0.3) as pbar:
        for i, (input1, target, snr) in enumerate(train_loader):
            images, labels = input1, target

            with autocast(device_type='cuda', enabled=config.autocast):
                if config.swav_is:
                    if config.softloss:
                        output = model(
                            images.to(device), images.to(device), images.to(device),
                            enable_mask=False, dataset_name='RML2016a+b_total_snr'
                        )
                    else:
                        output = model(
                            images.to(device), images.to(device),
                            enable_mask=False, dataset_name='RML2016a+b_total_snr'
                        )
                else:
                    output = model(
                        images.to(device), enable_mask=False,
                        dataset_name='RML2016a+b_total_snr'
                    )

                target_var = labels.to(device)
                loss = criterion(output, target_var)

            # 记录准确率
            acc.update(acc_classes(output.data, target, config.batchsize))
            if adsbis:
                acc_snrs(output, labels, snr - 10, acc_snr_pre, acc_snr_count)
            else:
                acc_snrs(output, labels, snr, acc_snr_pre, acc_snr_count)
            losses_class.update(loss.item())

            # 反向传播
            if config.autocast and torch.cuda.is_available():
                optimizer.zero_grad()
                scaler.scale(loss.mean()).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pbar.set_postfix(**{'train_loss_': losses_class.avg, 'acc': acc.avg})
            pbar.update(1)

    print(f"  SNR acc: {acc_snr_pre / (acc_snr_count + 1e-8) * 100}")
    return acc.avg, losses_class.avg


@torch.no_grad()
def validate_one_epoch(
    val_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    epoch: int,
    epoch_max: int,
    config: Any,
    adsbis: bool = False
) -> Tuple[float, float]:
    """
    验证一个 epoch（分类任务）。

    Args:
        参数同 train_one_epoch

    Returns:
        (accuracy, loss)
    """
    losses_class = AverageMeter()
    acc = AverageMeter()

    if adsbis:
        acc_snr_pre_val = np.zeros((1, 31))
        acc_snr_count_val = np.zeros((1, 31))
    elif config.trans_choose in ["pwvd", "gasf"]:
        acc_snr_pre_val = np.zeros((1, 21))
        acc_snr_count_val = np.zeros((1, 21))
    elif config.trans_choose == "pwvd_2018":
        acc_snr_pre_val = np.zeros((1, 26))
        acc_snr_count_val = np.zeros((1, 26))
    else:
        acc_snr_pre_val = np.zeros((1, 20))
        acc_snr_count_val = np.zeros((1, 20))

    acc_AA_pre = np.zeros((1, config.numclass))
    acc_AA_count = np.zeros((1, config.numclass))

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f'Epoch{epoch}/{epoch_max}', postfix=dict, mininterval=0.3,
                  colour='blue') as pbar:
            for i, (input1, target, snr) in enumerate(val_loader):
                val_image, val_label = input1, target

                with autocast(device_type='cuda', enabled=config.autocast):
                    if config.swav_is:
                        if config.softloss:
                            output = model(
                                val_image.to(device), val_image.to(device), val_image.to(device),
                                enable_mask=False, dataset_name='RML2016a+b_total_snr'
                            )
                        else:
                            output = model(
                                val_image.to(device), val_image.to(device),
                                enable_mask=False, dataset_name='RML2016a+b_total_snr'
                            )
                    else:
                        output = model(
                            val_image.to(device), enable_mask=False,
                            dataset_name='RML2016a+b_total_snr'
                        )

                    target_var = val_label.to(device)
                    loss = criterion(output, target_var)

                acc.update(acc_classes(output.data, target, config.batchsize))
                if adsbis:
                    acc_snrs(output, val_label, snr - 10, acc_snr_pre_val, acc_snr_count_val)
                else:
                    acc_snrs(output, val_label, snr, acc_snr_pre_val, acc_snr_count_val)
                losses_class.update(loss.item())

                pbar.set_postfix(**{'val_loss_class': losses_class.avg, 'acc': acc.avg})
                pbar.update(1)

    print(f"  Val SNR acc: {acc_snr_pre_val / (acc_snr_count_val + 1e-8) * 100}")
    return acc.avg, losses_class.avg


@torch.no_grad()
def validate_for_result(
    val_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    epoch: int,
    epoch_max: int,
    config: Any,
    adsbis: bool = False
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    验证并收集所有预测结果（用于保存最佳结果）。

    Returns:
        (accuracy, loss, true_label, pre_label, SNR)
    """
    losses_class = AverageMeter()
    acc = AverageMeter()

    if adsbis:
        acc_snr_pre = np.zeros((1, 31))
        acc_snr_count = np.zeros((1, 31))
    elif config.trans_choose in ["pwvd", "gasf", "pwvd_2018"]:
        acc_snr_pre = np.zeros((1, 26))
        acc_snr_count = np.zeros((1, 26))
    else:
        acc_snr_pre = np.zeros((1, 20))
        acc_snr_count = np.zeros((1, 20))

    model.eval()
    device = next(model.parameters()).device
    true_label_list = []
    pre_label_list = []
    SNR_list = []

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f'Epoch{epoch}/{epoch_max}', postfix=dict, mininterval=0.3,
                  colour='blue') as pbar:
            for i, (input1, target, snr) in enumerate(val_loader):
                val_image, val_label = input1, target

                if config.swav_is:
                    if config.softloss:
                        output = model(
                            val_image.to(device), val_image.to(device), val_image.to(device),
                            enable_mask=False, dataset_name='RML2016a+b_total_snr'
                        )
                    else:
                        output = model(
                            val_image.to(device), val_image.to(device),
                            enable_mask=False, dataset_name='RML2016a+b_total_snr'
                        )
                else:
                    output = model(
                        val_image.to(device), enable_mask=False,
                        dataset_name='RML2016a+b_total_snr'
                    )

                pre_y = torch.max(output, dim=1)[1]
                pre_y = pre_y.detach().cpu().numpy()
                snr_np = snr.detach().cpu().numpy()
                val_label_np = val_label.detach().cpu().numpy()

                SNR_list.append(snr_np)
                pre_label_list.append(pre_y)
                true_label_list.append(val_label_np)

                loss = 0
                if adsbis:
                    acc_snrs(output, val_label, snr - 10, acc_snr_pre, acc_snr_count)
                else:
                    acc_snrs(output, val_label, snr, acc_snr_pre, acc_snr_count)
                acc.update(acc_classes(output.data, target, config.batchsize))
                losses_class.update(loss)

                pbar.set_postfix(**{'val_loss_class': losses_class.avg, 'acc': acc.avg})
                pbar.update(1)

    true_label = np.concatenate(true_label_list)
    pre_label = np.concatenate(pre_label_list)
    SNR = np.concatenate(SNR_list)

    print(f"  SNR acc: {acc_snr_pre / (acc_snr_count + 1e-8) * 100}")
    return acc.avg, losses_class.avg, true_label, pre_label, SNR


def build_model(config: Any, yaml_config: Any) -> nn.Module:
    """根据配置构建模型"""
    from model.radiollm import RadioLLM
    from model.RadioLLM_RAG import RadioLLM_RAG_HNSW

    if config.model_name == 'RadioLLM':
        model = RadioLLM(config, yaml_config=yaml_config)
    elif config.model_name == 'RadioLLM_RAG':
        model = RadioLLM_RAG_HNSW(config, yaml_config=yaml_config)
    else:
        raise ValueError(f"Unknown model_name: {config.model_name}")

    return model


def run_finetune(config: Any) -> None:
    """
    运行微调主流程。

    Args:
        config: 解析后的配置对象
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Config:\n{config}')

    # 加载数据集 YAML 配置
    from tool.parser import load_config
    yaml_config = load_config(config.dataset_config)

    # 数据路径
    train_data_path = config.train_data_path
    val_data_path = config.test_data_path

    print(f'Train data: {train_data_path}')
    print(f'Test data: {val_data_path}')

    # 创建数据集
    if config.trans_choose in ["pwvd", "stft"]:
        norm_style = config.norm_style
        print(f'trans_choose: {config.trans_choose}')
        print(f'norm_style: {norm_style}')
        print(f'dataset: {config.dataset}')

        train_set = SigDataSet_sgn(
            train_data_path, newdata=config.newdata,
            adsbis=config.adsbis,
            resample_is=config.resample, samplenum=config.samplenum,
            resize_is=False, norm=norm_style,
            snr_range=[0, 20], sgnaug=True, return_label=True
        )

        val_set = SigDataSet_sgn(
            val_data_path, newdata=config.newdata,
            adsbis=config.adsbis,
            resample_is=config.resample, samplenum=config.samplenum,
            resize_is=False, norm=norm_style,
            snr_range=[0, 20], sgnaug=False, return_label=True
        )

    train_loader = DataLoader(
        train_set, batch_size=config.batchsize, shuffle=True,
        num_workers=config.numworks, prefetch_factor=config.pref,
        pin_memory=True, persistent_workers=True
    )

    val_loader = DataLoader(
        val_set, batch_size=config.batchsize,
        num_workers=config.numworks, prefetch_factor=config.pref,
        shuffle=True, pin_memory=True, persistent_workers=True
    )

    # 构建模型
    model = build_model(config, yaml_config)

    if torch.cuda.device_count() > 1:
        print(f"Use {torch.cuda.device_count()} gpus")
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        model = model.cuda()

    # 加载预训练权重
    if config.model_path and os.path.exists(config.model_path):
        load_dict(model, config.model_path, device)
        print(f"Loaded model from {config.model_path}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    optimizer_sgd = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=0.9,
        weight_decay=0.005,
        nesterov=True
    )

    optimizer_adam = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.95),
        weight_decay=0.3
    )

    optimizer = optimizer_adam
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-8
    )

    # 日志和早停
    csv_logger = CSVStats(save_dir=config.runs_dir)
    early_stopping = EarlyStopping(
        save_path=config.checkpoint_dir,
        patience=config.patience,
        wait=config.wait,
        choose=config.trans_choose,
        is_LORA=config.is_LORA,
        save_best=True
    )

    wait_idem = config.wait
    declay_count = 0
    best_val = 0

    # 创建结果保存目录
    os.makedirs(config.result_dir, exist_ok=True)

    # 训练循环
    for epoch in range(0, config.epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"{'=' * 60}")

        acc_train, loss_train = train_one_epoch(
            train_loader, model, criterion, optimizer,
            epoch, epoch_max=config.epochs,
            config=config, adsbis=config.adsbis
        )

        acc_val, loss_val, true_label, pre_label, SNR = validate_for_result(
            val_loader, model, criterion,
            epoch, epoch_max=config.epochs,
            config=config, adsbis=False
        )

        # 保存最佳结果
        if acc_val > best_val:
            best_val = acc_val
            result = {'true_label': true_label, 'pre_label': pre_label, 'SNR': SNR}
            save_path = os.path.join(
                config.result_dir,
                f'best_result_fewshot{config.torch_seed}_seed{config.few_shotnum}_epoch{epoch}.mat'
            )
            scio.savemat(save_path, result)
            print(f"  New best accuracy: {best_val:.4f}, saved to {save_path}")

        csv_logger.add(acc_train, acc_val, loss_train, loss_val, config.lr)
        csv_logger.write(
            patience=config.patience, wait=config.wait,
            choose=config.trans_choose, name=config.dataset,
            seed=config.torch_seed, few_shotnum=config.few_shotnum
        )

        early_stopping(loss_val, model)
        if early_stopping.counter >= config.wait:
            config.lr = adjust_learning_rate(optimizer, config.lr, config.declay)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("\nFinetuning completed.")
