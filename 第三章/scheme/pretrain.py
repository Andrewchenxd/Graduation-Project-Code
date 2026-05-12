"""
预训练策略模块

实现多数据集联合预训练的训练/验证循环。
支持 masking、noise、mixup 三种训练策略，
以及多数据集概率采样 (MergedLoader)。
"""

import time
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Any

from dataset import SigDataSet_freq_units2
from tool.rml2016a import prepare_data, split_dataset
from tool.utils import (
    AverageMeter, CSVStats, EarlyStopping, MergedLoader,
    load_dict, adjust_learning_rate, mixup, compute_ssim,
    calculate_mask_ratio, set_random_seed
)


def freq_norm(sgn, freq_choose='stft'):
    """频率特征归一化"""
    if freq_choose == 'fft':
        sgn_max, _ = torch.max(sgn, dim=1)
        sgn_min, _ = torch.min(sgn, dim=1)
        sgn_min = sgn_min.unsqueeze(1)
        sgn_max = sgn_max.unsqueeze(1)
        freq = (2 * sgn - sgn_min - sgn_max) / (sgn_max - sgn_min)
    elif freq_choose == 'stft':
        sgn_max, _ = torch.max(sgn, dim=1)
        sgn_max, _ = torch.max(sgn_max, dim=1)
        sgn_min, _ = torch.min(sgn, dim=1)
        sgn_min, _ = torch.min(sgn_min, dim=1)
        sgn_min = sgn_min.unsqueeze(1).unsqueeze(2)
        sgn_max = sgn_max.unsqueeze(1).unsqueeze(2)
        freq = (2 * sgn - sgn_min - sgn_max) / (sgn_max - sgn_min)
    return freq


def phases_freq_loss(x_pred, sgn_c, device):
    """计算相位和频率损失"""
    I_pred = x_pred[0, :].squeeze(1)
    Q_pred = x_pred[1, :].squeeze(1)
    I = sgn_c[0, :].squeeze(1).to(device)
    Q = sgn_c[1, :].squeeze(1).to(device)
    phases = torch.atan2(I, Q)
    phases_pred = torch.atan2(I_pred, Q_pred)
    freq = torch.abs(torch.fft.fft(I + Q * 1j))
    freq_pred = torch.abs(torch.fft.fft(I_pred + Q_pred * 1j))
    freq = freq_norm(freq, freq_choose='fft')
    freq_pred = freq_norm(freq_pred, freq_choose='fft')
    phases_pred = freq_norm(phases_pred, freq_choose='fft')
    phases = freq_norm(phases, freq_choose='fft')
    return freq, freq_pred, phases, phases_pred


def train_one_epoch(
    train_loaders: List[DataLoader],
    model: nn.Module,
    criterion: nn.Module,
    criterion2: nn.Module,
    criterion3: nn.Module,
    criterion4: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    epoch_max: int,
    config: Any,
    scheduler: Any = None,
    adsbis: bool = False,
    prob: List[float] = [],
    thresholds: Dict[str, List[float]] = {},
    balence: List[float] = [],
    Names: List[str] = []
) -> Tuple[float, float]:
    """
    训练一个 epoch。

    Args:
        train_loaders: 各数据集的 DataLoader 列表
        model: 模型
        criterion: MSE 损失
        criterion2: SmoothL1 损失
        criterion3: CrossEntropy 损失
        criterion4: CrossEntropy 损失
        optimizer: 优化器
        epoch: 当前 epoch
        epoch_max: 总 epoch 数
        config: 配置对象
        scheduler: 学习率调度器
        adsbis: 是否使用 ADSB 数据集
        prob: 各数据集的采样概率
        thresholds: 各数据集的策略阈值
        balence: 各数据集的平衡系数
        Names: 数据集名称列表

    Returns:
        (accuracy, loss)
    """
    losses_class = AverageMeter()
    losses_class1 = AverageMeter()
    losses_class2 = AverageMeter()
    losses_class3 = AverageMeter()
    acc = AverageMeter()

    total_len = 0
    merged_loader = MergedLoader(train_loaders)
    scaler = GradScaler()
    for loader in train_loaders:
        total_len += len(loader)

    model.train()
    device = next(model.parameters()).device

    with tqdm(total=total_len, desc=f'Epoch{epoch}/{epoch_max}', postfix=dict, mininterval=0.3) as pbar:
        for batch, i_loader in merged_loader:
            input1, input2, input3, input4, input5, input6, input7, input8, dataset_name = batch
            sgn_c, sgn_n, sgn_a, sgn_f, freq_c, freq_n, snrs, labels = \
                input1, input2, input3, input4, input5, input6, input7, input8

            with autocast(device_type='cuda', enabled=config.autocast):
                if random.random() <= thresholds[Names[i_loader]][0]:
                    # 策略1: Masking
                    x_pred, mask = model(sgn_c.to(device), enable_mask=True, dataset_name=dataset_name[0])
                    x = sgn_c.to(device)
                    mask = mask.unsqueeze(1).expand_as(x)
                    loss1 = criterion(balence[i_loader] * x_pred * mask, balence[i_loader] * x * mask)
                    freq, freq_pred, phases, phases_pred = phases_freq_loss(x_pred, sgn_c, device)
                    loss4 = criterion(phases_pred, phases)
                    loss = config.alpha * loss1 + config.delta * loss4
                    losses_class1.update(loss.item())

                elif random.random() <= thresholds[Names[i_loader]][1]:
                    # 策略2: Noise
                    x_pred = model(sgn_n.to(device), enable_mask=False, dataset_name=dataset_name[0])
                    loss2 = criterion(balence[i_loader] * x_pred, balence[i_loader] * sgn_f.to(device))
                    loss3 = criterion(balence[i_loader] * x_pred, balence[i_loader] * sgn_c.to(device))
                    freq, freq_pred, phases, phases_pred = phases_freq_loss(x_pred, sgn_c, device)
                    loss4 = criterion(phases_pred, phases)
                    loss = config.beta * loss2 + config.lamba * loss3 + config.delta * loss4
                    losses_class2.update(loss.item())

                elif random.random() <= thresholds[Names[i_loader]][2]:
                    # 策略3: Mixup
                    sgn_m = mixup(sgn_c)
                    x_pred = model(sgn_m.to(device), enable_mask=False, dataset_name=dataset_name[0])
                    loss2 = criterion(balence[i_loader] * x_pred, balence[i_loader] * sgn_f.to(device))
                    loss3 = criterion(balence[i_loader] * x_pred, balence[i_loader] * sgn_c.to(device))
                    freq, freq_pred, phases, phases_pred = phases_freq_loss(x_pred, sgn_c, device)
                    loss4 = criterion(phases_pred, phases)
                    loss = config.beta * loss2 + config.lamba * loss3 + config.delta * loss4
                    losses_class3.update(loss.item())

            acc.update(0)

            if config.autocast:
                optimizer.zero_grad()
                scaler.scale(loss.mean()).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()

            losses_class.update(loss.mean().item())
            if scheduler is not None:
                scheduler.step()

            pbar.set_postfix(**{'train_loss_': losses_class.avg, 'acc': acc.avg})
            pbar.update(1)

    torch.cuda.empty_cache()
    print(f"  Mask loss: {losses_class1.avg:.6f}, Noise loss: {losses_class2.avg:.6f}, Mixup loss: {losses_class3.avg:.6f}")

    return acc.avg, losses_class.avg


@torch.no_grad()
def validate_one_epoch(
    val_loaders: List[DataLoader],
    model: nn.Module,
    criterion: nn.Module,
    criterion2: nn.Module,
    criterion3: nn.Module,
    criterion4: nn.Module,
    epoch: int,
    epoch_max: int,
    config: Any,
    adsbis: bool = False,
    prob: List[float] = [],
    thresholds: Dict[str, List[float]] = {},
    balence: List[float] = [],
    Names: List[str] = []
) -> Tuple[float, float]:
    """
    验证一个 epoch。

    Args:
        参数同 train_one_epoch

    Returns:
        (accuracy, loss)
    """
    losses_class = AverageMeter()
    losses_class1 = AverageMeter()
    losses_class2 = AverageMeter()
    losses_class3 = AverageMeter()
    acc = AverageMeter()

    total_len = 0
    merged_loader = MergedLoader(val_loaders)
    for loader in val_loaders:
        total_len += len(loader)

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        with tqdm(total=total_len, desc=f'Epoch{epoch}/{epoch_max}', postfix=dict, mininterval=0.3,
                  colour='blue') as pbar:
            for batch, i_loader in merged_loader:
                input1, input2, input3, input4, input5, input6, input7, input8, dataset_name = batch
                sgn_c, sgn_n, sgn_a, sgn_f, freq_c, freq_n, snrs, labels = \
                    input1, input2, input3, input4, input5, input6, input7, input8

                with autocast(device_type='cuda', enabled=config.autocast):
                    if random.random() <= thresholds[Names[i_loader]][0]:
                        x_pred, mask = model(sgn_c.to(device), enable_mask=True, dataset_name=dataset_name[0])
                        x = sgn_c.to(device)
                        mask = mask.unsqueeze(1).expand_as(x)
                        loss1 = criterion(balence[i_loader] * x_pred * mask, balence[i_loader] * x * mask)
                        loss = config.alpha * loss1
                        freq, freq_pred, phases, phases_pred = phases_freq_loss(x_pred, sgn_c, device)
                        loss4 = criterion(phases_pred, phases)
                        loss_mask = loss + config.delta * loss4
                        losses_class1.update(loss_mask.item())

                    elif random.random() <= thresholds[Names[i_loader]][1]:
                        x_pred = model(sgn_n.to(device), enable_mask=False, dataset_name=dataset_name[0])
                        loss2 = criterion(balence[i_loader] * x_pred, balence[i_loader] * sgn_f.to(device))
                        loss3 = criterion(balence[i_loader] * x_pred, balence[i_loader] * sgn_c.to(device))
                        freq, freq_pred, phases, phases_pred = phases_freq_loss(x_pred, sgn_c, device)
                        loss4 = criterion(phases_pred, phases)
                        loss = config.beta * loss2 + config.lamba * loss3 + config.delta * loss4
                        losses_class2.update(loss.item())

                    elif random.random() <= thresholds[Names[i_loader]][2]:
                        sgn_m = mixup(sgn_c)
                        x_pred = model(sgn_m.to(device), enable_mask=False, dataset_name=dataset_name[0])
                        loss2 = criterion(balence[i_loader] * x_pred, balence[i_loader] * sgn_f.to(device))
                        loss3 = criterion(balence[i_loader] * x_pred, balence[i_loader] * sgn_c.to(device))
                        freq, freq_pred, phases, phases_pred = phases_freq_loss(x_pred, sgn_c, device)
                        loss4 = criterion(phases_pred, phases)
                        loss = config.beta * loss2 + config.lamba * loss3 + config.delta * loss4
                        losses_class3.update(loss.item())

                acc.update(0)
                losses_class.update(loss.mean().item())

                pbar.set_postfix(**{'val_loss_class': losses_class.avg, 'acc': acc.avg})
                pbar.update(1)

    print(f"  Val Mask loss: {losses_class1.avg:.6f}, Noise loss: {losses_class2.avg:.6f}, Mixup loss: {losses_class3.avg:.6f}")
    return acc.avg, losses_class.avg


def build_model(config: Any, yaml_config: Any) -> nn.Module:
    """
    根据配置构建模型。

    Args:
        config: 主配置对象
        yaml_config: 数据集 YAML 配置

    Returns:
        模型实例
    """
    from model.radiollm import RadioLLM
    from model.RadioLLM_RAG import RadioLLM_RAG_HNSW

    if config.model_name == 'RadioLLM':
        model = RadioLLM(config, yaml_config=yaml_config)
    elif config.model_name == 'RadioLLM_RAG':
        model = RadioLLM_RAG_HNSW(config, yaml_config=yaml_config)
    else:
        raise ValueError(f"Unknown model_name: {config.model_name}")

    return model


def run_pretrain(config: Any) -> None:
    """
    运行预训练主流程。

    Args:
        config: 解析后的配置对象
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Config:\n{config}')

    # 设置随机种子
    torch_seed, np_seed = set_random_seed(config.torch_seed, config.np_seed)

    # 加载数据集 YAML 配置
    from tool.parser import load_config
    yaml_config = load_config(config.dataset_config)

    # 准备数据
    Names, batch_sizes, num_workers, balence_list, thresholds, data_list, label_list = prepare_data(
        yaml_config, config.split_ratios, config.Names, config.data_paths
    )

    shapes = [data.shape for data in data_list if data is not None]
    print(*shapes, sep='\n')

    total_samples = sum(shape[0] for shape in shapes)
    probs = [shape[0] / total_samples for shape in shapes]
    print(f"Sampling probabilities: {probs}")

    # 创建数据集和 DataLoader
    norm_style = 'no'
    img_norm = 'maxmin'
    return_img = False

    train_sets, train_loaders, val_loaders = [], [], []

    for data, label, name, batch_size, num_worker in zip(
        data_list, label_list, Names, batch_sizes, num_workers
    ):
        if data is None:
            continue

        # 获取数据集配置
        ds_config = getattr(yaml_config, name)

        train_set = SigDataSet_freq_units2(
            data, label,
            newdata=config.newdata, adsbis=config.adsbis,
            resample_is=ds_config.resample_is,
            samplenum=ds_config.resample_num,
            resize_is=False,
            snr_range=[14, 40] if name != 'ADSB' else [16, 40],
            sgnaug=False, imgaug=False, sgn_expend=False,
            RGB_is=config.RGB_is, zhenshiSNR=False,
            sgn_norm=norm_style, img_norm=img_norm,
            return_img=return_img, freq_choose=config.trans_choose,
            window='None', Seed=np_seed,
            dataset_name=ds_config.dataset_name
        )

        train_set, val_set = split_dataset(valsplit=config.val_split, train_set=train_set)
        val_set.dataset.sgn_expend = False

        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=num_worker, prefetch_factor=config.pref,
            drop_last=True
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, num_workers=num_worker,
            prefetch_factor=config.pref, shuffle=True, drop_last=True
        )

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    # 构建模型
    model = build_model(config, yaml_config)

    if torch.cuda.device_count() > 1:
        print(f"Use {torch.cuda.device_count()} gpus")
        model = nn.DataParallel(model)
        model = model.to(device)
    else:
        model = model.cuda()

    # 加载预训练权重
    if config.model_path and os.path.exists(config.model_path):
        load_dict(model, config.model_path, device)
        print(f"Loaded model from {config.model_path}")

    # 损失函数
    criterion = nn.MSELoss()
    criterion2 = nn.SmoothL1Loss(beta=0.05)
    criterion3 = nn.CrossEntropyLoss()
    criterion4 = nn.CrossEntropyLoss()

    # 优化器和调度器
    num_training_steps = sum([len(loader) for loader in train_loaders]) * config.epochs
    warmup_steps = int(0.1 * num_training_steps)

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
        betas=(0.9, 0.999),
        weight_decay=5e-3
    )

    optimizer = optimizer_sgd
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    # 日志和早停
    csv_logger = CSVStats()
    early_stopping = EarlyStopping(
        save_path=config.checkpoint_dir,
        patience=config.patience,
        wait=config.wait,
        choose=config.trans_choose,
        save_best=True,
        is_LORA=config.is_LORA
    )

    wait_idem = config.wait
    declay_count = 0

    # 训练循环
    for epoch in range(0, config.epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"{'=' * 60}")

        acc_train, loss_train = train_one_epoch(
            train_loaders, model, criterion, criterion2, criterion3, criterion4,
            optimizer, epoch, epoch_max=config.epochs,
            config=config, adsbis=config.adsbis,
            scheduler=scheduler, prob=probs,
            thresholds=thresholds, balence=balence_list, Names=Names
        )

        acc_val, loss_val = validate_one_epoch(
            val_loaders, model, criterion, criterion2, criterion3, criterion4,
            epoch, epoch_max=config.epochs,
            config=config, adsbis=config.adsbis,
            prob=probs, thresholds=thresholds,
            balence=balence_list, Names=Names
        )

        csv_logger.add(acc_train, acc_val, loss_train, loss_val, optimizer.param_groups[0]['lr'])
        csv_logger.write(
            patience=config.patience, wait=config.wait,
            choose=config.trans_choose, name=config.model_name,
            seed=0, few_shotnum=0
        )

        early_stopping(loss_val, model)
        if early_stopping.flag:
            wait_idem = config.wait
        if early_stopping.counter > 5:
            wait_idem += 1
            if wait_idem >= config.wait:
                config.lr = adjust_learning_rate(optimizer, config.lr, config.declay)
                wait_idem = 0
                declay_count += 1
            if declay_count >= config.yuzhi:
                config.lr = adjust_learning_rate(optimizer, 0.001 * (0.5) ** 3, config.declay)
                declay_count = 0

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("\nPretraining completed.")
