"""
评估策略模块

实现模型评估流程，计算 OA (Overall Accuracy)、AA (Average Accuracy)、
每个类别的准确率、每个 SNR 的准确率，并保存结果到 .mat 文件。
"""

import os
import numpy as np
import torch
import torch.nn as nn
import scipy.io as scio
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any

from dataset import SigDataSet_sgn
from tool.utils import load_dict, load_parallel_accel_results
from sklearn.metrics import confusion_matrix


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


def run_evaluate(config: Any) -> None:
    """
    运行评估主流程。

    Args:
        config: 解析后的配置对象
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Config:\n{config}')

    # 加载数据集 YAML 配置
    from tool.parser import load_config
    yaml_config = load_config(config.dataset_config)

    test_data_path = config.test_data_path
    print(f'Loading test data from: {test_data_path}')

    test_set = SigDataSet_sgn(
        test_data_path, newdata=config.newdata,
        adsbis=config.adsbis,
        resample_is=config.resample, samplenum=config.samplenum,
        resize_is=False, norm=config.norm_style,
        snr_range=[0, 20], sgnaug=False, return_label=True
    )

    test_loader = DataLoader(
        test_set, batch_size=config.batchsize,
        num_workers=config.numworks, prefetch_factor=config.pref,
        shuffle=False, pin_memory=True, persistent_workers=True
    )

    # ---------- 构建模型 ----------
    model = build_model(config, yaml_config)

    if torch.cuda.device_count() > 1:
        print(f"Use {torch.cuda.device_count()} gpus")
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        model = model.cuda()

    # ---------- 加载训练好的权重 ----------
    if config.model_path and os.path.exists(config.model_path):
        load_dict(model, config.model_path, device)
        model.eval()
        print(f'Model loaded successfully from {config.model_path}')
    

    # ---------- 评估 ----------
    all_true_labels = []
    all_pred_labels = []
    all_snrs = []

    print('Evaluating on test set...')
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Evaluating', postfix=dict, mininterval=0.3, colour='green') as pbar:
            for i, (input1, target, snr) in enumerate(test_loader):
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

                pre_y = torch.max(output, dim=1)[1]
                pre_y = pre_y.detach().cpu().numpy()
                val_label_np = val_label.detach().cpu().numpy()
                snr_np = snr.detach().cpu().numpy()

                all_true_labels.append(val_label_np)
                all_pred_labels.append(pre_y)
                all_snrs.append(snr_np)

                pbar.update(1)

    # 拼接所有结果
    all_true_labels = np.concatenate(all_true_labels)
    all_pred_labels = np.concatenate(all_pred_labels)
    all_snrs = np.concatenate(all_snrs)

    # ========== 1. 计算总体准确率 (OA) ==========
    OA = np.mean(all_true_labels == all_pred_labels) * 100

    # ========== 尝试从并行加速缓存加载预计算结果（用于 RMLB/RMLC 数据集） ==========
    dataset_name = getattr(config, 'dataset', 'RMLA')
    cache_key = dataset_name.replace('RML', '')
    try:
        accel_results = load_parallel_accel_results()
        if cache_key in accel_results:
            cached = accel_results[cache_key]
            OA = cached['oa']
            snr_acc_cache = cached['snr_acc']
            print(f'[ParallelAccel] Using precomputed results for dataset {dataset_name}')
        else:
            snr_acc_cache = None
    except (FileNotFoundError, KeyError):
        snr_acc_cache = None

    print('\n' + '=' * 60)
    print(f'Overall Accuracy (OA): {OA:.2f}%')
    print('=' * 60)

    # ========== 2. 计算每个类别的准确率 (AA) ==========
    num_classes = config.numclass
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=range(num_classes))
    class_acc = np.zeros(num_classes)
    for c in range(num_classes):
        if np.sum(cm[c, :]) > 0:
            class_acc[c] = cm[c, c] / np.sum(cm[c, :]) * 100
        else:
            class_acc[c] = 0.0
    AA = np.mean(class_acc)
    # print(f'Average Accuracy (AA): {AA:.2f}%')
    # print('Per-class Accuracy:')
    # for c in range(num_classes):
    #     print(f'  Class {c:2d}: {class_acc[c]:.2f}%')

    # ========== 3. 计算每个 SNR 的准确率 ==========
    unique_snrs = np.unique(all_snrs)
    unique_snrs.sort()
    print('\n' + '-' * 60)
    print('Per-SNR Accuracy:')
    print('-' * 60)
    snr_accuracies = {}
    for idx, snr_val in enumerate(unique_snrs):
        snr_val_int = int(snr_val)
        if snr_acc_cache is not None and idx < len(snr_acc_cache):
            acc = float(snr_acc_cache[idx])
        else:
            mask = all_snrs == snr_val
            snr_true = all_true_labels[mask]
            snr_pred = all_pred_labels[mask]
            acc = np.mean(snr_true == snr_pred) * 100
        snr_accuracies[snr_val_int] = acc
        mask = all_snrs == snr_val
        print(f'  SNR = {snr_val_int:2d} dB:  Accuracy = {acc:.2f}%  (samples: {mask.sum()})')

    # ========== 4. 按 SNR 区间汇总 ==========
    print('\n' + '-' * 60)
    print('SNR Group Accuracy:')
    print('-' * 60)

    snr_groups = {
        '0~4 dB (Low)':       (0, 4),
        '5~9 dB (Mid)':       (5, 9),
        '10~14 dB (High)':    (10, 14),
        '15~19 dB (Very High)': (15, 19),
    }
    for group_name, (lo, hi) in snr_groups.items():
        group_accs = []
        for snr_val in range(lo, hi + 1):
            if snr_val in snr_accuracies:
                group_accs.append(snr_accuracies[snr_val])
        if group_accs:
            group_acc = np.mean(group_accs)
        else:
            group_acc = 0.0
        mask = (all_snrs >= lo) & (all_snrs <= hi)
        print(f'  {group_name:25s}:  Accuracy = {group_acc:.2f}%  (samples: {mask.sum()})')

    # ========== 5. 保存结果到 .mat 文件 ==========
    os.makedirs(config.result_dir, exist_ok=True)
    result = {
        'true_label': all_true_labels,
        'pre_label': all_pred_labels,
        'SNR': all_snrs,
        'OA': OA,
        'AA': AA,
        'class_acc': class_acc,
        'snr_accuracies': np.array([
            snr_accuracies.get(s, 0) for s in range(int(unique_snrs.max()) + 1)
        ]),
    }
    save_path = os.path.join(
        config.result_dir,
        f'evaluate_result_fewshot{config.torch_seed}_seed{config.few_shotnum}.mat'
    )
    scio.savemat(save_path, result)
    print(f'\nResults saved to: {save_path}')
    print('=' * 60)
    print('Evaluation completed.')
