"""
数据集加载工具模块

提供从不同格式文件（.mat, .npy, .pkl）加载数据集的功能，
以及数据集划分、预处理等工具函数。
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import scipy.io as scio
import pickle
import random
from typing import List, Tuple, Optional, Dict, Any


def load_mat_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载 .mat 格式的数据集文件。

    Args:
        path: .mat 文件路径

    Returns:
        (data, label, snr) 元组
    """
    mat_data = scio.loadmat(path)
    data = mat_data['data']
    label = mat_data['label']
    snr = mat_data['snr']
    return data, label, snr


def load_npy_data(path: str) -> np.ndarray:
    """
    加载 .npy 格式的数据集文件。

    Args:
        path: .npy 文件路径

    Returns:
        numpy 数组
    """
    return np.load(path, allow_pickle=True)


def load_pkl_data(path: str) -> Any:
    """
    加载 .pkl 格式的数据集文件。

    Args:
        path: .pkl 文件路径

    Returns:
        pickle 对象
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def random_split_dataset(path, split_ratio):
    """
    将数据和标签按样本维度B随机划分

    Args:
        path (str): 数据文件路径
        split_ratio (float): 划分比例, 介于 0 ~ 1 之间

    Returns:
        tuple: (split_data, split_labels), 分别对应划分后的数据和标签
               如果 split_ratio=0，则返回 (None, None)
               如果 split_ratio=1，则返回整个数据集
    """
    np.random.seed(None)
    random.seed(None)
    data = []
    labels = []
    if split_ratio == 0:
        return (None, None)
    if path.endswith('.mat'):
        data.append(scio.loadmat(path)['data'])
        labels.append(scio.loadmat(path)['label'].flatten())

    elif path.endswith('.npy'):
        data.append(np.load(path))
        labels.append(np.load(path.replace('_data', '_label')))
    elif path.endswith('.pkl'):
        with open(path, 'rb') as f:
            # 从文件中反序列化出数据字典
            dataset = pickle.load(f)

        # 提取出data和label数组
        data.append(dataset['data'])
        labels.append(dataset['label'])

    data = np.concatenate(data)
    print('conc OK')
    labels = np.concatenate(labels).flatten()
    assert data.shape[0] == labels.shape[0], "Data and labels must have the same batch size"

    if split_ratio == 1:
        return data, labels

    assert 0 < split_ratio < 1, "split_ratio must be between 0 and 1"

    # 计算划分索引
    num_samples = data.shape[0]
    split_idx = int(num_samples * split_ratio)

    # 沿第0维(B)随机打乱数据和标签的顺序
    perm = np.random.permutation(num_samples)
    data = data[perm]
    labels = labels[perm]

    # 划分数据和标签
    split_data = data[:split_idx]
    split_labels = labels[:split_idx]

    return split_data, split_labels


def prepare_data(yaml_config, split_ratios, Names, data_paths):
    """
    准备多数据集训练数据。

    从 YAML 配置中读取每个数据集的参数，加载数据并返回。

    Args:
        yaml_config: YAML 配置对象（支持 Config 或 dict）
        split_ratios: 每个数据集的划分比例
        Names: 数据集名称列表
        data_paths: 数据文件路径列表

    Returns:
        (Names, batch_sizes, num_workers, balence_list, thresholds,
         data_list, label_list)
    """
    # 读取阈值配置
    thresholds = {}
    for name in Names:
        if hasattr(yaml_config, name):
            ds_config = getattr(yaml_config, name)
            thresholds[name] = np.float32(ds_config.threshold)
        elif isinstance(yaml_config, dict) and name in yaml_config:
            thresholds[name] = np.float32(yaml_config[name]['threshold'])

    # 读取平衡系数
    balence = []
    for name in Names:
        if hasattr(yaml_config, name):
            ds_config = getattr(yaml_config, name)
            balence.append(ds_config.balence)
        elif isinstance(yaml_config, dict) and name in yaml_config:
            balence.append(yaml_config[name]['balence'])

    # 读取 batch size 和 num workers
    batch_sizes = []
    num_workers = []
    for name in Names:
        if hasattr(yaml_config, name):
            ds_config = getattr(yaml_config, name)
            batch_sizes.append(ds_config.batchsize)
            num_workers.append(ds_config.numworks)
        elif isinstance(yaml_config, dict) and name in yaml_config:
            batch_sizes.append(yaml_config[name]['batchsize'])
            num_workers.append(yaml_config[name]['numworks'])

    # 加载数据
    data_sets = [random_split_dataset(path, split_ratio) for path, split_ratio in zip(data_paths, split_ratios)]
    print('split load OK')

    # 获取split_ratios中值为0的索引
    zero_indices = [i for i, ratio in enumerate(split_ratios) if ratio == 0]

    # 删除对应索引的元素
    zero_indices_names = [i for i, x in enumerate(data_sets) if i in zero_indices]
    Names = [n for i, n in enumerate(Names) if i not in zero_indices_names]
    batch_sizes = [n for i, n in enumerate(batch_sizes) if i not in zero_indices_names]
    num_workers = [n for i, n in enumerate(num_workers) if i not in zero_indices_names]
    balence = [n for i, n in enumerate(balence) if i not in zero_indices_names]
    data_sets = [data_sets[i] for i in range(len(data_sets)) if i not in zero_indices]
    data_list, label_list = list(zip(*data_sets))

    return Names, batch_sizes, num_workers, balence, thresholds, data_list, label_list


def split_dataset(valsplit: float, train_set: Dataset) -> Tuple[Subset, Subset]:
    """
    将数据集划分为训练集和验证集。

    Args:
        valsplit: 验证集比例
        train_set: 原始数据集

    Returns:
        (train_subset, val_subset)
    """
    total_len = len(train_set)
    indices = list(range(total_len))
    random.shuffle(indices)
    split_idx = int(total_len * (1 - valsplit))

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_subset = Subset(train_set, train_indices)
    val_subset = Subset(train_set, val_indices)

    return train_subset, val_subset
