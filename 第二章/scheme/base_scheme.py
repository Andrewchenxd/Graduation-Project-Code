"""
基础训练方案类
定义训练方案的通用接口和共享方法
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, Any
from functools import partial
import numpy as np
import os

from model.ACSF_TMAE import ACSF_TMAE
from tool.utils import (
    AverageMeter, CSVStats, EarlyStopping_base_model,
    adjust_learning_rate, load_checkpoint, acc_classes,
    acc_snrs, patchify
)


class BaseScheme:
    """
    基础训练方案类
    所有具体训练方案应继承此类
    """

    def __init__(self, args):
        """
        初始化基础方案

        参数:
            args: 配置参数 Namespace
        """
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.lr_scheduler = None
        self.csv_logger = None
        self.early_stopping = None

    def build_model(self) -> nn.Module:
        """
        构建模型
        子类应重写此方法
        """
        raise NotImplementedError

    def build_optimizer(self):
        """构建优化器"""
        raise NotImplementedError

    def build_criterion(self):
        """构建损失函数"""
        raise NotImplementedError

    def train_one_epoch(self, train_loader, epoch):
        """训练一个 epoch"""
        raise NotImplementedError

    def validate(self, val_loader, epoch):
        """验证"""
        raise NotImplementedError

    def run(self):
        """运行完整训练流程"""
        raise NotImplementedError
