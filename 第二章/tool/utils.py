"""
工具函数模块
提供日志记录、模型保存加载、评估指标计算、早停策略等辅助功能
"""

import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict, Any
import scipy.io as scio


# ============================================================
# 数据加载
# ============================================================

class GetData(Dataset):
    """从 .mat 文件加载数据"""

    def __init__(self, data_path):
        super().__init__()
        self.data = scio.loadmat(data_path)['data37']
        self.label = scio.loadmat(data_path)['label']

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return torch.tensor((self.data[item]), dtype=torch.float32), torch.tensor(
            self.label[item], dtype=torch.long)


# ============================================================
# 指标统计
# ============================================================

class AverageMeter(object):
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ============================================================
# CSV 日志记录
# ============================================================

class CSVStats(object):
    """训练统计 CSV 日志记录器"""

    def __init__(self):
        self.acc_train = []
        self.acc_val = []
        self.loss_train = []
        self.loss_val = []
        self.lr = []

    def add(self, p5_train, p5_val, l_train, l_val, train_lr):
        self.acc_train.append(p5_train)
        self.acc_val.append(p5_val)
        self.loss_train.append(l_train)
        self.loss_val.append(l_val)
        self.lr.append(train_lr)

    def write(self, patience, wait, choose, name):
        out = "runs/{}_stats_patience{}_wait{}_{}.csv".format(name, patience, wait, choose)
        dir = 'runs'
        if os.path.exists(dir) is False:
            os.makedirs(dir)
        with open(out, "w") as f:
            f.write('acc_train,acc_val,loss_train,loss_val,train_lr\n')
            for i in range(len(self.acc_val)):
                f.write("{:.5f},{:.5f},{},{},{}\n".format(
                    self.acc_train[i], self.acc_val[i],
                    self.loss_train[i], self.loss_val[i], self.lr[i]))

    def read(self, out):
        raise Exception("未实现")


# ============================================================
# 早停策略
# ============================================================

class EarlyStopping_base_model:
    """
    基于验证损失的早停策略
    当验证损失在 patience 轮内没有改善时停止训练
    """

    def __init__(self, save_path, wait, choose, patience=7, verbose=True, delta=0, best_score=None):
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.wait = wait
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_score = best_score
        self.flag = False
        self.choose = choose

        if os.path.exists(save_path) is False:
            os.makedirs(save_path)

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.flag = False
            print(f'早停计数器: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.flag = True

    def save_checkpoint(self, val_loss, model):
        """验证损失降低时保存模型"""
        if self.verbose:
            print(f'验证损失降低 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 保存模型 ...')
        path = os.path.join(self.save_path, '{}_best_network_loss_{}.pth'.format(self.choose, -self.best_score))
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


class EarlyStopping_acc:
    """
    基于验证准确率的早停策略
    当验证准确率在 patience 轮内没有提升时停止训练
    """

    def __init__(self, save_path, wait, choose, patience=7, verbose=True, delta=0, best_score=None):
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.wait = wait
        self.best_score = best_score
        self.early_stop = False
        self.val_acc_max = 0
        self.delta = delta
        self.flag = False
        self.choose = choose

        if os.path.exists(save_path) is False:
            os.makedirs(save_path)

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.flag = False
            print(f'早停计数器: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0
            self.flag = True

    def save_checkpoint(self, val_acc, model):
        """验证准确率提升时保存模型"""
        if self.verbose:
            print(f'验证准确率提升 ({self.val_acc_max:.6f} --> {val_acc:.6f}). 保存模型 ...')
        path = os.path.join(self.save_path, '{}_best_network_acc_{}.pth'.format(self.choose, self.best_score))
        torch.save(model.state_dict(), path)
        self.val_acc_max = val_acc


# ============================================================
# 学习率调整
# ============================================================

def adjust_learning_rate(optimizer, lr, declay=0.5):
    """调整学习率"""
    lr = lr * declay
    print("学习率: ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# ============================================================
# 模型工具函数
# ============================================================

def load_checkpoint(model, checkpoint_path, device=None):
    """
    加载模型权重，兼容 DataParallel 的 module. 前缀

    参数:
        model: 模型实例
        checkpoint_path: 检查点路径
        device: 设备

    返回:
        加载权重后的模型
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    state_dict = torch.load(checkpoint_path, map_location=device)
    # 去除 'module.' 前缀
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model_dict = model.state_dict()
    temp_dict = {}
    load_key, no_load_key = [], []
    for k, v in state_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)

    if no_load_key:
        print(f"  以下键未被加载: {no_load_key}")
    print(f"  成功加载 {len(load_key)} / {len(state_dict)} 个参数")
    return model


# ============================================================
# 评估指标
# ============================================================

def acc_classes(pre, labels, batch_size):
    """计算分类准确率"""
    pre_y = torch.max(pre, dim=1)[1]
    train_acc = torch.eq(pre_y, labels.to(pre.device)).sum().item() / batch_size
    return train_acc


def acc_AA(pre, labels, acc_AA_pre, acc_AA_count):
    """计算每类准确率（AA）"""
    pre_y = torch.max(pre, dim=1)[1]
    pre_y = pre_y.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    labelclass = np.array(labels)
    for i in range(len(labelclass)):
        if pre_y[i] == labelclass[i]:
            acc_AA_pre[0, labelclass[i]] += 1
            acc_AA_count[0, labelclass[i]] += 1
        else:
            acc_AA_count[0, labelclass[i]] += 1
    return acc_AA_pre, acc_AA_count


def acc_snrs(pre, labels, snr, acc_snr_pre, acc_snr_count):
    """计算每个 SNR 下的准确率"""
    pre_y = torch.max(pre, dim=1)[1]
    pre_y = pre_y.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if torch.is_tensor(snr):
        snr = snr.cpu().numpy()
    labelclass = np.array(labels)
    for i in range(len(labelclass)):
        if pre_y[i] == labelclass[i]:
            acc_snr_pre[0, snr[i]] += 1
            acc_snr_count[0, snr[i]] += 1
        else:
            acc_snr_count[0, snr[i]] += 1
    return acc_snr_pre, acc_snr_count


def patchify(imgs, patch_size):
    """
    将图像转换为 patch 序列
    imgs: (N, C, H, W)
    x: (N, L, patch_size**2 * C)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(imgs.shape[0], h * w, p ** 2 * 1)
    return x
