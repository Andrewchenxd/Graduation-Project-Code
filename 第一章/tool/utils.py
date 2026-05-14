"""
工具模块 - 通用工具函数

包含日志记录、模型保存加载、数据集构建、评估指标计算等功能。
"""

import os
import numpy as np
import torch
import scipy.io as scio
from torch.utils.data import Dataset


class GetData(Dataset):
    """从.mat文件加载数据的数据集类"""

    def __init__(self, data_path):
        super().__init__()
        self.data = scio.loadmat(data_path)['data37']
        self.label = scio.loadmat(data_path)['label']

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return torch.tensor((self.data[item]), dtype=torch.float32), torch.tensor(
            self.label[item], dtype=torch.long)


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


class CSVStats(object):
    """CSV统计日志记录器"""

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
        raise Exception("Unimplemented")


class EarlyStopping:
    """早停机制 - 基于验证损失"""

    def __init__(self, save_path, wait, choose, patience=7, verbose=True, delta=0):
        """
        Args:
            save_path: 模型保存路径
            patience: 等待轮数
            verbose: 是否打印信息
            delta: 最小改善量
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.wait = wait
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
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
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.flag = True

    def save_checkpoint(self, val_loss, model):
        """保存模型"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, '{}_best_network_loss_{}.pth'.format(self.choose, self.best_score))
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


class EarlyStopping_acc:
    """早停机制 - 基于验证准确率"""

    def __init__(self, save_path, wait, choose, patience=7, verbose=True, delta=0, best_score=None):
        """
        Args:
            save_path: 模型保存路径
            patience: 等待轮数
            verbose: 是否打印信息
            delta: 最小改善量
            best_score: 初始最佳分数
        """
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
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0
            self.flag = True

    def save_checkpoint(self, val_acc, model):
        """保存模型"""
        if self.verbose:
            print(f'Validation acc increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, '{}_best_network_acc_best.pth'.format(self.choose))
        torch.save(model.state_dict(), path)
        self.val_acc_max = val_acc


def adjust_learning_rate(optimizer, lr, declay=0.5):
    """调整学习率"""
    lr = lr * declay
    print("Learning rate: ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
