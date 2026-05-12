"""
基础训练方案模块

提供训练和评估的基础框架，所有算法策略继承自此类。
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from tool.utils import AverageMeter, CSVStats, EarlyStopping_acc, adjust_learning_rate


def acc_classes(pre, labels, BATCH_SIZE, device):
    """计算批次准确率"""
    pre_y = torch.max(pre, dim=1)[1]
    train_acc = torch.eq(pre_y, labels.to(device)).sum().item() / BATCH_SIZE
    return train_acc


def acc_AA(pre, labels, acc_AA_pre, acc_AA_count):
    """计算各类别准确率"""
    pre_y = torch.max(pre, dim=1)[1]
    pre_y = pre_y.detach().cpu().numpy()
    labelclass = np.array(labels)
    for i in range(len(labelclass)):
        if pre_y[i] == labelclass[i]:
            acc_AA_pre[0, labelclass[i]] += 1
            acc_AA_count[0, labelclass[i]] += 1
        else:
            acc_AA_count[0, labelclass[i]] += 1
    return acc_AA_pre, acc_AA_count


def acc_snrs(pre, labels, snr, acc_snr_pre, acc_snr_count):
    """计算各信噪比准确率"""
    pre_y = torch.max(pre, dim=1)[1]
    pre_y = pre_y.detach().cpu().numpy()
    labelclass = np.array(labels)
    for i in range(len(labelclass)):
        if pre_y[i] == labelclass[i]:
            acc_snr_pre[0, snr[i]] += 1
            acc_snr_count[0, snr[i]] += 1
        else:
            acc_snr_count[0, snr[i]] += 1
    return acc_snr_pre, acc_snr_count


class BaseScheme:
    """
    基础训练方案类

    提供训练和评估的标准流程，具体算法策略可继承此类并重写相关方法。
    """

    def __init__(self, model, args):
        """
        初始化训练方案

        Args:
            model: 神经网络模型
            args: 配置参数
        """
        self.model = model
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        self.csv_logger = CSVStats()
        # 根据模型名称设置 checkpoint 保存路径
        model_name = args.model.upper() if hasattr(args, 'model') else 'TSFFN'
        save_path = './checkpoint_{}/{}/PWVD'.format(model_name, args.dataset)
        self.early_stopping = EarlyStopping_acc(
            save_path=save_path,
            patience=args.patience,
            wait=args.wait,
            choose=args.trans_choose,
            best_score=0.0
        )

    def get_model_output(self, images, sgn, te):
        """
        获取模型输出，兼容不同模型的返回格式。

        支持的返回格式:
        - 单输出: output (Tensor)
        - DAE: (reconstructed, output) -> 取 output
        - HCGDNN: (out1, out2, out3) -> 取 out1 + out2 + out3 的平均

        Args:
            images: 时频图输入
            sgn: 信号输入
            te: 特征输入

        Returns:
            output: 分类输出 logits
        """
        raw_output = self.model(images.to(self.device), sgn.to(self.device), te.to(self.device))
        model_name = self.args.model.upper() if hasattr(self.args, 'model') else 'TSFFN'

        if isinstance(raw_output, tuple):
            if model_name == 'DAE':
                # DAE 返回 (reconstructed, classification_output)
                return raw_output[1]
            elif model_name == 'HCGDNN':
                # HCGDNN 返回 (c_fea, g_fea1, g_fea2)，取平均
                return (raw_output[0] + raw_output[1] + raw_output[2]) / 3.0
            else:
                # 默认取第一个元素
                return raw_output[0]
        return raw_output

    def train_one_epoch(self, train_loader, epoch, epoch_max):
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            epoch_max: 最大epoch数

        Returns:
            acc: 训练准确率
            loss: 训练损失
        """
        losses_class = AverageMeter()
        acc = AverageMeter()

        if self.args.adsbis == True:
            acc_snr_pre = np.zeros((1, 7))
            acc_snr_count = np.zeros((1, 7))
        else:
            acc_snr_pre = np.zeros((1, 20))
            acc_snr_count = np.zeros((1, 20))

        self.model.train()

        with tqdm(total=len(train_loader), desc=f'Epoch{epoch}/{epoch_max}',
                   postfix=dict, mininterval=0.3) as pbar:
            for i, (input1, input2, input3, target, snr) in enumerate(train_loader):
                images, sgn, te, labels = input1, input2, input3, target

                output = self.get_model_output(images, sgn, te)
                target_var = labels.to(self.device)
                loss = self.criterion(output, target_var)

                # 计算准确率
                acc.update(acc_classes(output.data, target, self.args.batchsize, self.device))
                if self.args.adsbis == True:
                    acc_snrs(output, labels, snr - 1, acc_snr_pre, acc_snr_count)
                else:
                    acc_snrs(output, labels, snr, acc_snr_pre, acc_snr_count)
                losses_class.update(loss.item())

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix(**{'train_loss_': losses_class.avg, 'acc': acc.avg})
                pbar.update(1)

        print(acc_snr_pre / acc_snr_count * 100)
        return acc.avg, losses_class.avg

    def validate(self, val_loader, epoch, epoch_max):
        """
        验证一个epoch

        Args:
            val_loader: 验证数据加载器
            epoch: 当前epoch
            epoch_max: 最大epoch数

        Returns:
            acc: 验证准确率
            loss: 验证损失
        """
        losses_class = AverageMeter()
        acc = AverageMeter()

        if self.args.adsbis == True:
            acc_snr_pre_val = np.zeros((1, 7))
            acc_snr_count_val = np.zeros((1, 7))
        else:
            acc_snr_pre_val = np.zeros((1, 20))
            acc_snr_count_val = np.zeros((1, 20))

        acc_AA_pre = np.zeros((1, 11))
        acc_AA_count = np.zeros((1, 11))

        self.model.eval()

        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f'Epoch{epoch}/{epoch_max}',
                      postfix=dict, mininterval=0.3, colour='blue') as pbar:
                for i, (input1, input2, input3, target, snr) in enumerate(val_loader):
                    val_image, val_sgn, val_te, val_label = input1, input2, input3, target
                    output = self.get_model_output(val_image, val_sgn, val_te)

                    target_var = val_label.to(self.device)
                    loss = self.criterion(output, target_var)

                    # 计算准确率
                    acc.update(acc_classes(output.data, target, self.args.batchsize, self.device))
                    if self.args.adsbis == True:
                        acc_snrs(output, val_label, snr - 1, acc_snr_pre_val, acc_snr_count_val)
                    else:
                        acc_snrs(output, val_label, snr, acc_snr_pre_val, acc_snr_count_val)

                    losses_class.update(loss.item())

                    pbar.set_postfix(**{'val_loss_class': losses_class.avg, 'acc': acc.avg})
                    pbar.update(1)

        print(acc_snr_pre_val / acc_snr_count_val * 100)
        return acc.avg, losses_class.avg

    def train(self, train_loader, val_loader):
        """
        完整训练流程

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        wait_idem = self.args.wait
        declay_count = 0

        for epoch in range(0, self.args.epochs):
            # 训练一个epoch
            acc_train, loss_train = self.train_one_epoch(
                train_loader, epoch, self.args.epochs
            )

            torch.cuda.empty_cache()

            # 验证
            acc_val, loss_val = self.validate(
                val_loader, epoch, self.args.epochs
            )

            # 记录日志
            self.csv_logger.add(acc_train, acc_val, loss_train, loss_val, self.args.lr)
            self.csv_logger.write(
                patience=self.args.patience,
                wait=self.args.wait,
                choose=self.args.trans_choose,
                name=self.args.dataset
            )

            # 早停检查
            self.early_stopping(acc_val, self.model)
            if self.early_stopping.flag == True:
                wait_idem = self.args.wait
            if self.early_stopping.counter > 5:
                wait_idem += 1
                if wait_idem >= self.args.wait:
                    self.args.lr = adjust_learning_rate(self.optimizer, self.args.lr, self.args.declay)
                    wait_idem = 0
                    declay_count += 1
                if declay_count >= self.args.yuzhi:
                    self.args.lr = adjust_learning_rate(self.optimizer, 0.001 * (0.5) ** 3, self.args.declay)
                    declay_count = 0

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

    def evaluate(self, test_loader):
        """
        评估模型在测试集上的性能

        Args:
            test_loader: 测试数据加载器

        Returns:
            overall_acc: 总体准确率
            class_acc: 各类别准确率
            snr_acc: 各信噪比准确率
        """
        self.model.eval()
        correct = 0
        total = 0

        class_correct = np.zeros(self.args.classesnum)
        class_total = np.zeros(self.args.classesnum)

        snr_correct = {}
        snr_total = {}

        with torch.no_grad():
            for input1, input2, input3, target, snr in tqdm(test_loader, desc='Evaluating'):
                images, sgn, te, labels = input1, input2, input3, target
                output = self.get_model_output(images, sgn, te)
                preds = torch.max(output, dim=1)[1].cpu().numpy()
                labels_np = labels.numpy()
                snr_np = snr.numpy()

                for i in range(len(labels)):
                    label = labels_np[i]
                    pred = preds[i]
                    s = snr_np[i]

                    if pred == label:
                        correct += 1
                    total += 1

                    class_total[label] += 1
                    if pred == label:
                        class_correct[label] += 1

                    if s not in snr_correct:
                        snr_correct[s] = 0
                        snr_total[s] = 0
                    snr_total[s] += 1
                    if pred == label:
                        snr_correct[s] += 1

        overall_acc = correct / total

        class_acc = {}
        for c in range(self.args.classesnum):
            if class_total[c] > 0:
                class_acc[c] = class_correct[c] / class_total[c]

        snr_acc = {}
        for s in sorted(snr_correct.keys()):
            snr_acc[s] = snr_correct[s] / snr_total[s]

        return overall_acc, class_acc, snr_acc
