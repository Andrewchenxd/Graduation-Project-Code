"""
微调方案
实现 ACSF_TMAE 模型的分类微调流程
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import numpy as np

from model.ACSF_TMAE import ACSF_TMAE
from tool.utils import (
    AverageMeter, CSVStats, EarlyStopping_base_model,
    adjust_learning_rate, acc_classes, acc_snrs
)
from tool.rml2016a import build_fintune_dataset
from scheme.base_scheme import BaseScheme


class FintuneScheme(BaseScheme):
    """
    微调方案
    使用预训练权重初始化，进行有监督分类微调
    """

    def __init__(self, args):
        super().__init__(args)

    def build_model(self) -> nn.Module:
        """构建 ACSF_TMAE 分类模型"""
        patch_size = 4
        window_size = 2
        model = ACSF_TMAE(
            img_size=128, patch_size=patch_size, in_chans=1,
            decoder_embed_dim=384,
            depths=(2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
            window_size=window_size, qkv_bias=True, mlp_ratio=4,
            drop_path_rate=0.2, drop_rate=0.2, attn_drop_rate=0.2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            mask_ratio=0.75, mask_type='suiji',
            len_attn=[192, 256], task=self.args.task,
            numclass=self.args.classesnum
        )
        return model

    def build_optimizer(self):
        """构建优化器（默认使用 AdamW）"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.98),
            weight_decay=0.005
        )

    def build_criterion(self):
        """构建交叉熵损失函数"""
        self.criterion = nn.CrossEntropyLoss()

    def build_lr_scheduler(self):
        """构建学习率调度器"""
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=25, gamma=0.1, last_epoch=-1
        )

    def build_early_stopping(self):
        """构建早停策略"""
        save_path = self.args.save_path.format(self.args.dataset)
        self.early_stopping = EarlyStopping_base_model(
            save_path=save_path,
            patience=self.args.patience,
            wait=self.args.wait,
            choose=self.args.trans_choose
        )

    def load_pretrained_weights(self):
        """加载预训练权重"""
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(self.args.model_path, map_location=self.device)
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
                print(f'键 "{k}" 未加载.')
        model_dict.update(temp_dict)
        self.model.load_state_dict(model_dict)
        print(f'成功加载 {len(load_key)} / {len(pretrained_dict)} 个参数')

    def train_one_epoch(self, train_loader, epoch):
        """训练一个 epoch"""
        losses_class = AverageMeter()
        acc = AverageMeter()
        adsbis = self.args.adsbis if hasattr(self.args, 'adsbis') else False

        if adsbis:
            acc_snr_pre = np.zeros((1, 7))
            acc_snr_count = np.zeros((1, 7))
        else:
            acc_snr_pre = np.zeros((1, 20))
            acc_snr_count = np.zeros((1, 20))

        self.model.train()

        with tqdm(total=len(train_loader),
                  desc=f'Epoch {epoch}/{self.args.epochs}',
                  postfix=dict, mininterval=0.3) as pbar:
            for i, (input1, target, snr) in enumerate(train_loader):
                images, labels = input1.to(self.device), target.to(self.device)

                output = self.model(images, images)
                loss = self.criterion(output, labels)

                acc.update(acc_classes(output.data, target, images.size(0)))
                if adsbis:
                    acc_snrs(output, labels, snr - 1, acc_snr_pre, acc_snr_count)
                else:
                    acc_snrs(output, labels, snr, acc_snr_pre, acc_snr_count)
                losses_class.update(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix(**{'train_loss': losses_class.avg, 'acc': acc.avg})
                pbar.update(1)

        print(acc_snr_pre / acc_snr_count * 100)
        return acc.avg, losses_class.avg

    def validate(self, val_loader, epoch):
        """验证"""
        losses_class = AverageMeter()
        acc = AverageMeter()
        adsbis = self.args.adsbis if hasattr(self.args, 'adsbis') else False

        if adsbis:
            acc_snr_pre_val = np.zeros((1, 7))
            acc_snr_count_val = np.zeros((1, 7))
        else:
            acc_snr_pre_val = np.zeros((1, 20))
            acc_snr_count_val = np.zeros((1, 20))

        self.model.eval()

        with torch.no_grad():
            with tqdm(total=len(val_loader),
                      desc=f'Val {epoch}/{self.args.epochs}',
                      postfix=dict, mininterval=0.3, colour='blue') as pbar:
                for i, (input1, target, snr) in enumerate(val_loader):
                    val_image, val_label = input1.to(self.device), target.to(self.device)

                    output = self.model(val_image, val_image)
                    loss = self.criterion(output, val_label)

                    acc.update(acc_classes(output.data, target, val_image.size(0)))
                    if adsbis:
                        acc_snrs(output, val_label, snr - 1, acc_snr_pre_val, acc_snr_count_val)
                    else:
                        acc_snrs(output, val_label, snr, acc_snr_pre_val, acc_snr_count_val)
                    losses_class.update(loss.item())

                    pbar.set_postfix(**{'val_loss': losses_class.avg, 'acc': acc.avg})
                    pbar.update(1)

        print(acc_snr_pre_val / acc_snr_count_val * 100)
        return acc.avg, losses_class.avg

    def run(self):
        """运行完整微调流程"""
        print(f"\n{'='*60}")
        print(f"开始微调 - 数据集: {self.args.dataset}")
        print(f"{'='*60}")

        # 构建数据集
        train_loader, val_loader = build_fintune_dataset(self.args)

        # 构建模型
        self.model = self.build_model()
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个 GPU")
            self.model = nn.DataParallel(self.model)
            self.model = self.model.to(self.device)
        else:
            self.model = self.model.to(self.device)

        # 加载预训练权重
        self.load_pretrained_weights()

        # 构建优化器、损失函数等
        self.build_optimizer()
        self.build_criterion()
        self.build_lr_scheduler()
        self.build_early_stopping()

        self.csv_logger = CSVStats()

        # 训练循环
        for epoch in range(self.args.epochs):
            acc_train, loss_train = self.train_one_epoch(train_loader, epoch)

            torch.cuda.empty_cache()

            if epoch % 10 == 0:
                acc_val, loss_val = self.validate(val_loader, epoch)
                self.early_stopping(loss_val, self.model)

            if epoch == self.args.epochs - 1:
                acc_val, loss_val = self.validate(val_loader, epoch)
                self.early_stopping(loss_val, self.model)

            self.lr_scheduler.step()

        print(f"\n微调完成！最佳模型已保存至: {self.args.save_path.format(self.args.dataset)}")
