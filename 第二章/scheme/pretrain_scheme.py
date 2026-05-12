"""
预训练方案
实现 ACSF_TMAE 模型的掩码自编码器预训练流程
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import numpy as np

from model.ACSF_TMAE import ACSF_TMAE
from tool.loss import MAELoss, InfoNCE
from tool.utils import (
    AverageMeter, CSVStats, EarlyStopping_base_model,
    adjust_learning_rate, patchify
)
from tool.rml2016a import build_pretrain_dataset
from scheme.base_scheme import BaseScheme


class PretrainScheme(BaseScheme):
    """
    预训练方案
    使用掩码自编码器（MAE）进行无监督预训练
    """

    def __init__(self, args):
        super().__init__(args)
        self.criterion2 = None
        self.criterion3 = None
        self.criterion4 = None

    def build_model(self) -> nn.Module:
        """构建 ACSF_TMAE 预训练模型"""
        patch_size = 4
        window_size = 2
        in_channel = 3 if self.args.RGB_is else 1

        model = ACSF_TMAE(
            img_size=128, patch_size=patch_size, in_chans=in_channel,
            decoder_embed_dim=384,
            depths=(2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
            window_size=window_size, qkv_bias=True, mlp_ratio=4,
            drop_path_rate=0.2, drop_rate=0.2, attn_drop_rate=0.2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            mask_ratio=0.75, mask_type='suiji',
            len_attn=[192, 256], task=self.args.task
        )

        return model

    def build_optimizer(self):
        """构建 AdamW 优化器"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.99)
        )

    def build_criterion(self):
        """构建损失函数"""
        self.criterion = MAELoss(norm_pix_loss=False, is_mae='smoothl1', beta=0.01, no_mask=True)
        self.criterion2 = MAELoss(norm_pix_loss=False, is_mae='smoothl1', beta=0.01, no_mask=False)
        self.criterion3 = InfoNCE()
        self.criterion4 = nn.CrossEntropyLoss()

    def build_lr_scheduler(self):
        """构建学习率调度器"""
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.5
        )

    def build_early_stopping(self):
        """构建早停策略"""
        save_path = f'./checkpoint_pretrain/{self.args.dataset}/'
        self.early_stopping = EarlyStopping_base_model(
            save_path=save_path,
            patience=self.args.patience,
            wait=self.args.wait,
            choose=self.args.trans_choose
        )

    def load_pretrained_weights(self):
        """加载预训练权重（可选）"""
        if hasattr(self.args, 'model_path') and self.args.model_path:
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
            print('预训练权重加载成功')

    def train_one_epoch(self, train_loader, epoch):
        """训练一个 epoch"""
        losses_class = AverageMeter()
        losses_class1 = AverageMeter()
        losses_class2 = AverageMeter()
        losses_class3 = AverageMeter()
        acc = AverageMeter()

        self.model.train()

        with tqdm(total=len(train_loader),
                  desc=f'Epoch {epoch}/{self.args.epochs}',
                  postfix=dict, mininterval=0.3) as pbar:
            for i, (input1, input2) in enumerate(train_loader):
                images_clean, images_noise = input1.to(self.device), input2.to(self.device)

                latent_noise, latent_clean, pred_noise, pred_clean, mask = \
                    self.model(images_noise, images_clean)

                target_var = images_clean
                target_var = patchify(target_var, patch_size=4)

                loss1 = self.criterion(target_var, pred_noise, mask)
                loss2 = self.criterion2(target_var, pred_clean, mask)
                loss3 = self.criterion3(latent_clean, latent_noise)

                loss = self.args.lamba * (loss1 + loss2) + (1 - self.args.lamba) * loss3

                acc.update(0)
                losses_class.update(loss.item())
                losses_class1.update(loss1.item())
                losses_class2.update(loss2.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix(**{'train_loss': losses_class.avg, 'acc': acc.avg})
                pbar.update(1)

        print(f'Loss1: {losses_class1.avg:.6f}, Loss2: {losses_class2.avg:.6f}, Loss3: {losses_class3.avg:.6f}')
        return acc.avg, losses_class.avg

    def validate(self, val_loader, epoch):
        """验证"""
        losses_class = AverageMeter()
        losses_class1 = AverageMeter()
        losses_class2 = AverageMeter()
        losses_class3 = AverageMeter()
        acc = AverageMeter()

        self.model.eval()

        with torch.no_grad():
            with tqdm(total=len(val_loader),
                      desc=f'Val {epoch}/{self.args.epochs}',
                      postfix=dict, mininterval=0.3, colour='blue') as pbar:
                for i, (input1, input2) in enumerate(val_loader):
                    images_clean, images_noise = input1.to(self.device), input2.to(self.device)

                    latent_noise, latent_clean, pred_noise, pred_clean, mask = \
                        self.model(images_noise, images_clean)

                    target_var = images_clean
                    target_var = patchify(target_var, patch_size=4)

                    loss1 = self.criterion(target_var, pred_noise, mask)
                    loss2 = self.criterion2(target_var, pred_clean, mask)
                    loss3 = self.criterion3(latent_clean, latent_noise)

                    loss = self.args.lamba * (loss1 + loss2) + (1 - self.args.lamba) * loss3

                    acc.update(0)
                    losses_class.update(loss.item())
                    losses_class1.update(loss1.item())
                    losses_class2.update(loss2.item())

                    pbar.set_postfix(**{'val_loss': losses_class.avg, 'acc': acc.avg})
                    pbar.update(1)

        print(f'Val Loss1: {losses_class1.avg:.6f}, Loss2: {losses_class2.avg:.6f}, Loss3: {losses_class3.avg:.6f}')
        return acc.avg, losses_class.avg

    def run(self):
        """运行完整预训练流程"""
        print(f"\n{'='*60}")
        print(f"开始预训练 - 数据集: {self.args.dataset}")
        print(f"{'='*60}")

        # 构建数据集
        train_loader, val_loader = build_pretrain_dataset(self.args)

        # 构建模型
        self.model = self.build_model()
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个 GPU")
            self.model = nn.DataParallel(self.model)
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

            acc_val, loss_val = self.validate(val_loader, epoch)

            # 记录日志
            self.csv_logger.add(acc_train, acc_val, loss_train, loss_val, self.args.lr)
            self.csv_logger.write(
                patience=self.args.patience,
                wait=self.args.wait,
                choose=self.args.trans_choose,
                name=self.args.dataset
            )

            # 早停检查
            self.early_stopping(loss_val, self.model)
            if self.early_stopping.counter >= self.args.wait:
                self.args.lr = adjust_learning_rate(self.optimizer, self.args.lr, self.args.declay)
            if self.early_stopping.early_stop:
                print("早停触发，训练结束")
                break

        print(f"\n预训练完成！最佳模型已保存至: ./checkpoint_pretrain/{self.args.dataset}/")
