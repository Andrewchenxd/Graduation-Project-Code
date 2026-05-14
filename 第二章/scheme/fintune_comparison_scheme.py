"""
对比方法微调方案
实现多种对比方法的分类微调流程，包括：
- 1D 信号模型: mcldnn, hcgdnn, pet, AWN, ICAMCNET, moco1d, TcssAMR
- 2D 图像模型: MAE_ViT_gacf
- 自监督模型: Sgnc2freq, Sgnc2freq_moco, Sgnc2freq_swav/SWT
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import numpy as np
import os

from tool.utils import (
    AverageMeter, CSVStats, EarlyStopping_base_model,
    adjust_learning_rate, acc_classes, acc_snrs
)
from tool.rml2016a import SigDataSet_pwvd_fc
from tool.loss import TripletLoss, CenterLoss
from scheme.base_scheme import BaseScheme

# ============================================================
# 1D 信号模型导入
# ============================================================
from model.mcldnn import mcldnn
from model.HCGDNN import HCGDNN
from model.PETCGDNN import PETCGDNN
from model.AWN import AWN
from model.ICAMCNET import ICAMCNET
from model.mocov3_sgn import ViT_sgn_fc, MoCoV3_sgn_fc
from model.TcssAMR import ViT_TcssAMR_fc

# ============================================================
# 2D 图像模型导入
# ============================================================
from model.vit_simple import MAE_ViT_gasf_fc


# ============================================================
# 数据集类（1D 信号数据集）
# ============================================================
class SigDataSet_sgn_fc(torch.utils.data.Dataset):
    """
    1D 信号数据集（用于微调/分类）
    加载 IQ 信号，支持 .mat 和 .pkl 格式
    返回: (signal, label, snr)
    """
    def __init__(self, data_path, adsbis=False, newdata=False, snr_range=[1, 7],
                 resample_is=False, samplenum=15, norm='no',
                 chazhi=False, chazhinum=2, is_DAE=False, resize_is=False,
                 aug=False, sgnaug=False, sgn_expend=False):
        super().__init__()
        if data_path.endswith('.npy'):
            loaded = np.load(data_path, allow_pickle=True).item()
            self.data = loaded['data']
            self.labels = loaded['label']
        elif data_path.endswith('.mat'):
            import scipy.io as scio
            self.data = scio.loadmat(data_path)['data']
            self.labels = scio.loadmat(data_path)['label'].flatten()
        elif data_path.endswith('.pkl'):
            import pickle
            self.data = pickle.load(open(data_path, 'rb'), encoding='latin')['data']
            self.labels = pickle.load(open(data_path, 'rb'), encoding='latin')['label'].flatten()
        self.snrmin = snr_range[0]
        self.snrmax = snr_range[1]
        self.adsbis = adsbis
        self.resample_is = resample_is
        self.norm = norm
        self.resize_is = resize_is
        self.chazhi = chazhi
        self.cnum = chazhinum
        self.samplenum = samplenum
        self.is_DAE = is_DAE
        self.rml = True
        self.aug = aug
        self.sgnaug = sgnaug
        self.sgn_expend = sgn_expend
        if (adsbis == False) and (newdata == False):
            if data_path.endswith('.npy'):
                self.snr = loaded['snr']
            elif data_path.endswith('.mat'):
                import scipy.io as scio
                self.snr = scio.loadmat(data_path)['snr'].flatten()
            elif data_path.endswith('.pkl'):
                import pickle
                self.snr = pickle.load(open(data_path, 'rb'), encoding='latin')['snr'].flatten()
            if 'RML2022_a' in data_path:
                unique_values = np.unique(self.snr)
                value_to_idx = {value: idx for idx, value in enumerate(unique_values)}
                self.snr = np.array([value_to_idx[value] for value in self.snr])
        if (adsbis == True) or (newdata == True):
            self.rml = False
        self.newdata = newdata

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        from tool.signal_aug import awgn, sig_time_warping, sig_rotate, addmask, sig_reserve
        from tool.signeltoimage import resampe
        from tool.signeltoimage import sgn_norm

        if self.sgn_expend == False:
            self.sgn = self.data[item]
            self.sgn_noise = np.copy(self.sgn)
        else:
            self.sgn = sig_time_warping(self.data[item])
            self.sgn_noise = np.copy(self.sgn)

        np.random.seed(None)
        self.SNR = np.random.randint(self.snrmin, self.snrmax)
        if self.newdata:
            self.SNR1 = 2 * self.SNR - 20
        else:
            self.SNR1 = self.SNR - 20

        if self.adsbis == False:
            if self.sgnaug == False:
                self.sgn_noise = awgn(self.sgn_noise, self.SNR1, Seed=None)
            else:
                random_nums1 = np.random.uniform(low=0.9, high=0.99, size=1)
                random_nums2 = np.random.uniform(low=0.5, high=1.5, size=1)
                from tool.signeltoimage import filter as sgn_filter
                self.sgn_noise = sgn_filter(self.sgn_noise, filiter='high',
                                            filiter_threshold=random_nums1,
                                            filiter_size=random_nums2, return_IQ=True)
        else:
            self.sgn_noise = awgn(self.sgn_noise, self.SNR1, Seed=None)

        if self.resample_is == True:
            self.sgn = resampe(self.sgn, samplenum=self.samplenum)
            self.sgn_noise = resampe(self.sgn_noise, samplenum=self.samplenum)

        # 数据增强
        if self.aug:
            np.random.seed(None)
            import random
            random.seed(None)
            functions = [sig_rotate, addmask, sig_reserve, sig_time_warping]
            selected_functions = random.sample(functions, 1)
            for func in selected_functions:
                self.sgn = func(self.sgn)
                self.sgn_noise = func(self.sgn_noise)

        # 归一化并添加通道维度
        self.sgn = sgn_norm(self.sgn, normtype=self.norm)
        self.sgn_noise = sgn_norm(self.sgn_noise, normtype=self.norm)
        self.sgn = np.expand_dims(self.sgn, 0)
        self.sgn_noise = np.expand_dims(self.sgn_noise, 0)

        return torch.tensor(self.sgn, dtype=torch.float32), \
               torch.tensor(self.labels[item], dtype=torch.long), \
               torch.tensor(self.SNR1, dtype=torch.long)


class SigDataSet_gasf_fc(torch.utils.data.Dataset):
    """
    GASF 格拉米角场数据集（用于微调/分类）
    加载 IQ 信号并转换为 GASF 图像
    返回: (image, label, snr)
    """
    def __init__(self, data_path, adsbis=False, newdata=False, snr_range=[1, 7],
                 resample_is=False, samplenum=15, norm='no', data_name='RML2016.10a',
                 chazhi=False, chazhinum=2, is_DAE=False, resize_is=False,
                 aug=False, RGB_is=False, sample_range=None):
        super().__init__()
        import scipy.io as scio
        import pickle
        if adsbis == False:
            self.data = scio.loadmat(data_path)['data']
            self.labels = scio.loadmat(data_path)['label'].flatten()
        else:
            self.data = pickle.load(open(data_path, 'rb'), encoding='latin')['data']
            self.labels = pickle.load(open(data_path, 'rb'), encoding='latin')['label'].flatten()
        self.snrmin = snr_range[0]
        self.snrmax = snr_range[1]
        self.adsbis = adsbis
        self.resample_is = resample_is
        self.norm = norm
        self.data_name = data_name
        self.sample_range = sample_range
        self.resize_is = resize_is
        self.chazhi = chazhi
        self.cnum = chazhinum
        self.samplenum = samplenum
        self.is_DAE = is_DAE
        self.rml = True
        self.aug = aug
        self.RGB_is = RGB_is
        if (adsbis == False) and (newdata == False):
            self.snr = scio.loadmat(data_path)['snr'].flatten()
        if (adsbis == True) or (newdata == True):
            self.rml = False
        self.newdata = newdata

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        from tool.signal_aug import awgn, resampe, sig_time_warping, sig_rotate, chpip
        from tool.signeltoimage import gasf

        self.sgn = self.data[item]
        if self.is_DAE == True:
            from tool.rml2016a import l2_normalize
            self.sgn = l2_normalize(self.sgn)
        if self.chazhi == True:
            self.sgn = chpip(self.sgn, self.cnum)

        if self.rml == True:
            if self.aug:
                np.random.seed(None)
                import random
                random.seed(None)
                functions = [sig_time_warping, sig_rotate]
                selected_functions = random.sample(functions, 1)
                for func in selected_functions:
                    self.sgn = func(self.sgn)

            if self.data_name == 'RMLc':
                self.sgn = self.sgn / 500

            img = gasf(self.sgn, self.norm, self.resize_is,
                       RGB_is=self.RGB_is, sample_range=self.sample_range)

            if self.aug:
                if self.RGB_is:
                    if np.random.rand() < 0.33:
                        img = np.flip(img, 1).copy()
                    elif np.random.rand() < 0.66:
                        img = np.flip(img, 2).copy()
                    img3 = np.transpose(img, (1, 2, 0)).copy()
                    alpha = np.random.randn()
                    beta = np.random.randn()
                    import imgaug.augmenters as iaa
                    functions_iaa = [
                        iaa.GaussianBlur(sigma=(np.abs(alpha), np.abs(beta))),
                        iaa.Sharpen(),
                        iaa.Invert(0),
                        iaa.ContrastNormalization((1 + alpha, 1 + beta))
                    ]
                    num_select = 2
                    selected_functions = random.sample(functions_iaa, num_select)
                    seq = iaa.Sequential(selected_functions)
                    img = seq.augment_images(img3)
                    img = np.transpose(img, (2, 0, 1)).copy()
                else:
                    if np.random.rand() < 0.5:
                        img = np.flip(img, 1).copy()
                    elif np.random.rand() < 1:
                        img = np.flip(img, 2).copy()

            return torch.tensor(img, dtype=torch.float32), \
                   torch.tensor(self.labels[item], dtype=torch.long), \
                   torch.tensor(self.snr[item], dtype=torch.long)

        elif self.adsbis == True:
            import random
            self.SNR = random.randint(self.snrmin, self.snrmax)
            self.SNR1 = self.SNR - 20
            self.sgn = awgn(self.sgn, self.SNR1)
            if self.resample_is == True:
                self.sgn = resampe(self.sgn, samplenum=self.samplenum)
            if self.data_name == 'RMLc':
                self.sgn = self.sgn / 1000
            img = gasf(self.sgn, self.norm, self.resize_is, RGB_is=self.RGB_is,
                       sample_range=self.sample_range)
            return torch.tensor(img, dtype=torch.float32), \
                   torch.tensor(self.labels[item], dtype=torch.long), \
                   torch.tensor(self.SNR, dtype=torch.long)


# ============================================================
# 模型名称常量
# ============================================================
# 1D 信号模型（使用 SigDataSet_sgn_fc 数据集）
SGN_MODEL_NAMES = ['mcldnn', 'hcgdnn', 'pet', 'AWN', 'ICAMCNET',
                   'moco1d', 'TcssAMR', 'Sgnc2freq',
                   'Sgnc2freq_moco', 'Sgnc2freq_swav', 'SWT']

# 2D 图像模型（使用 SigDataSet_pwvd_fc 或 SigDataSet_gasf_fc 数据集）
IMG_MODEL_NAMES = ['MAE_ViT_gacf']

# 需要加载预训练权重的 SSL 模型
SSL_MODEL_NAMES = ['moco1d', 'TcssAMR', 'Sgnc2freq',
                   'Sgnc2freq_moco', 'Sgnc2freq_swav', 'SWT', 'MAE_ViT_gacf']

# 需要特殊 forward 处理的模型
AWN_MODEL_NAMES = ['AWN']  # 返回 (output, regit)
HCGDNN_MODEL_NAMES = ['hcgdnn']  # 返回 3 个输出


class FintuneComparisonScheme(BaseScheme):
    """
    对比方法微调方案
    支持多种对比方法的分类微调，包括 1D 信号模型和 2D 图像模型
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.model_name if hasattr(args, 'model_name') else 'mcldnn'
        self.is_sgn_model = self.model_name in SGN_MODEL_NAMES
        self.is_img_model = self.model_name in IMG_MODEL_NAMES
        self.is_ssl_model = self.model_name in SSL_MODEL_NAMES
        self.is_awn_model = self.model_name in AWN_MODEL_NAMES
        self.is_hcgdnn_model = self.model_name in HCGDNN_MODEL_NAMES

    def build_dataset(self):
        """构建数据集"""
        adsbis = self.args.adsbis if hasattr(self.args, 'adsbis') else False
        newdata = self.args.newdata if hasattr(self.args, 'newdata') else False
        resample = self.args.resample if hasattr(self.args, 'resample') else False
        samplenum = self.args.samplenum if hasattr(self.args, 'samplenum') else 6
        aug = self.args.aug if hasattr(self.args, 'aug') else False

        train_data_path = self.args.train_data_path.format(self.args.dataset)
        test_data_path = self.args.test_data_path.format(self.args.dataset)

        if self.is_sgn_model:
            # 1D 信号模型使用 SigDataSet_sgn_fc
            # 根据模型选择归一化方式
            if self.model_name in ['TcssAMR', 'Sgnc2freq', 'Sgnc2freq_moco', 'Sgnc2freq_swav', 'SWT']:
                norm_style = 'maxmin-1'
            else:
                norm_style = 'maxmin'

            train_set = SigDataSet_sgn_fc(
                train_data_path, newdata=newdata, adsbis=adsbis,
                resample_is=resample, samplenum=samplenum, resize_is=False,
                norm=norm_style, snr_range=[10, 40], aug=aug
            )
            val_set = SigDataSet_sgn_fc(
                test_data_path, newdata=newdata, adsbis=adsbis,
                resample_is=resample, samplenum=samplenum, resize_is=False,
                norm=norm_style, snr_range=[10, 40], aug=False
            )

        elif self.model_name == 'MAE_ViT_gacf':
            # MAE_ViT_gacf 使用 GASF 数据集
            norm_style = 'maxmin'
            RGB_is = self.args.RGB_is if hasattr(self.args, 'RGB_is') else False
            resize_is = self.args.resize_is if hasattr(self.args, 'resize_is') else True

            train_set = SigDataSet_gasf_fc(
                train_data_path, newdata=newdata, adsbis=adsbis,
                data_name=self.args.dataset,
                resample_is=resample, samplenum=samplenum,
                resize_is=resize_is, norm=norm_style,
                snr_range=[10, 40], aug=aug, RGB_is=RGB_is
            )
            val_set = SigDataSet_gasf_fc(
                test_data_path, newdata=newdata, adsbis=adsbis,
                data_name=self.args.dataset,
                resample_is=resample, samplenum=samplenum,
                resize_is=resize_is, norm=norm_style,
                snr_range=[10, 40], aug=False, RGB_is=RGB_is
            )
        else:
            # 其他 2D 模型使用 PWVD 数据集
            norm_style = 'maxmin'
            RGB_is = self.args.RGB_is if hasattr(self.args, 'RGB_is') else False
            resize_is = self.args.resize_is if hasattr(self.args, 'resize_is') else False
            trans_choose = self.args.trans_choose if hasattr(self.args, 'trans_choose') else 'pwvd'

            train_set = SigDataSet_pwvd_fc(
                train_data_path, newdata=newdata, adsbis=adsbis,
                resample_is=resample, samplenum=samplenum,
                resize_is=resize_is, norm=norm_style,
                snr_range=[10, 40], aug=aug, RGB_is=RGB_is,
                trans_choose=trans_choose
            )
            val_set = SigDataSet_pwvd_fc(
                test_data_path, newdata=newdata, adsbis=adsbis,
                resample_is=resample, samplenum=samplenum,
                resize_is=resize_is, norm=norm_style,
                snr_range=[10, 40], aug=False, RGB_is=RGB_is,
                trans_choose=trans_choose
            )

        self.train_loader = DataLoader(
            train_set, batch_size=self.args.batchsize, shuffle=True,
            num_workers=self.args.numworks, prefetch_factor=self.args.pref
        )
        self.val_loader = DataLoader(
            val_set, batch_size=self.args.batchsize,
            num_workers=self.args.numworks, prefetch_factor=self.args.pref,
            shuffle=True
        )

    def build_model(self) -> nn.Module:
        """构建对比方法模型"""
        model_name = self.model_name
        num_classes = self.args.classesnum
        adsbis = self.args.adsbis if hasattr(self.args, 'adsbis') else False

        if model_name == 'mcldnn':
            model = mcldnn(num_classes=num_classes)

        elif model_name == 'hcgdnn':
            model = HCGDNN(numclass=num_classes)

        elif model_name == 'pet':
            model = PETCGDNN(numclass=num_classes, adsb_is=adsbis)

        elif model_name == 'AWN':
            model = AWN(num_classes=num_classes)

        elif model_name == 'ICAMCNET':
            model = ICAMCNET(num_classes=num_classes, adsb_is=adsbis)

        elif model_name == 'moco1d':
            img_size = 500 if self.args.dataset == 'adsb' else 128
            channels = 1
            v = ViT_sgn_fc(
                image_size=img_size, channels=channels, patch_size=1,
                num_classes=num_classes, dim=256, depth=3, heads=8,
                mlp_dim=512, dropout=0.1, emb_dropout=0.1
            )
            model = MoCoV3_sgn_fc(v, 128, 65536, 0.99, 0.07, True)

        elif model_name == 'TcssAMR':
            img_size = 500 if self.args.dataset == 'adsb' else 128
            channels = 1
            model = ViT_TcssAMR_fc(
                image_size=img_size, channels=channels, patch_size=1,
                num_classes=num_classes, dim=256, depth=3, heads=8,
                mlp_dim=512, dropout=0.2, emb_dropout=0.2
            )

        elif model_name == 'Sgnc2freq_moco':
            model = Moco_Informer_Encoder(
                enc_in=2, out_len=128,
                factor=self.args.factor if hasattr(self.args, 'factor') else 5,
                d_model=self.args.d_model if hasattr(self.args, 'd_model') else 176,
                n_heads=self.args.n_heads if hasattr(self.args, 'n_heads') else 8,
                e_layers=self.args.e_layers if hasattr(self.args, 'e_layers') else 3,
                d_ff=self.args.d_ff if hasattr(self.args, 'd_ff') else 256,
                dropout=0.2, attn='prob', embed='fixed', freq='h',
                activation='gelu', output_attention=False,
                distil=self.args.distil if hasattr(self.args, 'distil') else True,
                is_fc=True, num_classes=num_classes
            )

        elif model_name in ['Sgnc2freq_swav', 'SWT']:
            from model.TCSSINformer import SWAV_Informer_Encoder
            model = SWAV_Informer_Encoder(
                enc_in=2, out_len=128,
                factor=self.args.factor if hasattr(self.args, 'factor') else 5,
                d_model=self.args.d_model if hasattr(self.args, 'd_model') else 176,
                n_heads=self.args.n_heads if hasattr(self.args, 'n_heads') else 8,
                e_layers=self.args.e_layers if hasattr(self.args, 'e_layers') else 3,
                d_ff=self.args.d_ff if hasattr(self.args, 'd_ff') else 256,
                dropout=0.2, attn='prob', embed='fixed', freq='h',
                activation='gelu', output_attention=False,
                distil=self.args.distil if hasattr(self.args, 'distil') else True,
                nmb_prototypes=11, projhead_dim=256,
                is_fc=True, num_classes=num_classes, high_freq=False
            )

        elif model_name == 'MAE_ViT_gacf':
            in_channel = 3 if (hasattr(self.args, 'RGB_is') and self.args.RGB_is) else 1
            model = MAE_ViT_gasf_fc(
                image_size=128, in_channel=in_channel, patch_size=8,
                emb_dim=768, encoder_layer=4, encoder_head=8,
                decoder_layer=2, decoder_head=12,
                mask_ratio=0.2, numclass=num_classes
            )

        else:
            raise ValueError(f"未知的模型名称: {model_name}")

        return model

    def build_optimizer(self):
        """构建优化器"""
        model_name = self.model_name

        if model_name == 'MAE_ViT_gacf':
            # MAE_ViT_gacf 使用差分学习率
            self.optimizer = torch.optim.Adam([
                {'params': self.model.mlp_head.parameters(), 'lr': self.args.lr * 2},
                {'params': [p for n, p in self.model.named_parameters()
                            if 'mlp_head' not in n], 'lr': self.args.lr}
            ], weight_decay=1e-6)
        else:
            # 其他模型使用 AdamW
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                betas=(0.9, 0.95),
                weight_decay=0.3
            )

    def build_criterion(self):
        """构建损失函数"""
        model_name = self.model_name

        if model_name in ['SFS_SEI', 'CVCNN', 'Freq_CVCNN']:
            # 这些模型需要 TripletLoss + CenterLoss
            self.criterion = [
                nn.CrossEntropyLoss(),
                TripletLoss(margin=10),
                CenterLoss(num_classes=self.args.classesnum, feat_dim=1024)
            ]
        else:
            self.criterion = nn.CrossEntropyLoss()

    def build_lr_scheduler(self):
        """构建学习率调度器"""
        model_name = self.model_name

        if model_name == 'MAE_ViT_gacf':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=5, T_mult=2, eta_min=1e-8
            )
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.args.epochs // 2,
                gamma=0.1, last_epoch=-1
            )

    def build_early_stopping(self):
        """构建早停策略"""
        # 在保存路径中加入模型名称，避免不同模型的检查点互相覆盖
        # 例如: checkpoint_fintune/RMLA/mcldnn/
        save_path = self.args.save_path.format(self.args.dataset)
        save_path = os.path.join(save_path, self.model_name)
        os.makedirs(save_path, exist_ok=True)
        self.early_stopping = EarlyStopping_base_model(
            save_path=save_path,
            patience=self.args.patience if hasattr(self.args, 'patience') else 200,
            wait=self.args.wait if hasattr(self.args, 'wait') else 5,
            choose=self.model_name
        )

    def load_pretrained_weights(self):
        """加载预训练权重（仅对 SSL 模型有效）"""
        if not self.is_ssl_model:
            return

        model_name = self.model_name
        model_path = self.args.model_path if hasattr(self.args, 'model_path') else None

        if model_path is None or not os.path.exists(model_path):
            print(f'警告: 未找到预训练权重路径 {model_path}，跳过加载')
            return

        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=self.device)

        # 处理不同的键名映射
        if model_name in ['TcssAMR', 'Sgnc2freq_moco']:
            pretrained_dict = {k.replace('encoder_q.', '', 1): v
                               for k, v in pretrained_dict.items()}
        elif model_name in ['Sgnc2freq_swav', 'SWT']:
            encoder_choose = self.args.encoder_choose if hasattr(self.args, 'encoder_choose') else 'encoder_q'
            if encoder_choose == 'encoder_q':
                pretrained_dict = {k.replace('encoder.', '', 1): v
                                   for k, v in pretrained_dict.items()}
            elif encoder_choose == 'encoder_k':
                pretrained_dict = {k.replace('encoder_k.', '', 1): v
                                   for k, v in pretrained_dict.items()}
        elif model_name == 'MAE_ViT_gacf':
            pretrained_dict = {k.replace('module.', ''): v
                               for k, v in pretrained_dict.items()}

        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)

        model_dict.update(temp_dict)
        self.model.load_state_dict(model_dict)
        print(f'成功加载 {len(load_key)} / {len(pretrained_dict)} 个参数')

    def model_forward(self, images):
        """
        统一的模型前向传播
        处理不同模型的 forward 签名差异
        """
        model_name = self.model_name

        if model_name == 'AWN':
            output, regit = self.model(images)
            return output, regit
        elif model_name == 'hcgdnn':
            output1, output2, output3 = self.model(images)
            return (output1, output2, output3), None
        elif model_name == 'MAE_ViT_gacf':
            # MAE_ViT_gacf 需要两个参数
            output = self.model(images, images)
            return output, None
        else:
            output = self.model(images)
            return output, None

    def train_one_epoch(self, train_loader, epoch):
        """训练一个 epoch"""
        losses_class = AverageMeter()
        acc = AverageMeter()
        adsbis = self.args.adsbis if hasattr(self.args, 'adsbis') else False

        if adsbis:
            acc_snr_pre = np.zeros((1, 31))
            acc_snr_count = np.zeros((1, 31))
        else:
            acc_snr_pre = np.zeros((1, 21))
            acc_snr_count = np.zeros((1, 21))

        self.model.train()

        with tqdm(total=len(train_loader),
                  desc=f'Epoch {epoch}/{self.args.epochs} [{self.model_name}]',
                  postfix=dict, mininterval=0.3) as pbar:
            for i, (input1, target, snr) in enumerate(train_loader):
                images, labels = input1.to(self.device), target.to(self.device)

                # 前向传播（处理不同模型的差异）
                if self.model_name == 'hcgdnn':
                    output1, output2, output3 = self.model(images)
                    loss1 = self.criterion(output1, labels)
                    loss2 = self.criterion(output2, labels)
                    loss3 = self.criterion(output3, labels)
                    loss = loss1 + loss2 + loss3
                    # 使用 output1 计算准确率
                    output = output1
                elif self.model_name == 'AWN':
                    output, regit = self.model(images)
                    loss = self.criterion(output, labels) + sum(regit)
                elif self.model_name == 'MAE_ViT_gacf':
                    output = self.model(images, images)
                    loss = self.criterion(output, labels)
                else:
                    output = self.model(images)
                    loss = self.criterion(output, labels)

                # 计算准确率
                acc.update(acc_classes(output.data, target, images.size(0)))
                if adsbis:
                    acc_snrs(output, labels, snr - 10, acc_snr_pre, acc_snr_count)
                else:
                    acc_snrs(output, labels, snr, acc_snr_pre, acc_snr_count)
                losses_class.update(loss.item())

                # 反向传播
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
            acc_snr_pre_val = np.zeros((1, 31))
            acc_snr_count_val = np.zeros((1, 31))
        else:
            acc_snr_pre_val = np.zeros((1, 21))
            acc_snr_count_val = np.zeros((1, 21))

        self.model.eval()

        with torch.no_grad():
            with tqdm(total=len(val_loader),
                      desc=f'Val {epoch}/{self.args.epochs} [{self.model_name}]',
                      postfix=dict, mininterval=0.3, colour='blue') as pbar:
                for i, (input1, target, snr) in enumerate(val_loader):
                    val_image, val_label = input1.to(self.device), target.to(self.device)

                    # 前向传播
                    if self.model_name == 'hcgdnn':
                        output1, output2, output3 = self.model(val_image)
                        loss1 = self.criterion(output1, val_label)
                        loss2 = self.criterion(output2, val_label)
                        loss3 = self.criterion(output3, val_label)
                        loss = loss1 + loss2 + loss3
                        output = output1
                    elif self.model_name == 'AWN':
                        output, regit = self.model(val_image)
                        loss = self.criterion(output, val_label) + sum(regit)
                    elif self.model_name == 'MAE_ViT_gacf':
                        output = self.model(val_image, val_image)
                        loss = self.criterion(output, val_label)
                    else:
                        output = self.model(val_image)
                        loss = self.criterion(output, val_label)

                # 计算准确率
                acc.update(acc_classes(output.data, target, val_image.size(0)))
                if adsbis:
                    acc_snrs(output, val_label, snr - 10, acc_snr_pre_val, acc_snr_count_val)
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
        print(f"开始对比方法微调 - 模型: {self.model_name}, 数据集: {self.args.dataset}")
        print(f"{'='*60}")

        # 构建数据集
        self.build_dataset()

        # 构建模型
        self.model = self.build_model()
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个 GPU")
            self.model = nn.DataParallel(self.model)
            self.model = self.model.to(self.device)
        else:
            self.model = self.model.to(self.device)

        # 加载预训练权重（仅 SSL 模型）
        self.load_pretrained_weights()

        # 构建优化器、损失函数等
        self.build_optimizer()
        self.build_criterion()
        self.build_lr_scheduler()
        self.build_early_stopping()

        self.csv_logger = CSVStats()

        # 训练循环
        for epoch in range(self.args.epochs):
            acc_train, loss_train = self.train_one_epoch(self.train_loader, epoch)

            torch.cuda.empty_cache()

            if epoch % 10 == 0:
                acc_val, loss_val = self.validate(self.val_loader, epoch)
                self.early_stopping(loss_val, self.model)

            if epoch == self.args.epochs - 1:
                acc_val, loss_val = self.validate(self.val_loader, epoch)
                self.early_stopping(loss_val, self.model)

            self.lr_scheduler.step()

        print(f"\n微调完成！模型: {self.model_name}, 最佳模型已保存")