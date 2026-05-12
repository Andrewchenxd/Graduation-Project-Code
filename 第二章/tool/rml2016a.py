"""
RML2016.10a 数据集加载和预处理模块
提供数据集的加载、时频变换、数据增强等功能
"""

import numpy as np
import torch
import random
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample, hilbert2

from tool.signal_aug import *
from tool.signeltoimage import *


def toIQ(sgn):
    """将信号转换为 IQ 格式"""
    newsgn = np.zeros((2, sgn.shape[0]))
    y = hilbert2(sgn)
    newsgn[0] = np.real(y)
    newsgn[1] = np.imag(y)
    return newsgn


def l2_normalize(x, axis=-1):
    """L2 归一化"""
    y = np.sum(x ** 2, axis, keepdims=True)
    return x / np.sqrt(y)


def Img_aug(img, imgn):
    """图像数据增强（翻转等）"""
    if np.random.rand() < 0.2:
        img = np.flip(img, 1).copy()
        imgn = np.flip(imgn, 1).copy()
    elif np.random.rand() < 0.4:
        img = np.flip(img, 2).copy()
        imgn = np.flip(imgn, 2).copy()
    elif np.random.rand() < 0.6:
        img = np.flip(np.flip(img, 2), 1).copy()
        imgn = np.flip(np.flip(imgn, 2), 1).copy()
    elif np.random.rand() < 1:
        img = img
        imgn = imgn
    return img, imgn


# ============================================================
# 数据集类
# ============================================================

class SigDataSet_pwvd(Dataset):
    """
    PWVD 时频图数据集（用于预训练）
    加载 IQ 信号并转换为 PWVD 时频图
    """

    def __init__(self, data_path, adsbis=False, newdata=False, snr_range=[1, 7],
                 resample_is=False, samplenum=15, norm='no', imgaug=False,
                 chazhi=False, chazhinum=2, is_DAE=False, resize_is=False,
                 return_label=False, sgnaug=False, sgn_expend=False,
                 RGB_is=False, zhenshiSNR=False, freq_fliter=False):
        super().__init__()
        if adsbis == False:
            loaded = np.load(data_path, allow_pickle=True).item()
            self.data = loaded['data']
            self.labels = loaded['label']
        else:
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
        self.sgnaug = sgnaug
        self.imgaug = imgaug
        self.sgn_expend = sgn_expend
        self.return_label = return_label
        self.RGB_is = RGB_is
        self.zhenshiSNR = zhenshiSNR
        self.freq_fliter = freq_fliter
        if (adsbis == False) and (newdata == False):
            self.snr = loaded['snr']
        if (adsbis == True) or (newdata == True):
            self.rml = False
        self.newdata = newdata

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        if self.sgn_expend == False:
            self.sgn = self.data[item]
            self.sgn_noise = np.copy(self.sgn)
        else:
            if np.random.random() <= 0.5:
                self.sgn = sig_time_warping(self.data[item])
            else:
                self.sgn = self.data[item]
            self.sgn_noise = np.copy(self.sgn)

        np.random.seed(None)
        self.SNR = random.randint(self.snrmin, self.snrmax)
        self.SNR1 = self.SNR - 20
        if self.sgnaug == False:
            self.sgn_noise = awgn(self.sgn_noise, self.SNR1, self.zhenshiSNR, Seed=None)
        else:
            if np.random.random() <= 0.75:
                self.sgn_noise = awgn(self.sgn_noise, self.SNR1, self.zhenshiSNR, Seed=None)
            else:
                self.sgn_noise = rayleigh_noise(self.sgn_noise, self.SNR1, self.zhenshiSNR, Seed=None)
        if self.freq_fliter:
            self.sgn_noise1 = filter(self.sgn_noise, filiter='low', filiter_threshold=0.85,
                                      filiter_size=0.001, middle_zero=True, freq_smooth=True, return_IQ=True)
            self.sgn_noise = self.sgn_noise1 + self.sgn_noise
        if self.resample_is == True:
            self.sgn = resampe(self.sgn, samplenum=self.samplenum)
            self.sgn_noise = resampe(self.sgn_noise, samplenum=self.samplenum)
        img = pwvd(data=self.sgn, norm=self.norm, resize_is=self.resize_is, RGB_is=self.RGB_is)
        imgn = pwvd(data=self.sgn_noise, norm=self.norm, resize_is=self.resize_is, RGB_is=self.RGB_is)
        if self.imgaug == True:
            img, imgn = Img_aug(img, imgn)

        if self.return_label == False:
            return torch.tensor(img, dtype=torch.float32), \
                   torch.tensor(imgn, dtype=torch.float32)
        elif self.return_label == True:
            return torch.tensor(img, dtype=torch.float32), \
                   torch.tensor(imgn, dtype=torch.float32), \
                   torch.tensor(self.labels[item], dtype=torch.long)


class SigDataSet_pwvd_fc(Dataset):
    """
    PWVD 时频图数据集（用于微调/分类）
    加载 IQ 信号并转换为 PWVD/STFT/小波时频图
    """

    def __init__(self, data_path, adsbis=False, newdata=False, snr_range=[1, 7],
                 resample_is=False, samplenum=15, norm='no', seed=None,
                 chazhi=False, chazhinum=2, is_DAE=False, resize_is=False,
                 aug=False, RGB_is=False, trans_choose='pwvd'):
        super().__init__()
        if adsbis == False:
            loaded = np.load(data_path, allow_pickle=True).item()
            self.data = loaded['data']
            self.labels = loaded['label']
        else:
            self.data = pickle.load(open(data_path, 'rb'), encoding='latin')['data']
            self.labels = pickle.load(open(data_path, 'rb'), encoding='latin')['label'].flatten()
        self.snrmin = snr_range[0]
        self.snrmax = snr_range[1]
        self.adsbis = adsbis
        self.seed = seed
        self.resample_is = resample_is
        self.norm = norm
        self.trans_choose = trans_choose
        self.resize_is = resize_is
        self.chazhi = chazhi
        self.cnum = chazhinum
        self.samplenum = samplenum
        self.is_DAE = is_DAE
        self.rml = True
        self.aug = aug
        self.RGB_is = RGB_is
        if (adsbis == False) and (newdata == False):
            self.snr = loaded['snr']
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
        self.sgn = self.data[item]
        if self.is_DAE == True:
            self.sgn = l2_normalize(self.sgn)
        if self.chazhi == True:
            self.sgn = chpip(self.sgn, self.cnum)
        if self.rml == True:
            if self.aug == True:
                functions = [sig_time_warping, sig_rotate, sig_reserve]
                num_select = np.random.randint(0, 1)
                selected_functions = random.sample(functions, num_select)
                for function in selected_functions:
                    self.sgn = function(self.sgn)

            if self.trans_choose == 'pwvd':
                img = pwvd(self.sgn, self.norm, self.resize_is, RGB_is=self.RGB_is)
            elif self.trans_choose == 'stft':
                img = stp(data=self.sgn, norm=self.norm, resize_is=self.resize_is)
            elif self.trans_choose == 'wave':
                img = wave(data=self.sgn, norm=self.norm, resize_is=self.resize_is)

            if self.aug == True:
                if self.RGB_is:
                    if np.random.rand() < 0.5:
                        img = np.flip(img, 1).copy()
                    if np.random.rand() < 0.5:
                        img = np.flip(img, 2).copy()
                    if np.random.rand() < 0.5:
                        img3 = np.transpose(img, (1, 2, 0)).copy()
                        functions = [iaa.Flipud(),
                                     iaa.GaussianBlur(sigma=(0, 3.0)),
                                     iaa.Sharpen(),
                                     iaa.Invert(1.0),
                                     iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)]
                        num_select = 2
                        selected_functions = random.sample(functions, num_select)
                        seq = iaa.Sequential(selected_functions)
                        img = seq.augment_images(img3)
                        img = np.transpose(img, (2, 0, 1)).copy()
                else:
                    if np.random.rand() < 0.25:
                        img = np.flip(img, 1).copy()
                    elif np.random.rand() < 0.5:
                        img = np.flip(img, 2).copy()
                    elif np.random.rand() < 0.75:
                        img = np.flip(np.flip(img, 2), 1).copy()
                    elif np.random.rand() < 1:
                        img = img

            return torch.tensor(img, dtype=torch.float32), torch.tensor(
                self.labels[item], dtype=torch.long), torch.tensor(
                self.snr[item], dtype=torch.long)
        elif self.adsbis == True:
            self.SNR = random.randint(self.snrmin, self.snrmax)
            self.SNR1 = self.SNR - 20
            self.sgn = awgn(self.sgn, self.SNR1)
            if self.resample_is == True:
                self.sgn = resampe(self.sgn, samplenum=self.samplenum)
            if self.trans_choose == 'pwvd':
                self.img = pwvd(self.sgn, self.norm, self.resize_is, RGB_is=self.RGB_is)
            elif self.trans_choose == 'stft':
                self.img = stp(data=self.sgn, norm=self.norm, resize_is=self.resize_is)
            elif self.trans_choose == 'wave':
                self.img = wave(data=self.sgn, norm=self.norm, resize_is=self.resize_is)
            if self.aug == True:
                self.img, imgn = Img_aug(self.img, self.img)
            else:
                self.img = self.img
            return torch.tensor(self.img, dtype=torch.float32), torch.tensor(
                self.labels[item], dtype=torch.long), torch.tensor(
                self.SNR, dtype=torch.long)
        elif self.newdata == True:
            self.SNR = random.randint(self.snrmin, self.snrmax)
            self.SNR1 = 2 * self.SNR - 20
            self.sgn = awgn(self.sgn, self.SNR1, Seed=self.seed)

            if self.trans_choose == 'pwvd':
                img = pwvd(self.sgn, self.norm, self.resize_is, RGB_is=self.RGB_is)
            elif self.trans_choose == 'stft':
                img = stp(data=self.sgn, norm=self.norm, resize_is=self.resize_is)
            elif self.trans_choose == 'wave':
                img = wave(data=self.sgn, norm=self.norm, resize_is=self.resize_is)

            if self.aug == True:
                if self.RGB_is:
                    if np.random.rand() < 0.5:
                        img = np.flip(img, 1).copy()
                    if np.random.rand() < 0.5:
                        img = np.flip(img, 2).copy()
                    if np.random.rand() < 0.5:
                        img3 = np.transpose(img, (1, 2, 0)).copy()
                        functions = [iaa.Flipud(),
                                     iaa.GaussianBlur(sigma=(0, 3.0)),
                                     iaa.Sharpen(),
                                     iaa.Invert(1.0),
                                     iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)]
                        num_select = 2
                        selected_functions = random.sample(functions, num_select)
                        seq = iaa.Sequential(selected_functions)
                        img = seq.augment_images(img3)
                        img = np.transpose(img, (2, 0, 1)).copy()
                else:
                    if np.random.rand() < 0.25:
                        img = np.flip(img, 1).copy()
                    elif np.random.rand() < 0.5:
                        img = np.flip(img, 2).copy()
                    elif np.random.rand() < 0.75:
                        img = np.flip(np.flip(img, 2), 1).copy()
                    elif np.random.rand() < 1:
                        img = img

            return torch.tensor(img, dtype=torch.float32), torch.tensor(
                self.labels[item], dtype=torch.long), torch.tensor(
                self.SNR, dtype=torch.long)


class SigDataSet_stft(Dataset):
    """STFT 时频图数据集"""

    def __init__(self, data_path, adsbis=False, newdata=False, snr_range=[1, 7],
                 resample_is=False, samplenum=15, norm='no',
                 chazhi=False, chazhinum=2, is_DAE=False, resize_is=False,
                 return_label=False, sgnaug=False, sgn_expend=False):
        super().__init__()
        loaded = np.load(data_path, allow_pickle=True).item()
        self.data = loaded['data']
        self.labels = loaded['label']
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
        self.sgnaug = sgnaug
        self.sgn_expend = sgn_expend
        self.return_label = return_label
        if (adsbis == False) and (newdata == False):
            self.snr = loaded['snr']
        if (adsbis == True) or (newdata == True):
            self.rml = False
        self.newdata = newdata

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        if self.sgn_expend == False:
            self.sgn = self.data[item]
            self.sgn_noise = np.copy(self.sgn)
        else:
            self.sgn = sig_time_warping(self.data[item])
            self.sgn_noise = np.copy(self.sgn)
        if self.rml == True:
            np.random.seed(None)
            self.SNR = random.randint(self.snrmin, self.snrmax)
            self.SNR1 = self.SNR - 20
            if self.sgnaug == False:
                self.sgn_noise = awgn(self.sgn_noise, self.SNR1)
            else:
                if np.random.random() <= 0.5:
                    self.sgn_noise = awgn(self.sgn_noise, self.SNR1)
                else:
                    self.sgn_noise = addmask(self.sgn_noise)
            if self.return_label == False:
                return torch.tensor(stp(data=self.sgn, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32), \
                       torch.tensor(stp(data=self.sgn_noise, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32), \
                       self.SNR1
            elif self.return_label == True:
                return torch.tensor(stp(data=self.sgn, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32), \
                       torch.tensor(stp(data=self.sgn_noise, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32), \
                       torch.tensor(self.labels[item], dtype=torch.long)
        elif self.adsbis == True:
            np.random.seed(None)
            self.SNR = random.randint(self.snrmin, self.snrmax)
            self.SNR1 = self.SNR - 20
            if self.sgnaug == False:
                self.sgn_noise = awgn(self.sgn_noise, self.SNR1)
            else:
                if np.random.random() <= 0.5:
                    self.sgn_noise = awgn(self.sgn_noise, self.SNR1)
                else:
                    self.sgn_noise = addmask(self.sgn_noise)
            if self.resample_is == True:
                self.sgn = resampe(self.sgn, samplenum=self.samplenum)
                self.sgn_noise = resampe(self.sgn_noise, samplenum=self.samplenum)
            if self.return_label == False:
                return torch.tensor(stp(data=self.sgn, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32), \
                       torch.tensor(stp(data=self.sgn_noise, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32)
            elif self.return_label == True:
                return torch.tensor(stp(data=self.sgn, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32), \
                       torch.tensor(stp(data=self.sgn_noise, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32), \
                       torch.tensor(self.labels[item], dtype=torch.long)


class SigDataSet_gasf(Dataset):
    """GASF 格拉米角场数据集"""

    def __init__(self, data_path, adsbis=False, newdata=False, snr_range=[1, 7],
                 resample_is=False, samplenum=15, norm='no', data_name='RML2016.10a',
                 chazhi=False, chazhinum=2, is_DAE=False, resize_is=False,
                 return_label=False, sgnaug=False, sgn_expend=False, RGB_is=False):
        super().__init__()
        if adsbis == False:
            loaded = np.load(data_path, allow_pickle=True).item()
            self.data = loaded['data']
            self.labels = loaded['label']
        else:
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
        self.data_name = data_name
        self.rml = True
        self.sgnaug = sgnaug
        self.sgn_expend = sgn_expend
        self.return_label = return_label
        self.RGB_is = RGB_is
        if (adsbis == False) and (newdata == False):
            self.snr = loaded['snr']
        if (adsbis == True) or (newdata == True):
            self.rml = False
        self.newdata = newdata

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        if self.sgn_expend == False:
            self.sgn = self.data[item]
            self.sgn_noise = np.copy(self.sgn)
        else:
            self.sgn = sig_time_warping(self.data[item])
            self.sgn_noise = np.copy(self.sgn)
        if self.resample_is == True:
            self.sgn = resampe(self.sgn, samplenum=self.samplenum)
            self.sgn_noise = resampe(self.sgn_noise, samplenum=self.samplenum)
        if self.data_name == 'RMLc':
            self.sgn = self.sgn / 500
            self.sgn_noise = self.sgn_noise / 500
        np.random.seed(None)
        self.SNR = random.randint(self.snrmin, self.snrmax)
        self.SNR1 = self.SNR - 20
        if self.return_label == False:
            return torch.tensor(gasf(self.sgn, self.norm, self.resize_is, RGB_is=self.RGB_is), dtype=torch.float32), \
                   torch.tensor(gasf(self.sgn_noise, self.norm, self.resize_is, RGB_is=self.RGB_is), dtype=torch.float32)
        elif self.return_label == True:
            return torch.tensor(gasf(self.sgn, self.norm, self.resize_is, RGB_is=self.RGB_is), dtype=torch.float32), \
                   torch.tensor(gasf(self.sgn_noise, self.norm, self.resize_is, RGB_is=self.RGB_is), dtype=torch.float32), \
                   torch.tensor(self.labels[item], dtype=torch.long)


class SigDataSet_wave(Dataset):
    """小波变换数据集"""

    def __init__(self, data_path, adsbis=False, newdata=False, snr_range=[1, 7],
                 resample_is=False, samplenum=15, norm='no',
                 chazhi=False, chazhinum=2, is_DAE=False, resize_is=False,
                 return_label=False, sgnaug=False, sgn_expend=False):
        super().__init__()
        loaded = np.load(data_path, allow_pickle=True).item()
        self.data = loaded['data']
        self.labels = loaded['label']
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
        self.sgnaug = sgnaug
        self.sgn_expend = sgn_expend
        self.return_label = return_label
        if (adsbis == False) and (newdata == False):
            self.snr = loaded['snr']
        if (adsbis == True) or (newdata == True):
            self.rml = False
        self.newdata = newdata

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        if self.sgn_expend == False:
            self.sgn = self.data[item]
            self.sgn_noise = np.copy(self.sgn)
        else:
            self.sgn = sig_time_warping(self.data[item])
            self.sgn_noise = np.copy(self.sgn)
        if self.rml == True:
            np.random.seed(None)
            self.SNR = random.randint(self.snrmin, self.snrmax)
            self.SNR1 = self.SNR - 20
            if self.sgnaug == False:
                self.sgn_noise = awgn(self.sgn_noise, self.SNR1)
            else:
                if np.random.random() <= 0.5:
                    self.sgn_noise = awgn(self.sgn_noise, self.SNR1)
                else:
                    self.sgn_noise = addmask(self.sgn_noise)
            if self.return_label == False:
                return torch.tensor(wave(data=self.sgn, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32), \
                       torch.tensor(wave(data=self.sgn_noise, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32), \
                       self.SNR1
            elif self.return_label == True:
                return torch.tensor(stp(data=self.sgn, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32), \
                       torch.tensor(stp(data=self.sgn_noise, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32), \
                       torch.tensor(self.labels[item], dtype=torch.long)


# ============================================================
# 数据集构建工具函数
# ============================================================

def build_pretrain_dataset(args):
    """
    构建预训练数据集

    参数:
        args: 配置参数 Namespace

    返回:
        train_loader, val_loader: 训练和验证数据加载器
    """
    data_path = args.data_path.format(args.dataset)

    train_set = SigDataSet_pwvd(
        data_path,
        newdata=args.newdata,
        adsbis=args.adsbis,
        resample_is=args.resample,
        samplenum=args.samplenum,
        resize_is=True,
        norm='maxmin',
        snr_range=[16, 30],
        sgn_expend=True,
        sgnaug=True,
        RGB_is=args.RGB_is if hasattr(args, 'RGB_is') else False,
        zhenshiSNR=False,
        freq_fliter=False
    )

    valsplit = 0.7
    train_set, val_set = torch.utils.data.random_split(
        train_set,
        [int(len(train_set) * valsplit),
         int(len(train_set)) - int(len(train_set) * valsplit)]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.numworks,
        prefetch_factor=args.pref
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batchsize,
        num_workers=args.numworks,
        prefetch_factor=args.pref,
        shuffle=True
    )

    return train_loader, val_loader


def build_fintune_dataset(args):
    """
    构建微调数据集

    参数:
        args: 配置参数 Namespace

    返回:
        train_loader, val_loader: 训练和验证数据加载器
    """
    train_data_path = args.train_data_path.format(args.dataset)
    test_data_path = args.test_data_path.format(args.dataset)

    train_set = SigDataSet_pwvd_fc(
        train_data_path,
        newdata=args.newdata,
        adsbis=args.adsbis,
        resample_is=args.resample,
        samplenum=args.samplenum,
        resize_is=False,
        norm='maxmin',
        snr_range=[10, 20],
        aug=args.aug if hasattr(args, 'aug') else False
    )

    val_set = SigDataSet_pwvd_fc(
        test_data_path,
        newdata=args.newdata,
        adsbis=args.adsbis,
        resample_is=args.resample,
        samplenum=args.samplenum,
        resize_is=False,
        norm='maxmin',
        snr_range=[10, 20],
        aug=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.numworks,
        prefetch_factor=args.pref
    )

    val_loader = DataLoader(
        val_set,
        batch_size=2 * args.batchsize,
        num_workers=args.numworks,
        prefetch_factor=args.pref,
        shuffle=True
    )

    return train_loader, val_loader
