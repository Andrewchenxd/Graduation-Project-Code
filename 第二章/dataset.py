import numpy as np
import torch
import random
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torch.utils.data import Dataset  # Dataset是个抽象类，只能用于继承
from scipy.signal import resample, hilbert2
from utils.signal_aug import *
from utils.signeltoimage import *
import pickle
def toIQ(sgn):
    newsgn = np.zeros((2, sgn.shape[0]))
    y = hilbert2(sgn)
    newsgn[0] = np.real(y)
    newsgn[1] = np.imag(y)
    return newsgn


def l2_normalize(x, axis=-1):
    y = np.sum(x ** 2, axis, keepdims=True)
    return x / np.sqrt(y)


# dataset:RML2016.10a (train test 7:3)
class SigDataSet_stft(Dataset):
    def __init__(self, data_path, adsbis=False, newdata=False, snr_range=[1, 7], resample_is=False, samplenum=15,
                 norm='no'
                 , chazhi=False, chazhinum=2, is_DAE=False, resize_is=False, return_label=False, sgnaug=False
                 , sgn_expend=False):
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
                       torch.tensor(stp(data=self.sgn_noise, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32),\
                        self.SNR1
            elif self.return_label == True:
                return torch.tensor(stp(data=self.sgn, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32), \
                       torch.tensor(stp(data=self.sgn_noise, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32),\
                    torch.tensor(self.labels[item], dtype=torch.long)
        elif self.adsbis==True:
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
                       torch.tensor(stp(data=self.sgn_noise, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32),\
                    torch.tensor(self.labels[item], dtype=torch.long)

def Img_aug(img,imgn):
    if np.random.rand() < 0.2:
        img = np.flip(img, 1).copy()
        imgn = np.flip(imgn, 1).copy()
    elif np.random.rand() < 0.4:
        img = np.flip(img, 2).copy()
        imgn = np.flip(imgn, 2).copy()
    elif np.random.rand() < 0.6:
        img = np.flip(np.flip(img, 2),1).copy()
        imgn = np.flip(np.flip(imgn, 2),1).copy()
    elif np.random.rand() < 1:
        img=img
        imgn=imgn
    return img,imgn

class SigDataSet_pwvd(Dataset):
    def __init__(self, data_path, adsbis=False, newdata=False, snr_range=[1, 7], resample_is=False, samplenum=15,
                 norm='no',imgaug=False
                 , chazhi=False, chazhinum=2, is_DAE=False, resize_is=False, return_label=False, sgnaug=False
                 , sgn_expend=False, RGB_is=False,zhenshiSNR=False,freq_fliter=False):
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
        self.imgaug=imgaug
        self.sgn_expend = sgn_expend
        self.return_label = return_label
        self.RGB_is=RGB_is
        self.zhenshiSNR=zhenshiSNR
        self.freq_fliter=freq_fliter
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
            self.sgn_noise = awgn(self.sgn_noise, self.SNR1,self.zhenshiSNR,Seed=None)
        else:
            if np.random.random() <= 0.75:
                self.sgn_noise = awgn(self.sgn_noise, self.SNR1,self.zhenshiSNR,Seed=None)
            else :
                self.sgn_noise = rayleigh_noise(self.sgn_noise, self.SNR1,self.zhenshiSNR,Seed=None)
        if self.freq_fliter:
            self.sgn_noise1 =filter(self.sgn_noise, filiter='low', filiter_threshold=0.85,
                filiter_size=0.001,middle_zero=True,freq_smooth=True,return_IQ=True)
            self.sgn_noise=self.sgn_noise1+self.sgn_noise
        if self.resample_is == True:
            self.sgn = resampe(self.sgn, samplenum=self.samplenum)
            self.sgn_noise = resampe(self.sgn_noise, samplenum=self.samplenum)
        img=pwvd(data=self.sgn, norm=self.norm, resize_is=self.resize_is,RGB_is=self.RGB_is)
        imgn=pwvd(data=self.sgn_noise, norm=self.norm, resize_is=self.resize_is,RGB_is=self.RGB_is)
        if self.imgaug == True:
            img,imgn=Img_aug(img,imgn)

        if self.return_label == False:
            return torch.tensor(img, dtype=torch.float32), \
                   torch.tensor(imgn, dtype=torch.float32)
        elif self.return_label == True:
            return torch.tensor(img, dtype=torch.float32), \
                   torch.tensor(imgn, dtype=torch.float32), \
                   torch.tensor(self.labels[item], dtype=torch.long)


class SigDataSet_gasf(Dataset):
    def __init__(self, data_path, adsbis=False, newdata=False, snr_range=[1, 7], resample_is=False, samplenum=15,
                 norm='no',data_name='RML2016.10a'
                 , chazhi=False, chazhinum=2, is_DAE=False, resize_is=False, return_label=False, sgnaug=False
                 , sgn_expend=False,RGB_is=False):
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
        self.data_name=data_name
        self.rml = True
        self.sgnaug = sgnaug
        self.sgn_expend = sgn_expend
        self.return_label = return_label
        self.RGB_is=RGB_is
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
        # if self.sgnaug == False:
        #     self.sgn_noise = awgn(self.sgn_noise, self.SNR1)
        # else:
        #     if np.random.random() <= 0.5:
        #         self.sgn_noise = awgn(self.sgn_noise, self.SNR1)
        #     else:
        #         self.sgn_noise = addmask(self.sgn_noise)
        if self.return_label == False:
            return torch.tensor(gasf(self.sgn, self.norm, self.resize_is,RGB_is=self.RGB_is), dtype=torch.float32), \
                   torch.tensor(gasf(self.sgn_noise, self.norm, self.resize_is,RGB_is=self.RGB_is), dtype=torch.float32)
        elif self.return_label == True:
            return torch.tensor(gasf(self.sgn, self.norm, self.resize_is,RGB_is=self.RGB_is), dtype=torch.float32), \
                   torch.tensor(gasf(self.sgn_noise, self.norm, self.resize_is,RGB_is=self.RGB_is), dtype=torch.float32), \
                   torch.tensor(self.labels[item], dtype=torch.long)


class SigDataSet_wave(Dataset):
    def __init__(self, data_path, adsbis=False, newdata=False, snr_range=[1, 7], resample_is=False, samplenum=15,
                 norm='no'
                 , chazhi=False, chazhinum=2, is_DAE=False, resize_is=False, return_label=False, sgnaug=False
                 , sgn_expend=False):
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
                       torch.tensor(stp(data=self.sgn_noise, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32),\
                    torch.tensor(self.labels[item], dtype=torch.long)


class SigDataSet_pwvd_fc(Dataset):
    def __init__(self, data_path, adsbis=False, newdata=False, snr_range=[1, 7], resample_is=False, samplenum=15,
                 norm='no',seed=None
                 , chazhi=False, chazhinum=2, is_DAE=False, resize_is=False, aug=False,RGB_is=False,trans_choose='pwvd'):
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
        self.seed=seed
        self.resample_is = resample_is
        self.norm = norm
        self.trans_choose=trans_choose
        self.resize_is = resize_is
        self.chazhi = chazhi
        self.cnum = chazhinum
        self.samplenum = samplenum
        self.is_DAE = is_DAE
        self.rml = True
        self.aug = aug
        self.RGB_is=RGB_is
        if (adsbis == False) and (newdata == False):
            self.snr = loaded['snr']
            if 'RML2022_a' in data_path:
                unique_values = np.unique(self.snr)
                # 创建映射字典
                value_to_idx = {value: idx for idx, value in enumerate(unique_values)}
                # 将不连续的值映射为连续的整数索引
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

            if self.trans_choose=='pwvd':
                img = pwvd(self.sgn, self.norm, self.resize_is,RGB_is=self.RGB_is)  # 1,224,224
            elif self.trans_choose=='stft':
                img = stp(data=self.sgn, norm=self.norm, resize_is=self.resize_is)
            elif self.trans_choose=='wave':
                img = wave(data=self.sgn, norm=self.norm, resize_is=self.resize_is)  # 1,224,224
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
                        img =  np.flip(np.flip(img, 2),1).copy()
                    elif np.random.rand() < 1:
                        img = img

            return torch.tensor(img, dtype=torch.float32), torch.tensor(
                self.labels[item], dtype=torch.long), torch.tensor(
                self.snr[item], dtype=torch.long)
        elif self.adsbis == True:

            self.SNR = random.randint(self.snrmin, self.snrmax)
            self.SNR1 = self.SNR  - 20

            self.sgn = awgn(self.sgn, self.SNR1)
            if self.resample_is == True:
                self.sgn = resampe(self.sgn, samplenum=self.samplenum)
            if self.trans_choose == 'pwvd':
                self.img = pwvd(self.sgn, self.norm, self.resize_is, RGB_is=self.RGB_is)  # 1,224,224
            elif self.trans_choose == 'stft':
                self.img = stp(data=self.sgn, norm=self.norm, resize_is=self.resize_is)
            elif self.trans_choose == 'wave':
                self.img = wave(data=self.sgn, norm=self.norm, resize_is=self.resize_is)  # 1,224,224
            if self.aug == True:
                self.img, imgn = Img_aug(self.img, self.img)
            else:
                # self.img=pwvd(self.sgn)
                self.img = self.img
            return torch.tensor(self.img, dtype=torch.float32), torch.tensor(
                self.labels[item], dtype=torch.long), torch.tensor(
                self.SNR, dtype=torch.long)
        elif self.newdata == True:

            self.SNR = random.randint(self.snrmin, self.snrmax)
            self.SNR1 = 2 * self.SNR - 20

            self.sgn = awgn(self.sgn, self.SNR1, Seed=self.seed)
            # if self.aug == True:
            # functions = [sig_time_warping, sig_rotate, sig_reserve]
            # num_select = np.random.randint(0, 1)
            # selected_functions = random.sample(functions, num_select)
            # for function in selected_functions:
            #     self.sgn = function(self.sgn)

            if self.trans_choose == 'pwvd':
                img = pwvd(self.sgn, self.norm, self.resize_is, RGB_is=self.RGB_is)  # 1,224,224
            elif self.trans_choose == 'stft':
                img = stp(data=self.sgn, norm=self.norm, resize_is=self.resize_is)
            elif self.trans_choose == 'wave':
                img = wave(data=self.sgn, norm=self.norm, resize_is=self.resize_is)  # 1,224,224
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

