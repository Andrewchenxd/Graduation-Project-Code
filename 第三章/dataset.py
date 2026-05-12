import time
import numpy as np
import torch
import random
import scipy.io as scio
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torch.utils.data import Dataset  
from tool.signal_aug import *
from tool.sig_tools import *
from imgaug import augmenters as iaa
import os
from tqdm import tqdm
import h5py
import pickle


def l2_normalize(x, axis=-1):
    y = np.sum(x ** 2, axis, keepdims=True)
    return x / np.sqrt(y)


class SigDataSet_freq_units2(Dataset):
    def __init__(self, data=None, label=None, snrs=None, adsbis=False, newdata=False, snr_range=[1, 7], resample_is=False, samplenum=15,
                 sgn_norm='no',img_norm='no',imgaug=False,freq_choose='stft'
                 , chazhi=False, chazhinum=2, is_DAE=False, resize_is=False, return_label=False, sgnaug=False
                 , sgn_expend=False, RGB_is=False,zhenshiSNR=False,return_img=False,window=None,Seed=1,dataset_name=None):
        super().__init__()
        self.data =data
        self.labels =label
        self.snrs=snrs
        self.snrmin = snr_range[0]
        self.snrmax = snr_range[1]
        self.adsbis = adsbis
        self.resample_is = resample_is
        self.norm = sgn_norm
        self.img_norm=img_norm
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
        self.return_img=return_img
        self.RGB_is=RGB_is
        self.freq_choose=freq_choose
        self.zhenshiSNR=zhenshiSNR
        self.window=window
        self.Seed=Seed
        self.dataset_name=dataset_name
        if (adsbis == False) and (newdata == False):
            self.snr = []

        if (adsbis == True) or (newdata == True):
            self.rml = False
        self.newdata = newdata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.sgn_expend == False:
            self.sgn = self.data[item]
            # self.sgn = filter(self.sgn, filiter='low', filiter_threshold=0.85, filiter_size=0.0,
            #        middle_zero=True, freq_smooth=True,return_IQ=True)
            self.sgn_noise = np.copy(self.sgn)
        else:
            np.random.seed(self.Seed)
            if np.random.random() <= 0.5:
                functions = [sig_rotate, sig_reserve,sig_time_warp]
                selected_functions = random.sample(functions, 1)

                for function in selected_functions:
                    self.sgn = function(self.data[item])
            else:
                self.sgn = self.data[item]
            self.sgn_noise = np.copy(self.sgn)

        if self.rml == True:
            np.random.seed(self.Seed)
            self.SNR = random.randint(self.snrmin, self.snrmax)
            self.SNR1 = self.SNR - 20
            if self.sgnaug == False:
                self.sgn_noise = awgn(self.sgn_noise, self.SNR1,self.zhenshiSNR,self.Seed)
                # random_nums1 = np.random.uniform(low=0.9, high=0.99, size=1)
                # random_nums2 = np.random.uniform(low=0.35, high=1.6, size=1)
                # self.sgn_noise = filter(self.sgn_noise, filiter='high', filiter_threshold=random_nums1,
                #                         filiter_size=random_nums2, return_IQ=True)
            else:
                if np.random.random() <= 0.5:
                    self.sgn_noise = awgn(self.sgn_noise, self.SNR1,self.zhenshiSNR,self.Seed)
                else :
                    self.sgn_noise = rayleigh_noise(self.sgn_noise, self.SNR1,self.zhenshiSNR,self.Seed)

            self.sgn_aug = np.copy(self.sgn)
            self.sgn_fliter = np.copy(self.sgn_noise)
            if np.random.random() <= 0.6:
                random_nums1 = np.random.uniform(low=0.9, high=0.99, size=1)
                random_nums2 = np.random.uniform(low=0.4, high=1.6, size=1)
                self.sgn_aug = filter(self.sgn_aug, filiter='high', filiter_threshold=random_nums1,
                                      filiter_size=random_nums2, return_IQ=True)
            else:
                functions = [sig_rotate, sig_reserve,sig_time_warp]
                selected_functions = random.sample(functions, 1)

                for function in selected_functions:
                    self.sgn_aug = function(self.sgn_aug)
            # self.sgn =sig_time_warp(self.sgn)

            self.sgn_fliter =moving_avg_filter_numba(self.sgn_fliter,window_size=9)
            self.sgn_fliter=gaussian_filter_numba(self.sgn_fliter, sigma=2, kernel_radius=7)

            if self.resample_is == True:

                self.sgn = resampe(self.sgn, samplenum=self.samplenum)
                self.sgn_noise = resampe(self.sgn_noise, samplenum=self.samplenum)
                self.sgn_aug = resampe(self.sgn_aug, samplenum=self.samplenum)
                self.sgn_fliter = resampe(self.sgn_fliter, samplenum=self.samplenum)

            self.sgn_fliter = sgn_norm(self.sgn_fliter, normtype=self.norm)
            self.sgn = sgn_norm(self.sgn, normtype=self.norm)
            self.sgn_noise = sgn_norm(self.sgn_noise, normtype=self.norm)
            self.sgn_aug = sgn_norm(self.sgn_aug, normtype=self.norm)

            self.freq_clean=sgn_freq(self.sgn,freq_choose=self.freq_choose,window=self.window)
            self.freq_noise = sgn_freq(self.sgn_noise,freq_choose=self.freq_choose,window=self.window)

            # self.freq =sgn_norm(self.freq , normtype='maxmin-1')
            if self.return_label == False:
                if self.return_img==False:
                    return torch.tensor(self.sgn, dtype=torch.float32),\
                        torch.tensor(self.sgn_noise, dtype=torch.float32),\
                            torch.tensor(self.sgn_aug, dtype=torch.float32), \
                        torch.tensor(self.sgn_fliter, dtype=torch.float32), \
                        self.freq_clean, self.freq_noise,self.SNR1, self.labels[item],self.dataset_name
                elif self.return_img == True:
                    return torch.tensor(self.sgn, dtype=torch.float32), \
                        torch.tensor(self.sgn_noise, dtype=torch.float32), \
                        torch.tensor(img, dtype=torch.float32), torch.tensor(imgn, dtype=torch.float32)
            elif self.return_label == True:
                if self.return_img==False:
                    return torch.tensor(self.sgn, dtype=torch.float32),\
                        torch.tensor(self.sgn_noise, dtype=torch.float32),\
                            torch.tensor(self.sgn_aug, dtype=torch.float32), \
                        torch.tensor(self.sgn_fliter, dtype=torch.float32), \
                        self.snrs[item], self.labels[item],self.dataset_name
                elif self.return_img == True:
                    return torch.tensor(self.sgn, dtype=torch.float32), \
                        torch.tensor(self.sgn_noise, dtype=torch.float32), \
                        torch.tensor(img, dtype=torch.float32), \
                        self.snrs[item], self.labels[item],self.dataset_name

                    
class SigDataSet_sgn(Dataset):
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
            self.sgn = sig_time_warp(self.data[item])
            self.sgn_noise = np.copy(self.sgn)

        np.random.seed(None)
        self.SNR = random.randint(self.snrmin, self.snrmax)
        if self.newdata:
            self.SNR1 = 2*self.SNR - 20
        else:
            self.SNR1 = self.SNR - 20
        if self.sgnaug == False:
            self.sgn_noise = awgn(self.sgn_noise, self.SNR1,Seed=None)
        else:
            random_nums1 = np.random.uniform(low=0.9, high=0.99, size=1)
            random_nums2 = np.random.uniform(low=0.5, high=1.5, size=1)
            self.sgn_noise =filter(self.sgn_noise , filiter='high', filiter_threshold=random_nums1,
                                   filiter_size=random_nums2,return_IQ=True)
        if self.adsbis == True :
            self.sgn = resampe(self.sgn, samplenum=self.samplenum)
            self.sgn_noise = resampe(self.sgn_noise, samplenum=self.samplenum)
        if self.return_label == False:
            self.sgn_noise = sgn_norm(self.sgn_noise, normtype=self.norm)
            self.sgn = sgn_norm(self.sgn, normtype=self.norm)
            # self.sgn = np.expand_dims(self.sgn, 0)
            # self.sgn_noise = np.expand_dims(self.sgn_noise, 0)
            return torch.tensor(self.sgn, dtype=torch.float32), \
                   torch.tensor(self.sgn_noise, dtype=torch.float32)
        elif self.return_label == True:
            if self.sgnaug == True:
                np.random.seed(None)
                random.seed(None)
                functions = [sig_rotate, addmask, sig_reserve,
                             sig_time_warping]
                # functions = [sig_reserve, sig_rotate,sig_time_warp]
                # num_select = np.random.randint(0, 2)
                selected_functions = random.sample(functions, 1)
                for function in selected_functions:
                    self.sgn = function(self.sgn)

                # random_nums1 = np.random.uniform(low=0.9, high=0.99, size=1)
                # random_nums2 = np.random.uniform(low=0.6, high=1.4, size=1)
                # self.sgn = filter(self.sgn, filiter='high', filiter_threshold=random_nums1,
                #                       filiter_size=random_nums2, return_IQ=True)
            self.sgn = sgn_norm(self.sgn, normtype=self.norm)
            # self.sgn = np.expand_dims(self.sgn, 0)
            # self.sgn_noise = np.expand_dims(self.sgn_noise, 0)
            if self.adsbis==True:
                return torch.tensor(self.sgn, dtype=torch.float32), \
                       torch.tensor(self.labels[item], dtype=torch.long), \
                       torch.tensor(self.SNR, dtype=torch.long)
            elif self.rml==True:
                return torch.tensor(self.sgn, dtype=torch.float32), \
                       torch.tensor(self.labels[item], dtype=torch.long), \
                       torch.tensor(self.snr[item], dtype=torch.long)
            elif self.newdata == True:
                return torch.tensor(self.sgn_noise, dtype=torch.float32), \
                    torch.tensor(self.labels[item], dtype=torch.long), \
                    torch.tensor(self.SNR, dtype=torch.long)


                    