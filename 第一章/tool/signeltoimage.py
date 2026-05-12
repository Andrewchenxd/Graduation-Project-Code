from scipy.signal import spectrogram
import numpy as np
import matplotlib.pyplot as plt
from tftb.generators import atoms
from tftb.processing import *
import scipy.io as scio
from scipy.signal import kaiser, hamming
from scipy.signal import resample,hilbert2
import cv2
def resampe(sgn,samplenum=2):
    newsgn=np.zeros((sgn.shape[0],sgn.shape[1]// samplenum))
    newsgn[0,:]=resample(sgn[0,:], sgn.shape[1] // samplenum)
    newsgn[1, :] = resample(sgn[1, :], sgn.shape[1] // samplenum)
    return newsgn
def toIQ(sgn):
    newsgn = np.zeros((2, sgn.shape[0]))
    y = hilbert2(sgn)
    newsgn[0] = np.real(y)
    newsgn[1] = np.imag(y)
    return newsgn
'''
实现功能：输入2维雷达信号sgn，大小如（2,128），进行短时傅里叶变换
'''
def stp(data,nperseg=10,noverlap=9):
    fs = 200*1000
    sgn=data[0,:]+data[1,:]*1j
    f, tt, p = spectrogram(x=sgn, fs=fs, nperseg=nperseg,
                           noverlap=noverlap,
                           return_onesided=False)
    # ampLog =np.abs(np.log(1 + np.abs(p)))
    #ampLog=np.abs(p)+np.random.normal(0,size=(51,290))
    normImg =p/np.linalg.norm(p)
    normImg =np.expand_dims(normImg, axis=0)
    # threshold = 0.04
    # normImg[np.where(normImg > threshold)] = threshold
    # plt.pcolormesh(normImg)
    # normImg =  (p - p.min()) / (p.max() - p.min() )  # 归一化
    #
    # plt.figure(figsize=(10, 5))
    # plt.pcolormesh(tt, f, normImg, cmap='hot', shading='goudard')plt.pcolormesh(tt, f, normImg, cmap='hot', shading='goudard')
    # plt.xlabel('Time (s)', fontsize=20)
    # plt.ylabel('Frequency (Hz)', fontsize=20)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.show()
    return normImg


def pwvd(data,norm='no',resize_is=False):
    t = np.linspace(0, 1, data.shape[1])
    sgn = data[0, :] + data[1, :] * 1j
    spec = PseudoWignerVilleDistribution(sgn, timestamps=t)
    # spec = smoothed_pseudo_wigner_ville(sgn, timestamps=t)
    img,_,_=spec.run()
    if resize_is==True:
        img=cv2.resize(img, (224,224))
    if norm=='maxmin':
        # img = np.abs(np.log(1 + np.abs(img)))
        ampLog=img
        img = (ampLog - ampLog.min()) / (ampLog.max() - ampLog.min())
    img=np.expand_dims(img, axis=0)

    return img

# data37=scio.loadmat(r'C:\planedemo\电磁论文\data37\RML2016.10a_train_data.mat')['data37']
# sgn=data37[0,:,:]
# a=psvd(sgn)

def spwvd(data):
    twindow = hamming(13)
    fwindow = hamming(33)
    sgn = data[0, :] + data[1, :] * 1j
    img = smoothed_pseudo_wigner_ville(sgn, twindow=twindow, fwindow=fwindow,
                                       freq_bins=128)
    img = np.expand_dims(img, axis=0)
    return img


# import matplotlib.pyplot as plt
# from tftb.generators import atoms
# from tftb.processing import *
# import scipy.io as scio
# from scipy.signal import kaiser, hamming
# data37=scio.loadmat(r'C:\planedemo\电磁论文\data37\RML2016.10a_train_data.mat')['data37']
# sgn=data37[3,:,:]
#
# sgnf=sgn[0, :] + sgn[1, :] * 1j
# spec = PseudoWignerVilleDistribution(sgnf)
# img,_,_=spec.run()
# plt.pcolormesh(img)
# twindow = hamming(13)
# fwindow = hamming(33)
# # twindow = kaiser(13, 2 * np.pi)
# # fwindow = kaiser(33, 2 * np.pi)
# img1 = smoothed_pseudo_wigner_ville(sgnf, twindow=twindow, fwindow=fwindow,freq_bins=128)
# normImg = (img1 - img1.min()) / (img1.max() - img1.min())
# plt.pcolormesh(normImg)

