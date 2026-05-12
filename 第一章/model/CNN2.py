import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from tool.signeltoimage import *
import torch.fft
import math

class CNN2(nn.Module):
    def __init__(self, numclass, in_features=None):
        """
        CNN2 模型

        Args:
            numclass: 分类类别数
            in_features: 全连接层输入特征数。
                         如果为 None，则在 forward 中根据输入动态计算（推荐）。
                         如果指定具体数值，则使用固定值（如 1024 或 2304）。
        """
        super(CNN2, self).__init__()
        self._in_features = in_features
        self.conv1=nn.Sequential(nn.Conv2d(1, 256, kernel_size=(1,7), stride=1,padding=(0,3) ,bias=True),
                                # nn.Dropout(0.2),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=(1, 2)),
                                 nn.BatchNorm2d(256)
                                # nn.Dropout(0.2)
                                )

        self.conv2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(1,7), stride=1,padding=(0,3) ,bias=True),
                                   #nn.Dropout(0.2),
                                   nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(1, 2)),
                                   nn.BatchNorm2d(128)
                                   # nn.Dropout(0.2)
                                )

        self.conv3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(1,7), stride=1,padding=(0,3) ,bias=True),
                                   #nn.Dropout(0.2),
                                   nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(1, 2)),
                                   nn.BatchNorm2d(64)
                                   # nn.Dropout(0.2)
                                   )

        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(1,7), stride=1,padding=(0,3) ,bias=True),
                                   #nn.Dropout(0.2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(1, 2)),
                                   nn.BatchNorm2d(64)
                                   # nn.Dropout(0.2)
                                   )

        # fc1 在第一次 forward 时动态构建
        self.fc1 = None
        self.fc2 = nn.Linear(in_features=256, out_features=numclass)

    def forward(self, x, y, z):
        y = y.unsqueeze(1)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = y.view(y.size(0), -1)

        # 动态计算 fc1 的输入维度（仅在第一次 forward 时构建）
        if self.fc1 is None:
            in_features = self._in_features if self._in_features is not None else y.size(1)
            self.fc1 = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=256),
                nn.ReLU(),
                nn.Dropout(0.2)
            ).to(y.device)

        y = self.fc1(y)
        y = self.fc2(y)
        return y


# net1= CNN2(10,11904)
# def count_parameters_in_MB(model):
#     return sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)
#
# num_params = count_parameters_in_MB(net1)
# print(f'Number of parameters: {num_params}')
#a=torch.randn((2,1,128,128))
#b=torch.randn((3,2,1500))
#c=torch.randn((2,1,17))
#
#net1(a,b,c)