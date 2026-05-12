import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from tool.signeltoimage import *
import torch.fft
import math

class PETCGDNN(nn.Module):
    def __init__(self, numclass):
        super(PETCGDNN, self).__init__()
        self.fc1=nn.Linear(256,1)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 75, kernel_size=(2, 8), stride=1, padding=(0, 0), bias=True),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(75),
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(75, 25, kernel_size=(1, 5), stride=1, padding=(0, 0), bias=True),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(25),
                                   )
        self.gru=nn.GRU(input_size=25, hidden_size=128,batch_first=True)

        # fc2 在第一次 forward 时根据展平维度动态构建
        self.fc2 = None
        self.numclass = numclass

    def forward(self, x, y, z):
        y = y.unsqueeze(1)
        input1=y[:,:,0,:]
        input2 = y[:, :, 1, :]
        y=y.view(y.size(0), -1)
        y=self.fc1(y)
        cos1=torch.cos(y)
        cos1 = cos1.unsqueeze(1)
        sin1=torch.sin(y)
        sin1 = sin1.unsqueeze(1)
        x11=torch.matmul(cos1,input1)
        x12 = torch.matmul(sin1, input2)
        x21 = torch.matmul(cos1, input2)
        x22 = torch.matmul(sin1, input1)
        y1=x11+x12
        y2 = x21 - x22
        x3=torch.cat((y1,y2),dim=1)
        x3 = x3.unsqueeze(1)
        x3=self.conv1(x3)
        x3 = self.conv2(x3)
        x3 = x3.squeeze(2)
        x3 = x3.transpose(1, 2)
        self.gru.flatten_parameters()
        x3, (hidden)=self.gru(x3)
        # x3 = x3[:, -1, :]
        x3 = x3.contiguous().view(x3.size()[0], -1)

        # 动态构建 fc2（仅在第一次 forward 时）
        if self.fc2 is None:
            in_features = x3.size(1)
            self.fc2 = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, self.numclass)
            ).to(x3.device)

        y=self.fc2(x3)
        return y

#
# net1= PETCGDNN(10)
# a=torch.randn((2,1,128,128))
# b=torch.randn((3,2,128))
# c=torch.randn((2,1,17))
#
# net1(a,b,c)
# def count_parameters_in_MB(model):
#     return sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)
#
# num_params = count_parameters_in_MB(net1)
# print(f'Number of parameters: {num_params}')