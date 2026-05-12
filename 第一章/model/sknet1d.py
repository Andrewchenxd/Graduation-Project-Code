import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from tool.signeltoimage import *
import torch.fft
import math
torch.manual_seed(42)
torch.cuda.manual_seed(42)

class SKConv(nn.Module):
    def __init__(self, channels, branches=2, groups=32, reduce=16, stride=1, len=32):
        super(SKConv, self).__init__()
        len = max(channels // reduce, len)
        self.convs = nn.ModuleList([])
        for i in range(branches):
            self.convs.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i,
                          groups=groups, bias=False),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, len, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(len),
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([])
        for i in range(branches):
            self.fcs.append(
                nn.Conv1d(len, channels, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = [conv(x) for conv in self.convs]
        x = torch.stack(x, dim=1)
        attention = torch.sum(x, dim=1)
        attention = self.gap(attention)
        attention = self.fc(attention)
        attention = [fc(attention) for fc in self.fcs]
        attention = torch.stack(attention, dim=1)
        attention = self.softmax(attention)
        x = torch.sum(x * attention, dim=1)
        return x

class SKUnit(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, branches=2, group=32, reduce=16, stride=1, len=32):
        super(SKUnit, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = SKConv(mid_channels, branches=branches, groups=group, reduce=reduce, stride=stride, len=len)

        self.conv3 = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        if in_channels == out_channels:  # when dim not change, input_features could be added directly to out
            self.shortcut = nn.Sequential()
        else:  # when dim not change, input_features should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        residual = self.shortcut(residual)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += residual
        return self.relu(x)

class sknet1d(nn.Module):
    def __init__(self, num_classes, hidden_size=64, num_block_lists=[3, 4, 6, 3]):
        super(sknet1d, self).__init__()

        self.basic_conv = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.stage_1 = self._make_layer(64, 128, 256, nums_block=num_block_lists[0], stride=1)
        self.stage_2 = self._make_layer(256, 256, 512, nums_block=num_block_lists[1], stride=2)
        self.stage_3 = self._make_layer(512, 512, 1024, nums_block=num_block_lists[2], stride=2)
        self.stage_4 = self._make_layer(1024, 1024, 2048, nums_block=num_block_lists[3], stride=2)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gap1 = nn.AdaptiveAvgPool1d(1)
        self.gap2 = nn.AdaptiveAvgPool1d(1)
        self.gap3 = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(1024, num_classes)
        self.fc = nn.Linear(in_features=2048, out_features=1024)
        self.drop=nn.Dropout(0.2)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


    def _make_layer(self, in_channels, mid_channels, out_channels, nums_block, stride=1):
        layers = [SKUnit(in_channels, mid_channels, out_channels, stride=stride)]
        for _ in range(1, nums_block):
            layers.append(SKUnit(out_channels, mid_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, y,z):
        x = self.basic_conv(y)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.gap(x)
        x=x.view(x.size(0), -1)
        x=self.fc(x)
        x=self.drop(x)
        x = self.classifier(x)
        return x


# net1= sknet1d(10,64)
# a=torch.randn((2,1,128,128))
# b=torch.randn((2,2,128))
# c=torch.randn((2,1,17))
#
# net1(a,b,c)