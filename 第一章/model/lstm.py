import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from tool.signeltoimage import *
import torch.fft
import math
class lstm2(nn.Module):

    def __init__(self, output_size=128, numclass=10):
        super(lstm2, self).__init__()
        # lstm的输入 #batch,seq_len(词的长度）, input_size(词的维数）
        self.rnn1 = nn.LSTM(input_size=2, hidden_size=output_size, batch_first=True)
        self.norm1=nn.Sequential(
                                 nn.LayerNorm(output_size),
                                 # nn.ReLU(),
                                 )
        self.rnn2 = nn.LSTM(input_size=output_size, hidden_size=output_size, batch_first=True)
        self.norm2 = nn.Sequential(
                                   nn.LayerNorm(output_size),
                                   # nn.ReLU(),
                                   )
        # fc 在第一次 forward 时根据展平维度动态构建
        self.fc = None
        self.numclass = numclass

    def forward(self, x, y, z):
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()
        y = y.transpose(1, 2)

        out, (hidden, cell) = self.rnn1(
            y)  # y.shape : batch,seq_len,hidden_size , hn.shape and cn.shape : num_layes * direction_numbers,batch,hidden_size
        out=self.norm1(out)
        out, (hidden, cell) = self.rnn2(out)
        out = self.norm2(out)
        # out =out[:, -1, :]
        out = out.contiguous().view(out.size()[0], -1)

        # 动态构建 fc（仅在第一次 forward 时）
        if self.fc is None:
            in_features = out.size(1)
            self.fc = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, self.numclass)
            ).to(out.device)

        out = self.fc(out)
        return out
if __name__ == '__main__':
    net1= lstm2(128,10)
    a=torch.randn((2,1,128,128))
    b=torch.randn((2,2,300))
    c=torch.randn((2,1,17))
    
    net1(a,b,c)











