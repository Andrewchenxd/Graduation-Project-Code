import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1d(nn.Module):
    def __init__(self, block=Bottleneck, layers=[2,2,2,2],numclass=11):
        super(ResNet1d, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.mlp_head=nn.Sequential(nn.Linear(512,256),
                                    nn.GELU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(256,numclass))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool1d(out, out.size(2))
        out = out.view(out.size(0), -1)
        out=self.mlp_head(out)
        return out



class ResNet1d_fuse(nn.Module):
    def __init__(self, block=Bottleneck, layers=[2,2,2,2],numclass=11):
        super(ResNet1d_fuse, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv_n1 = nn.Conv1d(2, 32, kernel_size=1, stride=1)
        self.conv_n2 = nn.Conv1d(32, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_n1 = nn.BatchNorm1d(32)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.mlp_head=nn.Sequential(nn.Linear(512,256),
                                    nn.GELU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(256,numclass))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x_pred,x_org):
        out = F.relu(self.bn1(self.conv1(x_pred)))
        out_n=F.relu(self.bn_n1(self.conv_n2(self.conv_n1(x_org))))
        out=torch.concatenate([out,out_n],1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool1d(out, out.size(2))
        out = out.view(out.size(0), -1)
        out=self.mlp_head(out)
        return out

class ResNet1d_enc(nn.Module):
    def __init__(self, block=Bottleneck, layers=[2,2,2,2],numclass=11):
        super(ResNet1d_enc, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.mlp_head=nn.Sequential(nn.Linear(512 , 256),
                                    nn.GELU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(256 , numclass))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool1d(out, out.size(2))
        out = out.view(out.size(0), -1)
        out=self.mlp_head(out)
        return out



if __name__ == '__main__':
    net1= ResNet1d_enc(block=Bottleneck,layers=[2,2,2,2])
    a=torch.randn((2,1,128,128))
    b=torch.randn((3,768,32))
    c=torch.randn((3,2,128))

    out=net1(b)
    print(out.shape)