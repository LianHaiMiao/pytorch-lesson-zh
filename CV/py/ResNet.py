import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
# 我们这里以 ResNets34 为例子

# 先实现一个Block
class Block(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1, same_shape=True):
        super(Block, self).__init__()
        self.same_shape = same_shape
        if not same_shape:
            strides = 2
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel)
    def forward(self, x):
        out = self.block(x)
        if not self.same_shape:
            x = self.bn3(self.conv3(x))
        return F.relu(out + x)

# 开始实现 ResNets34
class ResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()
        # 最开始的几层
        self.pre = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1))
        # 从论文的图中，可以看到，我们有3，4，6，3个block
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self,  in_channel, out_channel, block_num, stride=1):
        layers = []
        if stride != 1:
            layers.append(Block(in_channel, out_channel, stride, same_shape=False))
        else:
            layers.append(Block(in_channel, out_channel, stride))
        
        for i in range(1, block_num):
            layers.append(Block(out_channel, out_channel))
        return nn.Sequential(*layers)
    
    # 在jupyter notebook中，可以尝试输出每一层的size，来查看每一层的输入、输出是否正确。
    def forward(self, x):
        x = self.pre(x)
        print("pre层的size是：", x.size())
        x = self.layer1(x)
        print("layer1的size是：", x.size())
        x = self.layer2(x)
        print("layer2的size是：", x.size())
        x = self.layer3(x)
        print("layer3的size是：", x.size())
        x = self.layer4(x)
        print("layer4的size是：", x.size())
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        print("最后的结果是：", x.size())
        return self.fc(x)



resnet = ResNet34()
x = Variable(torch.randn(1, 3, 224, 224))
print(resnet(x).size())

# 实现 ResNets50 以上，需要先实现一个Block， ResNets50 做为练习，可以尝试完成一下。
class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1, same_shape=True, bottle=True):
        super(Bottleneck, self).__init__()
        self.same_shape = same_shape
        self.bottle = bottle
        if not same_shape:
            strides = 2
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel*4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel*4)
        )
        if not same_shape or not bottle:
            self.conv4 = nn.Conv2d(in_channel, out_channel*4, kernel_size=1, stride=strides, bias=False)
            self.bn4 = nn.BatchNorm2d(out_channel*4)
            print(self.conv4)
    def forward(self, x):
        print(x.size())
        out = self.block(x)
        print(out.size())
        if not self.same_shape or not self.bottle:
            x = self.bn4(self.conv4(x))
        return F.relu(out + x)