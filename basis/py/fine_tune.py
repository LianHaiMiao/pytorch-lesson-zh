# 1.对层进行处理

# 现在，我们得到了一个 LeNet，实例化之后打印，结果如下所示：

# ```
# LeNet5(
#   (conv1): Conv2d(224, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#   (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
#   (fc1): Linear(in_features=400, out_features=120, bias=True)
#   (fc2): Linear(in_features=120, out_features=84, bias=True)
#   (fc3): Linear(in_features=84, out_features=10, bias=True)
# )
# ```

# 然后，我们现在需要把所有的带有 "conv" 的层全部删除（替换）我们该怎么做呢？

# 我们需要重新写一个模型嘛？

# NOOOOO！！！

# 不需要， Pytorch 的层是可以随意替换的！！！我们随时可以增加，修改。




import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, in_dim, n_class):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)
        
        # 参数初始化函数
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.xavier_normal(p.weight.data)
            elif isinstance(p, nn.Linear):
                nn.init.normal(p.weight.data)

    # 向前传播
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



import re

# 实例化 LeNet

lenet = LeNet5(224, 10)

del_list = [] # 将所有需要删除的层的名字放进这列表里面
for name, module in lenet.named_children():
    # 对所有的层的名字进行匹配，如果层中含有conv则把名字放入等待删除列表中
    if re.match("conv", name) != None:
        del_list.append(name)

# 获取了所有带 "conv" 层的名字了，现在开始删除
for name in del_list:
    delattr(lenet, name)




# print(lenet) 得到的结果如下所示

# ```
# LeNet5(|
#   (fc1): Linear(in_features=400, out_features=120, bias=True)
#   (fc2): Linear(in_features=120, out_features=84, bias=True)
#   (fc3): Linear(in_features=84, out_features=10, bias=True)
# )
# ```

# ----------------------------------------------- 分割线 ---------------------------------------------------------

# 2.改变预训练模型中的某 sequential 层里面的某一层

# 这里我们以Alexnet为例

# ```
# AlexNet(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
#     (1): ReLU(inplace)
#     (2): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
#     (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (4): ReLU(inplace)
#     (5): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
#     (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (7): ReLU(inplace)
#     (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (9): ReLU(inplace)
#     (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace)
#     (12): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
#   )
#   (classifier): Sequential(
#     (0): Dropout(p=0.5)
#     (1): Linear(in_features=9216, out_features=4096, bias=True)
#     (2): ReLU(inplace)
#     (3): Dropout(p=0.5)
#     (4): Linear(in_features=4096, out_features=4096, bias=True)
#     (5): ReLU(inplace)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )
# ```

# 某一天，我们突然发现了 AvgPool 的效果比 Maxpool 要好，于是我们想要改变把 AlextNet中的所有 Maxpool 用 AvgPool来代替该怎么办呢？

# > 查看 Alexnet 模型中的 Maxpool 的位置，我们可以清晰的发现 Maxpool 处于 features 下面的 (2)、(5)、(12)的位置上


# 于是，我们改变 features 层下面的 (2)、(5)、(12) 就行了，具体操作如下所示




from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.optim as optim


alexnet = models.alexnet(pretrained=False)


# 改变前的 alexnet
print(alexnet)

# 开始改变

alexnet.features._modules['2'] = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2), ceil_mode=False)
alexnet.features._modules['5'] = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2), ceil_mode=False)
alexnet.features._modules['12'] = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2), ceil_mode=False)

 # 改变后的 alexnet
print(alexnet)