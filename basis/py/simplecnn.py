# 构建模型： LeNet-5 模型
# 接下来我们会使用LeNet模型来处理MNIST数据集。
# LeNet-5的模型结构如下图所示：

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, in_dim, n_class):
        super(LeNet5, self).__init__()
        # 从结构图中可以看出，第一层：卷积层输入是1 channel, 输出是 6 channel, kennel_size = (5,5)
        self.conv1 = nn.Conv2d(in_dim, 6, 5, padding=2)
        # 第二层：依旧是 卷积层， 输入 6 channel 输出 6 channel , kennel_size = (5,5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 第三层：全连接层（线性表示）
        self.fc1 = nn.Linear(16*5*5, 120)
        # 第四层：全连接层
        self.fc2 = nn.Linear(120, 84)
        # 第五层：输出层
        self.fc3 = nn.Linear(84, n_class)
    # 向前传播
    def forward(self, x):
        # Subsampling 1 process
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        
        # Subsampling 2 process
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        
        # -1的话，意味着最后的相乘为维数
        x = x.view(-1, self.num_flat_features(x))
        # full connect 1
        x = F.relu(self.fc1(x))
        # full connect 2
        x = F.relu(self.fc2(x))
        # full connect 3
        x = self.fc3(x)
        return x
    
    # 6 channel 卷积层 转全连接层的处理
    def num_flat_features(self, x):
        # 得到 channel * iW * iH 的值
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

leNet = LeNet5(1, 10)
print(leNet)

# 数据集是 MNIST dataset
# 我们会使用 torchvision 来加载数据
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

# mini-batch
batch_size = 128

# 未下载数据，使用True表示下载数据
DOWNLOAD = False 

train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=DOWNLOAD)

test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 现在，我们有了模型，也有了数据集，就让我们开始进行测试

import torch.optim as optim

# hyper-parameters
learning_rate = 0.0001
num_epoches = 2
use_gpu = torch.cuda.is_available()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(leNet.parameters(), lr = learning_rate)
tt = 0
# 开始训练
for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        tt +=1
        img, label = data
        img = Variable(img)
        label = Variable(label)
        # 向前传播
        out = leNet(img)
        loss = criterion(out, label)
        running_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.data[0]
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (batch_size * i),
                running_acc / (batch_size * i)))
print("Done!")