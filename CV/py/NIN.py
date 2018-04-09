import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


# 超参数类，用于控制各种超参数
class Config(object):
    def __init__(self):
        self.lr = 0.005
        self.batch_size = 256
        self.use_gpu = torch.cuda.is_available()
        self.DOWNLOAD = True
        self.epoch_num = 5 # 因为只是demo，就跑了2个epoch，可以自己多加几次试试结果
        self.class_num = 10 # CIFAR10 共有10类
config = Config()


# NiN提出只对通道层做全连接并且像素之间共享权重来解决上述两个问题
# 这种“一卷卷到底”最后加一个平均池化层的做法也成为了深度卷积神经网络的常用设计。
def mlpconv(in_chanels, out_chanels, kernel_size, padding, strides=1, max_pooling=True):
    layers = []
    layers += [nn.Conv2d(in_chanels, out_chanels, kernel_size=kernel_size, padding=padding, stride=strides), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(out_chanels, out_chanels, kernel_size=1, padding=0, stride=1), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(out_chanels, out_chanels, kernel_size=1, padding=0, stride=1), nn.ReLU(inplace=True)]
    if max_pooling:
        layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
    return nn.Sequential(*layers)


class NIN(nn.Module):
    """
       输入图片的尺寸一定得是 224 × 224 的
    """
    def __init__(self, class_num):
        super(NIN, self).__init__()
        self.net = nn.Sequential(
            mlpconv(3, 96, 11, 0, strides=4),
            mlpconv(96, 256, 5, 2),
            mlpconv(256, 384, 3, 1),
            nn.Dropout(0.5),
            # 目标类为10类
            mlpconv(384, 10, 3, 1, max_pooling=False),
            # 输入为 batch_size x 10 x 5 x 5, 通过AvgPool2D转成
            # batch_size x 10 x 1 x 1。
            nn.AvgPool2d(kernel_size=5, stride=1)
        )
        self.class_num = class_num

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, self.class_num )
        return out


# 图像预处理，因为NIN是使用224 * 224大小的图片，但是 CIFAR10 只有32 * 32
transform = transforms.Compose([
    transforms.Resize(224), # 缩放到 224 * 224 大小
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 归一化
])


# 下载 CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transform, download=config.DOWNLOAD)
test_dataset = torchvision.datasets.CIFAR10(root='./data/', train=False, transform=transform)

# dataloader

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=config.batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=config.batch_size,
                         shuffle=False)


nin = NIN(config.class_num)

# 是否使用GPU
if config.use_gpu:
    nin = nin.cuda()

# loss and optimizer
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(nin.parameters(), lr=config.lr)

for epoch in range(config.epoch_num):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)

        if config.use_gpu:
            images = images.cuda()
            labels = labels.cuda()

        # forward + backward + optimize
        optimizer.zero_grad()
        y_pred = nin(images)

        loss = loss_fn(y_pred, labels)

        loss.backward()

        optimizer.step()
        
        if (i + 1) % 100 == 0:
            print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (epoch + 1, config.epoch_num, i + 1, 100, loss.data[0]))

        # 计算训练精确度
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
    # 结束一次迭代
    print('Accuracy of the model on the train images: %d %%' % (100 * correct / total))
 
    # Decaying Learning Rate
    if (epoch+1) % 2 == 0:
        config.lr /= 3
        optimizer = torch.optim.Adam(nin.parameters(), lr=config.lr)


# Test
nin.eval()

correct = 0
total = 0

for images, labels in test_loader:
    images = Variable(images)
    labels = Variable(labels)
    if config.use_gpu:
        images = images.cuda()
        labels = labels.cuda()
    y_pred = nin(images)
    _, predicted = torch.max(y_pred.data, 1)
    total += labels.size(0)
    temp = (predicted == labels.data).sum()
    correct += temp


print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))