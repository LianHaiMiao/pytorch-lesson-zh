import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


# 图像预处理，因为VGG是使用224 * 224大小的图片，但是MNIST只有32 * 32, 为了能快点跑出结果，
# 我们将它们放大到96*96，而不是原始论文的224 * 224
transform = transforms.Compose([
    transforms.Scale(96), # 缩放到 96 * 96 大小
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 归一化
])

# 超参数
DOWNLOAD = True
BATCH_SIZE = 256
EPOCH = 2
learning_rate = 0.001

# 是否使用GPU
use_gpu = True

# 下载 CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transform, download=DOWNLOAD)
test_dataset = torchvision.datasets.CIFAR10(root='./data/', train=False, transform=transform)

# dataloader

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, conv_features):
        super(VGG, self).__init__()
        self.conv_features = conv_features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4608, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv_features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 构建 循环的 conv层
def make_layers(struct, in_channels=1, batch_norm=False):
    layers = []
    for out_channels in struct:
        if out_channels == 'pooling':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = out_channels
    return nn.Sequential(*layers)


# 模型初始化 
vgg_conv_layers = [64, 64, 'pooling', 128, 128, 'pooling', 256, 256, 256, 'pooling', 512, 512, 512, 'pooling', 512, 512, 512, 'pooling']

# 初始通道—— 三通道
vgg16 = VGG(make_layers(vgg_conv_layers, in_channels=3))

# 是否使用GPU
if use_gpu:
    vgg16 = vgg16.cuda()
    
# loss and optimizer
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(vgg16.parameters(), lr=learning_rate)

# Training
vgg16.train()

for epoch in range(EPOCH):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        
        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()
        
        # forward + backward + optimize
        optimizer.zero_grad()
        y_pred = vgg16(images)

        loss = loss_fn(y_pred, labels)

        loss.backward()

        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (epoch + 1, EPOCH, i + 1, 200, loss.data[0]))

        # 计算训练精确度
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
    
    # 结束一次迭代
    print('Accuracy of the model on the train images: %d %%' % (100 * correct / total))
    
    # Decaying Learning Rate
    if (epoch+1) % 2 == 0:
        learning_rate /= 3
        optimizer = torch.optim.Adam(vgg16.parameters(), lr=learning_rate)


# Test
vgg16.eval()

correct = 0
total = 0

for images, labels in test_loader:
    images = Variable(images)
    labels = Variable(labels)
    if use_gpu:
        images = images.cuda()
        labels = labels.cuda()
    y_pred = vgg16(images)
    _, predicted = torch.max(y_pred.data, 1)
    total += labels.size(0)
    temp = (predicted == labels.data).sum()
    correct += temp


    print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))





