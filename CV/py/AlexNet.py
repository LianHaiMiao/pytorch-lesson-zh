import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class AlextNet(nn.Module):
    def __init__(self, in_channel, n_class):
        super(AlextNet, self).__init__()
        # 第一阶段
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # 第二阶段
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # 第三阶段
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # 第四阶段 全连接层
        self.fc = nn.Sequential(
            nn.Linear(1*1*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, n_class) # AlexNet上面是1000 ...如果测试的话用MNIST则可以使用10
        )
    # 向前传播
    def forward(self, x):
        con1_x = self.conv1(x)
        con2_x = self.conv2(con1_x)
        con3_x = self.conv3(con2_x)
        lin_x = con3_x.view(con3_x.size(0), -1)
        y_hat = self.fc(lin_x)
        return y_hat


alex = AlextNet(3, 10) # in_channel, class_num
print(alex)

# 图像预处理，因为Alex 是使用 227 * 227 大小的图片，但是 CIFAR10 只有 32 * 32 ,经过测试， 227 * 227 的效果不好。
# 所以这里， 我们将图像放大到 96*96
transform = transforms.Compose([
    transforms.Scale(96),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 超参数
DOWNLOAD = True
BATCH_SIZE = 256
EPOCH = 5
learning_rate = 0.001

# 是否使用GPU
use_gpu = True

# CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transform, download=DOWNLOAD)

test_dataset = torchvision.datasets.CIFAR10(root='./data/', train=False, transform=transform)

# Data Loader
train_loader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False)

# 定义模型
alex = AlextNet(3, 10)
if use_gpu:
    alex = alex.cuda()

# loss and optimizer

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(alex.parameters(), lr=learning_rate)

# Training
alex.train()

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
        y_pred = alex(images)

        loss = loss_fn(y_pred, labels)

        loss.backward()

        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (epoch + 1, EPOCH, i + 1, 100, loss.data[0]))

        # 计算训练精确度
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
    print('Accuracy of the model on the train images: %d %%' % (100 * correct / total))

    # Decaying Learning Rate
    if (epoch+1) % 2 == 0:
        learning_rate /= 3
        optimizer = torch.optim.Adam(alex.parameters(), lr=learning_rate)


# Test
alex.eval()

correct = 0
total = 0

for images, labels in test_loader:
    images = Variable(images)
    labels = Variable(labels)
    if use_gpu:
        images = images.cuda()
        labels = labels.cuda()

    y_pred = alex(images)
    _, predicted = torch.max(y_pred.data, 1)
    total += labels.size(0)
    temp = (predicted == labels.data).sum()
    correct += temp

print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))


