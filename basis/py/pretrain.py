import torch
import torch.nn as nn

# 参数初始化
class LeNet5(nn.Module):
    def __init__(self, in_dim, n_class):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)
        
        # 参数初始化函数，还有很多很多，可以上官网查api.
        # torch.nn.init.constant(tensor, val)
        # torch.nn.init.normal(tensor, mean=0, std=1)
        # torch.nn.init.xavier_uniform(tensor, gain=1)
        
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



# 使用预训练的AlexNet模型来对CIFAR10进行训练
# 现在，假设，我们知道AlexNet是个好用的模型，我们想用它来对CIFAR10进行训练，但是我们不想自己训练，想用别人训练好的，我们该怎么做呢？
# 分三步：
# 1、加载预训练模型
# 2、对模型结构进行更改
# 3、重新训练，我们需要进行训练的那几层

from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.optim as optim

# -----------------get Alexnet model-------------------------
def getAlexNet(DOWNLOAD=True):
    alexnet = models.alexnet(pretrained=DOWNLOAD)
    return alexnet


# 把 tensor 变成 Variable
def to_var(x):
    x = Variable(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

# 是否下载
DOWNLOAD = True

# 需要分类的类别
CLASS_NUM = 10

# 下载预训练好的AlexNet模型
alexnet = getAlexNet(DOWNLOAD)



# -----------------修改预训练好的 Alexnet 模型中的分类层-------------------------
alexnet.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features=9216, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=4096, out_features=CLASS_NUM, bias=True)
)


# 使用GPU
if torch.cuda.is_available():
    alexnet = alexnet.cuda()

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载数据
train_data = datasets.CIFAR10('./data', train=True, transform=transform, download=True)
test_data = datasets.CIFAR10('./data', train=False, transform=transform, download=False)

train_data_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=256, shuffle=True)

# optim
learning_rate = 0.0001
num_epoches = 5
criterion = nn.CrossEntropyLoss()

# 训练的时候，我们只更新 classifier 层的参数
optimizer = optim.Adam(alexnet.classifier.parameters(), lr=learning_rate)

# training
alexnet.train()
for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))
    runnin_acc = 0.0
    running_loss = 0.0
    for data, label in train_data_loader:
        img = to_var(data)
        label = to_var(label)
        out = alexnet(img)
        loss = criterion(out, label)
        running_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        runnin_acc += num_correct.data[0]
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Acc: {:.6f}".format(runnin_acc / len(train_data)))

# evaluation
alexnet.eval()
runnin_acc = 0.0
for data, label in test_data_loader:
    img = to_var(data)
    label = to_var(label)
    out = alexnet(img)
    loss = criterion(out, label)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    accuracy = (pred == label).float().mean()
    runnin_acc += num_correct.data[0]
print("Acc: {:.6f}".format(runnin_acc / len(test_data)))


print("Done!")







