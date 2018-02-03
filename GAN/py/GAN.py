import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.utils.data import DataLoader

# 超参数
DOWNLOAD = True
use_GPU = torch.cuda.is_available()
Learning_Rate = 0.0003
EPOCH = 200

# 图像预处理
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# 使用 MNIST 数据集
mnist = datasets.MNIST(root='./data/', train=True, transform=transform, download=DOWNLOAD)

# 包装成 DataLoader
data_loader = DataLoader(dataset=mnist, batch_size=100, shuffle=True)

# Discriminator Model
D = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid())

if use_GPU:
    D = D.cuda()

# Generator Model
G = nn.Sequential(
    nn.Linear(64, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 784),
    nn.Tanh())



if use_GPU:
    G = G.cuda()


# Loss and Optimizer
loss_fn = nn.BCELoss()

d_optimizer = torch.optim.Adam(D.parameters(), lr=Learning_Rate)
g_optimizer = torch.optim.Adam(G.parameters(), lr=Learning_Rate)


# Tensor to Variable
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

# start training
for epoch in range(EPOCH):
    for i, (images, _) in enumerate(data_loader):
        batch_size = images.size(0)
        images = to_var(images.view(batch_size, -1))

        # 给出正标签和负标签
        real_labels = to_var(torch.ones(batch_size))
        fake_labels = to_var(torch.zeros(batch_size))

        ###             训练 Discriminatro               ###
        # BCEloss(x, y) = -y * log(D(x)) - (1-y) * log(1 - D(x))
        # 当前 y = 1， 那么我们求得的Loss = -y * log(D(x)) 其中 x 来自于真实数据集
        outputs = D(images)
        d_loss_real = loss_fn(outputs, real_labels)
        real_score = outputs

        # 当前 y = 0， 那么我们求得的Loss = -(1-y) * log(1-D(x)) 其中 x 来自生成数据集
        z = to_var(torch.randn(batch_size, 64))
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = loss_fn(outputs, fake_labels)
        fake_score = outputs

        # 损失计算完成，开始方向传播
        d_loss = d_loss_real + d_loss_fake
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()


        ###             训练 Generator               ###
        z = to_var(torch.randn(batch_size, 64))
        fake_images = G(z)
        outputs = D(fake_images)


        # 本来我们应该是训练数据 使得 log(D(G(z))) 最大化，但是我们可以采取 minimzing log(1 - D(G(z)))的方式来进行代替
        # 这时候，我们将此时的标签看成正例即可。
        g_loss = loss_fn(outputs, real_labels)


        # 反向传播
        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()


        if (i + 1) % 300 == 0:
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, '
                  'g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f'
                  % (epoch, 200, i + 1, 600, d_loss.data[0], g_loss.data[0],
                     real_score.data.mean(), fake_score.data.mean()))

    # Save real images
    if (epoch + 1) == 1:
        images = images.view(images.size(0), 1, 28, 28)
        save_image((images.data), './data/test_GAN/real_images.png')

    # Save sampled images
    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    save_image((fake_images.data), './data/test_GAN/fake_images-%d.png' % (epoch + 1))


# 保存训练参数
torch.save(G.state_dict(), './generator.pkl')
torch.save(D.state_dict(), './discriminator.pkl')
