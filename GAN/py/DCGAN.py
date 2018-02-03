import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils import data
import os
from PIL import Image
from torch import optim
import torch.nn.functional as F
from torchvision.utils import save_image

# 数据：https://pan.baidu.com/s/1eSifHcA 提取码：g5qa
# 运行代码前，请先确保下载训练数据。

# 构建生成模型
class Generator(nn.Module):
    def __init__(self, noise_dim=100, conv_dim=64, g_img_size=64):
        super(Generator, self).__init__()
        self.fc = nn.ConvTranspose2d(noise_dim, conv_dim*16, kernel_size=int(g_img_size/16), stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(conv_dim*16, conv_dim*8, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_dim * 8)
        self.deconv3 = nn.ConvTranspose2d(conv_dim*8, conv_dim*4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_dim * 4)
        self.deconv4 = nn.ConvTranspose2d(conv_dim*4, conv_dim*2, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(conv_dim * 2)
        self.deconv5 = nn.ConvTranspose2d(conv_dim*2, 3, kernel_size=4, stride=2, padding=1)


    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1)
        A1 = self.fc(x)
        A2 = F.relu(self.bn2(self.deconv2(A1)))
        A3 = F.relu(self.bn3(self.deconv3(A2)))
        A4 = F.relu(self.bn4(self.deconv4(A3)))
        y_hat = F.tanh(self.deconv5(A4))
        return y_hat

# 构建判别模型
class Discriminator(nn.Module):
    def __init__(self, input_chanel=3, conv_dim=64, d_img_size=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_chanel, conv_dim * 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_dim * 4)
        self.conv3 = nn.Conv2d(conv_dim * 4, conv_dim * 8, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_dim * 8)
        self.conv4 = nn.Conv2d(conv_dim * 8, conv_dim * 16, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(conv_dim * 16)
        self.fc = nn.Conv2d(conv_dim * 16, 1, int(d_img_size / 16), 1, 0)

    def forward(self, x):
        A1 = F.leaky_relu(self.conv1(x))
        A2 = F.leaky_relu(self.bn2(self.conv2(A1)), 0.05)
        A3 = F.leaky_relu(self.bn3(self.conv3(A2)), 0.05)
        A4 = F.leaky_relu(self.bn4(self.conv4(A3)), 0.05)
        y_hat = F.sigmoid(self.fc(A4).squeeze())
        return y_hat



# 数据集处理
class ImageDataset(data.Dataset):
    def __init__(self, path, transform=None):
        """
            path 是存在图像的文件夹。
        """
        self.images = list(map(lambda x: os.path.join(path, x), os.listdir(path)))
        self.transform = transform

    def __getitem__(self, index):
        image_file = self.images[index]
        image = Image.open(image_file).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)


def get_dataset(path, img_scale, batch_size):
    transform = transforms.Compose([
        transforms.Resize(img_scale),
        # 也可以使用 Scale
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageDataset(path, transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)
    return data_loader



# 工具函数:

# 生成 噪音 z
def gen_noisy(batch_size, noisy_dim):
    return torch.randn(batch_size, noisy_dim)


# tensor to variable
def to_variable(x):
    x = Variable(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x



# 这里可以定义一个 Config 类，用来保存这些超参数
class Config(object):
    def __init__(self):
        self.batch_size = 128
        self.image_path = './faces/' # 图像数据存放的文件，如有需要请自行调整。
        self.noisy_dim = 100
        self.G_lr = 2*1e-6
        self.D_lr = 2*1e-6
        self.EPOCH = 500
        self.img_scale = 64
        self.k_step = 5
        self.use_gpu = True
        self.g_img_size = 64
        self.d_img_size = 64



# 训练阶段 在正式开始训练前，我们先来看看写完的 Generator 和 Discriminator
config = Config()

G = Generator(noise_dim=config.noisy_dim, g_img_size=config.g_img_size)
if config.use_gpu:
    G = G.cuda()
    
D = Discriminator(d_img_size=config.d_img_size)
if config.use_gpu:
    D = D.cuda()

print("Generator的结构是：")
print(G)
print("Discriminator的结构是：")
print(D)


# 正式开始训练的阶段：

train_data_loader = get_dataset(config.image_path, config.img_scale, config.batch_size)
loss_fn = torch.nn.BCELoss()
g_optimizer = optim.Adam(G.parameters(), lr=config.G_lr)
d_optimizer = optim.Adam(D.parameters(), lr=config.D_lr)

for epoch in range(config.EPOCH):
    g_total_loss = torch.FloatTensor([0])
    d_total_loss = torch.FloatTensor([0])
    count = 0
    
    for i, data in enumerate(train_data_loader):
        count += 1
        true_inputs = data
        images = to_variable(true_inputs)
        batch_size = images.size(0)

        z = to_variable(gen_noisy(batch_size, config.noisy_dim))

        real_labels = to_variable(torch.ones(batch_size))
        fake_labels = to_variable(torch.zeros(batch_size))


        ###          train D           ###
        outputs = D(images)
        d_loss_real = loss_fn(outputs, real_labels)
        real_score = outputs

        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = loss_fn(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        # count total loss
        d_total_loss += d_loss.data[0]
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()


        ###          train G           ###
        z = to_variable(gen_noisy(batch_size, config.noisy_dim))
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = loss_fn(outputs, real_labels)
        g_total_loss += g_loss.data[0]
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 150 == 0:
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, '
                  'g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f'
                  % (epoch+1, config.EPOCH, i + 1, count, d_loss.data[0], g_loss.data[0],
                     real_score.data.mean(), fake_score.data.mean()))
    print('Epoch [%d/%d]'% (epoch, config.EPOCH))
    print('D 的 total loss', d_total_loss / count)
    print('G 的 total loss', g_total_loss / count)
    # Save real images

    fake_images = fake_images.view(fake_images.size(0), 3, 64, 64)
    # 注意 这里需要创建 test_DCGAN 文件夹，否则会报错
    save_image(fake_images.data, './test_DCGAN/fake_images-%d.png' % (epoch + 1))







