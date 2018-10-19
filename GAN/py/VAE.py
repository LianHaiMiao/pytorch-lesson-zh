import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils import data
import os
from PIL import Image
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision


# 数据预处理阶段
class ImageDataset(data.Dataset):
    def __init__(self, path, transform=None):
        """
            path 是保存图像的文件夹。
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

# 处理数据的函数
def get_dataset(path, img_scale, batch_size):
    # 注意，这里我们把所有的图片中央裁剪到 64x64 的大小
    transform = transforms.Compose([
        transforms.CenterCrop((img_scale, img_scale)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    datasets = ImageDataset(path, transform)
    data_loader = data.DataLoader(dataset=datasets,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)
    return data_loader



# 这里可以定义一个 Config 类，用来保存这些超参数
class Config(object):
    def __init__(self):
        self.batch_size = 128
        self.image_path = './celebA/'  # 图像数据存放的文件，如有需要请自行调整。
        self.z_dim = 512  # 隐含变量的大小
        self.lr = 2*1e-6
        self.EPOCH = 1000
        self.img_scale = 64
        self.use_gpu = torch.cuda.is_available()

# VAE Model
class VAE(nn.Module):
    def __init__(self, input_chanel=3, z_dim=512, img_size=64):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        # encoder part
        self.encoder_conv1 = nn.Conv2d(input_chanel, z_dim//16, kernel_size=4, stride=2, padding=1)
        self.encoder_bn1 = nn.BatchNorm2d(z_dim//16)
        self.encoder_conv2 = nn.Conv2d(z_dim//16, z_dim//8, kernel_size=4, stride=2, padding=1)
        self.encoder_bn2 = nn.BatchNorm2d(z_dim//8)
        self.encoder_conv3 = nn.Conv2d(z_dim//8, z_dim//4, kernel_size=4, stride=2, padding=1)
        self.encoder_bn3 = nn.BatchNorm2d(z_dim//4)
        self.encoder_conv4 = nn.Conv2d(z_dim//4, z_dim//2, kernel_size=4, stride=2, padding=1)
        self.encoder_bn4 = nn.BatchNorm2d(z_dim//2)
        self.encoder_conv5 = nn.Conv2d(z_dim//2, z_dim, kernel_size=4, stride=2, padding=1)
        self.encoder_bn5 = nn.BatchNorm2d(z_dim)
        self.encoder_avg_pooling = nn.AvgPool2d(kernel_size=(2, 2))
        self.encoder_means = nn.Linear(z_dim, z_dim)
        self.encoder_log_var = nn.Linear(z_dim, z_dim)
        #decoder part
        self.decoder_conv1 = nn.ConvTranspose2d(z_dim, z_dim//2, kernel_size=4, stride=1, padding=0)
        self.decoder_bn1 = nn.BatchNorm2d(z_dim//2)
        self.decoder_conv2 = nn.ConvTranspose2d(z_dim//2, z_dim//4, kernel_size=4, stride=2, padding=1)
        self.decoder_bn2 = nn.BatchNorm2d(z_dim//4)
        self.decoder_conv3 = nn.ConvTranspose2d(z_dim//4, z_dim//8, kernel_size=4, stride=2, padding=1)
        self.decoder_bn3 = nn.BatchNorm2d(z_dim//8)
        self.decoder_conv4 = nn.ConvTranspose2d(z_dim//8, z_dim//16, kernel_size=4, stride=2, padding=1)
        self.decoder_bn4 = nn.BatchNorm2d(z_dim//16)
        self.decoder_conv5 = nn.ConvTranspose2d(z_dim//16, 3, kernel_size=4, stride=2, padding=1)
        self.decoder_linear = nn.Linear(z_dim, z_dim)

    def encode(self, x):
        x1 = F.leaky_relu(self.encoder_bn1(self.encoder_conv1(x)), negative_slope=0.2)
        x2 = F.leaky_relu(self.encoder_bn2(self.encoder_conv2(x1)), negative_slope=0.2)
        x3 = F.leaky_relu(self.encoder_bn3(self.encoder_conv3(x2)), negative_slope=0.2)
        x4 = F.leaky_relu(self.encoder_bn4(self.encoder_conv4(x3)), negative_slope=0.2)
        x5 = F.leaky_relu(self.encoder_bn5(self.encoder_conv5(x4)), negative_slope=0.2)
        x6 = self.encoder_avg_pooling(x5)  # batch*z_dim*1*1
        encoder_means = self.encoder_means(x6.view(-1, self.z_dim))
        encoder_log_var = self.encoder_log_var(x6.view(-1, self.z_dim))
        return encoder_means, encoder_log_var


    def decode(self, z):
        # z: batch * z_dim -> batch * z_dim * 1 * 1
        z_f = self.decoder_linear(z)
        z0 = z_f.view(-1, self.z_dim, 1, 1)
        z1 = F.relu(self.decoder_bn1(self.decoder_conv1(z0)))
        z2 = F.relu(self.decoder_bn2(self.decoder_conv2(z1)))
        z3 = F.relu(self.decoder_bn3(self.decoder_conv3(z2)))
        z4 = F.relu(self.decoder_bn4(self.decoder_conv4(z3)))
        z5 = torch.tanh(self.decoder_conv5(z4))
        return z5

    # 重参数技巧：本质上就是避免直接从 N(mu, sigma) 中采样； 而是从 N(0, 1) 中采样 得到 epsilon
    # 然后令 epsilon * variance + means 也就相当于从 N(mu, sigma) 中采样
    def reparameterize(self, z_means, z_log_var):
        var = torch.exp(z_log_var/2)  # log(sigma^2) -> sigma
        epsilon = torch.randn_like(z_means)  # 从标准正态分布中采样 epsilon
        return z_means + var * epsilon


    def forward(self, x):
        # 通过 encoder 求出均值和 log(方差)
        z_means, z_log_var = self.encode(x)  # batch*z_dim
        z = self.reparameterize(z_means, z_log_var)
        # 将采样出来的结果进行重构
        x_reconst = self.decode(z)
        return x_reconst, z_means, z_log_var



config = Config()

train_data_loader = get_dataset(config.image_path, config.img_scale, config.batch_size)

# 在当前目录下创建一个 samples 的文件夹，用来保存采样的结果。
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


model = VAE()
if config.use_gpu:
    model = model.cuda()

# Loss and Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

# Start training
for epoch in range(config.EPOCH):
    for i, data in enumerate(train_data_loader):
        # Forward pass
        x = data
        # x_reconst, mu, log_var = model(x)
        x_reconst, z_means, z_log_var = model(x)

        # reconstruction loss
        reconst_loss = F.mse_loss(x_reconst, x, reduction="sum")

        # kl divergence
        kl_div = - 0.5 * torch.sum(1 + z_log_var - z_means.pow(2) - z_log_var.exp())

        # Backprop and optimize
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                  .format(epoch + 1, config.EPOCH
                          , i + 1, len(train_data_loader), reconst_loss.item(), kl_div.item()))

    
    # 随机采样保存图片
    with torch.no_grad():
        # 保存采样的图片
        z = torch.randn(config.batch_size, config.z_dim)
        out = model.decode(z).view(-1, 3, 64, 64)
        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)))

        # 保存重构的图片和原始图片
        out, _, _ = model(x)
        x_concat = torch.cat([x.view(-1, 3, 64, 64), out.view(-1, 3, 64, 64)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))

