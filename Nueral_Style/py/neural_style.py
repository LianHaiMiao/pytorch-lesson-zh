import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision

class Config(object):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]  # IMAGENET中的图片归一化
    IMAGENET_STD = [0.229, 0.224,  0.225]  # IMAGENET中的图片归一化
    imsize = 512  # 统一 content image 和 style image 的 size
    style_image_path = '../images/candy.jpg' # 样式图片的路径
    content_image_path = '../images/hoovertowernight.jpg' # 内容图片的路径
    DOWNLOAD = True  # 是否下载预训练模型
    lr = 0.05  # 学习速率
    epoches = 5000  # 训练epoch
    show_epoch = 5  # 显示损失的epoch
    sample_epoch = 500  # 采样的epoch
    c_weight = 10  # content_loss 的权重
    s_weight = 1500  # style_loss的权重
    use_cuda = torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    
# 参数配置实例化
config = Config()

# 定义预处理函数
# 用于把原始图片进行归一化，转换成卷积神经网络可以接收的输入格式
def preprocess(image_path, trasform=None):
    image = Image.open(image_path)
    image = trasform(image)
    image = image.unsqueeze(0)
    return image.type(config.dtype)

transformer = transforms.Compose([
        transforms.Scale(config.imsize),
        transforms.ToTensor()
        transforms.Normalize(mean = config.IMAGENET_MEAN,std = config.IMAGENET_STD)
    ])

pltshow = transforms.ToPILImage()

# 图片展示
def imshow(tensor, title=None):
    image = tensor.clone().cpu()
    image = image.view(3, config.imsize, config.imsize)
    image = pltshow(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# 定义模型
class VGG_extract(nn.Module):
    def __init__(self):
        super(VGG_extract, self).__init__()
        self.style_layers = [0, 5, 10, 19, 28]
        self.content_layers = [25]
        self.net = models.vgg19(pretrained=config.DOWNLOAD).features

    def forward(self, x):
        # 由于我们只需要指定层的输出，所以，我们同时构建特征提取函数只保留制定层的值
        contents = []
        styles = []
        for i in range(len(self.net)):
            x = self.net[i](x)
            if i in self.style_layers:
                styles.append(x)
            if i in self.content_layers:
                contents.append(x)
        return contents, styles


# 构建损失函数 （最麻烦的地方了...）

# 内容匹配只涉及一层，所以可以看成回归问题，直接使用均方误差。
def content_loss(y_hat, y):
    con_loss = 0
    for y_i, y_j in zip(y_hat, y):
        con_loss += torch.mean((y_i - y_j) ** 2)
    return con_loss


# 样式匹配则是通过拟合Gram矩阵
def gram(y):
    b, c, h, w = y.size()
    y = y.view(c, h * w)
    norm = h * w
    return torch.mm(y, y.t()) / norm

# 计算样式损失
def style_loss(y_hat, y):
    sty_loss = 0
    for y_i, y_j in zip(y_hat, y):
        sty_loss += torch.mean((gram(y_i) - gram(y_j)) ** 2)
    return sty_loss

# 开始训练
style = preprocess(config.style_image_path, transformer)

content = preprocess(config.content_image_path, transformer)

target = Variable(content.clone(), requires_grad=True)  # 这就是我们需要更新的是输入 x

optimizer = torch.optim.Adam([target], lr=config.lr, betas=[0.5, 0.999])

# 加载模型
vgg = VGG_extract()

if config.use_cuda:
    vgg = vgg.cuda()

for epcho in range(config.epoches):
    # 获取特征
    t_c_fea, t_s_fea = vgg(target)
    c_fea, _ = vgg(Variable(content))
    _, s_fea = vgg(Variable(style))

    # 计算损失
    c_loss = content_loss(t_c_fea, c_fea)
    s_loss = style_loss(t_s_fea, s_fea)
    loss = config.c_weight * c_loss + config.s_weight * s_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(epcho)
    if (epcho + 1) % config.show_epoch == 0:
        print('Epcho [%d/%d], Content Loss: %.4f, Style Loss: %.4f'
              % (epcho + 1, config.epoches, c_loss.data[0], s_loss.data[0]))
        # 保存图片
    if (epcho + 1) % config.sample_epoch == 0:
        denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
        img = target.clone().cpu().squeeze()
        img = denorm(img.data).clamp_(0, 1)
        torchvision.utils.save_image(img, 'output-%d.png' % (epcho + 1))

print("Done!")