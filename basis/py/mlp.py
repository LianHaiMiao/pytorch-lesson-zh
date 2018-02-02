# 先做一个热身题目，我们使用Tensor构建一个两层神经网络
# Tips:通常构建一个神经网络，我们有如下步骤
# 1、构建好网络模型
# 2、参数初始化
# 3、前向传播
# 4、计算损失
# 5、反向传播求出梯度
# 6、更新权重

# 在我们构建神经网络之前，我们先介绍一个Tensor的内置函数 clamp()
# 该函数的功能是：将输入 Tensor 的每个元素夹紧到区间 [min,max]中，并返回结果到一个新的Tensor。
# 这样，我们就可以使用 x.clamp(min=0) 来代替 relu函数

import torch
# M是样本数量，input_size是输入层大小
# hidden_size是隐含层大小，output_size是输出层大小
M, input_size, hidden_size, output_size = 64, 1000, 100, 10

# 生成随机数当作样本
x = torch.randn(M, input_size) #size(64, 1000)
y = torch.randn(M, output_size) #size(64, 10)

# 参数初始化
def init_parameters():
    w1 = torch.randn(input_size, hidden_size)
    w2 = torch.randn(hidden_size, output_size)
    b1 = torch.randn(1, hidden_size)
    b2 = torch.randn(1, output_size)
    return {"w1": w1, "w2":w2, "b1": b1, "b2": b2}

# 定义模型
def model(x, parameters):
    Z1 = x.mm(parameters["w1"]) + parameters["b1"] # 线性层
    A1 = Z1.clamp(min=0) # relu激活函数
    Z2 = A1.mm(parameters["w2"]) + parameters["b2"] #线性层
    # 为了方便反向求导，我们会把当前求得的结果保存在一个cache中
    cache = {"Z1": Z1, "A1": A1}
    return Z2, cache

# 计算损失
def loss_fn(y_pred, y):
    loss = (y_pred - y).pow(2).sum() # 我们这里直接使用 MSE(均方误差) 作为损失函数
    return loss

# 反向传播，求出梯度
def backpropogation(x, y, y_pred, cache, parameters):
    m = y.size()[0] # m个样本
    # 以下是反向求导的过程：
    d_y_pred = 1/m * (y_pred - y)
    d_w2 = 1/m * cache["A1"].t().mm(d_y_pred)
    d_b2 = 1/m * torch.sum(d_y_pred, 0, keepdim=True)
    d_A1 = d_y_pred.mm(parameters["w2"].t())
    # 对 relu 函数求导: start
    d_Z1 = d_A1.clone()
    d_Z1[cache["Z1"] < 0] = 0
    # 对 relu 函数求导: end
    d_w1 = 1/m * x.t().mm(d_Z1)
    d_b1 = 1/m * torch.sum(d_Z1, 0, keepdim=True)
    grads = {
        "d_w1": d_w1, 
        "d_b1": d_b1, 
        "d_w2": d_w2, 
        "d_b2": d_b2
    }
    return grads

# 更新参数
def update(lr, parameters, grads):
    parameters["w1"] -= lr * grads["d_w1"]
    parameters["w2"] -= lr * grads["d_w2"]
    parameters["b1"] -= lr * grads["d_b1"]
    parameters["b2"] -= lr * grads["d_b2"]
    return parameters

## 设置超参数 ##

learning_rate = 1e-2
EPOCH = 400

# 参数初始化
parameters = init_parameters()

## 开始训练 ##
for t in range(EPOCH):    
    # 向前传播
    y_pred, cache = model(x, parameters)
    
    # 计算损失
    loss = loss_fn(y_pred, y)
    if (t+1) % 50 == 0:
        print(loss)
    # 反向传播
    grads = backpropogation(x, y, y_pred, cache, parameters)
    # 更新参数
    parameters = update(learning_rate, parameters, grads)


# ----------------------------------------------------
# ----------------------------------------------------
# ----------------------------------------------------


# 加上Variable来写多层感知机
# 现在，我们来使用 Variable 重新构建上述的两层神经网络，这个时候，我们已经不需要再使用手动求导了（因为有了自动求导的机制啊~）
# 可以看到，我们的下面的代码已经精简很多了...

import torch
from torch.autograd import Variable # 导入 Variable 对象

# M是样本数量，input_size是输入层大小
# hidden_size是隐含层大小，output_size是输出层大小
M, input_size, hidden_size, output_size = 64, 1000, 100, 10

# 生成随机数当作样本，同时用Variable 来包装这些数据，设置 requires_grad=False 表示在方向传播的时候，
# 我们不需要求这几个 Variable 的导数
x = Variable(torch.randn(M, input_size), requires_grad=False)
y = Variable(torch.randn(M, output_size), requires_grad=False)

# 参数初始化，同时用Variable 来包装这些数据，设置 requires_grad=True 表示在方向传播的时候，
# 我们需要自动求这几个 Variable 的导数
def init_parameters():
    w1 = Variable(torch.randn(input_size, hidden_size), requires_grad=True)
    w2 = Variable(torch.randn(hidden_size, output_size), requires_grad=True)
    b1 = Variable(torch.randn(1, hidden_size), requires_grad=True)
    b2 = Variable(torch.randn(1, output_size), requires_grad=True)
    return {"w1": w1, "w2":w2, "b1": b1, "b2": b2}

# 向前传播
def model(x, parameters):
    Z1 = x.mm(parameters["w1"]) + parameters["b1"] # 线性层
    A1 = Z1.clamp(min=0) # relu激活函数
    Z2 = A1.mm(parameters["w2"]) + parameters["b2"] #线性层
    return Z2

# 计算损失
def loss_fn(y_pred, y):
    loss = (y_pred - y).pow(2).sum() # 我们这里直接使用 MSE(均方误差) 作为损失函数
    return loss

## 设置超参数 ##
learning_rate = 1e-6
EPOCH = 300

# 参数初始化
parameters = init_parameters()

## 开始训练 ##
for t in range(EPOCH):    
    # 向前传播
    y_pred= model(x, parameters)
    # 计算损失
    loss = loss_fn(y_pred, y)
    # 计算和打印都是在 Variables 上进行操作的，这时候的 loss 时一个 Variable ，
    # 它的size() 是 (1,)，所以 loss.data 的size() 也是 (1,)
    # 所以， loss.data[0] 才是一个实值
    if (t+1) % 50 == 0:
        print(loss.data[0])
    # 使用自动求导来计算反向传播过程中的梯度，这个方法会把所有的设置了requires_grad=True 的Variable 对象的梯度全部自动出来
    # 在这里，就是求出了 w1, w2, b1, b2的梯度
    loss.backward()
    
    # 更新参数， .data 表示的都是Tensor
    parameters["w1"].data -= learning_rate * parameters["w1"].grad.data
    parameters["w2"].data -= learning_rate * parameters["w2"].grad.data
    parameters["b1"].data -= learning_rate * parameters["b1"].grad.data
    parameters["b2"].data -= learning_rate * parameters["b2"].grad.data
    
    # 由于PyTorch中的梯度是会累加的，所以如果你没有手动清空梯度，那么下次你家的grad就是这次和上次的grad的累加和。
    # 所以，为了每次都只是使用当前的梯度，我们需要手动清空梯度
    parameters["w1"].grad.data.zero_()
    parameters["w2"].grad.data.zero_()
    parameters["b1"].grad.data.zero_()
    parameters["b2"].grad.data.zero_()

# ----------------------------------------------------
# ----------------------------------------------------
# ----------------------------------------------------

# 使用nn和optim来构建多层感知机
# 我们之前已经学过了，使用 nn 快速搭建一个线性模型。
# 现在就用 nn 来快速的搭建一个多层感知机，同样的optim来为我们提供优化功能

import torch
from torch.autograd import Variable
import torch.nn as nn

# M是样本数量，input_size是输入层大小
# hidden_size是隐含层大小，output_size是输出层大小
M, input_size, hidden_size, output_size = 64, 1000, 100, 10

# 生成随机数当作样本，同时用Variable 来包装这些数据，设置 requires_grad=False 表示在方向传播的时候，
# 我们不需要求这几个 Variable 的导数
x = Variable(torch.randn(M, input_size))
y = Variable(torch.randn(M, output_size))

# 使用 nn 包的 Sequential 来快速构建模型，Sequential可以看成一个组件的容器。
# 它涵盖神经网络中的很多层，并将这些层组合在一起构成一个模型.
# 之后，我们输入的数据会按照这个Sequential的流程进行数据的传输，最后一层就是输出层。
# 默认会帮我们进行参数初始化
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size),
)

# 定义损失函数
loss_fn = nn.MSELoss(size_average=False)

## 设置超参数 ##
learning_rate = 1e-4
EPOCH = 300

# 使用optim包来定义优化算法，可以自动的帮我们对模型的参数进行梯度更新。这里我们使用的是随机梯度下降法。
# 第一个传入的参数是告诉优化器，我们需要进行梯度更新的Variable 是哪些，
# 第二个参数就是学习速率了。
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


## 开始训练 ##
for t in range(EPOCH):    
    
    # 向前传播
    y_pred= model(x)
    
    # 计算损失
    loss = loss_fn(y_pred, y)
    
    # 显示损失
    if (t+1) % 50 == 0:
        print(loss.data[0])
    
    # 在我们进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度。
    optimizer.zero_grad()
    
    # 计算梯度
    loss.backward()
    
    # 更新梯度
    optimizer.step()



# ----------------------------------------------------
# ----------------------------------------------------
# ----------------------------------------------------


# 自己定制一个 nn Modules
# 很多时候，我们要建立的模型会比PyTorch现有的模型更加复杂，所以，在这个时候，
# 我们需要自己定制自己的 nn.Module 这个时候，我们也需要定义 forward 方法，这个方法的输入是Variable，输出也是Variable。
# 
# 定制一个自己的 nn Modules，其实就是在 init初始化函数中，将模型需要用到的层给定义好了。
# 然后重写forward()，在里面把数据在模型中流动的过程给写出来，就完成自己模型的定制了。
# 还是使用上面的多层感知机的例子。


###   这里我们展示两种定义模型的写法: 第一种如下   ###

import torch
from torch.autograd import Variable

# 一定要继承 nn.Module
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        在构造器中，我们会实例化两个线性层，并且注意，下面这句 super(TwoLayerNet, self).__init__()
        千万别忘记了
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        在forward函数中，我们会接受一个Variable，然后我们也会返回一个Varible
        """
        Z1 = self.linear1(x)
        A1 = self.relu1(Z1)
        y_pred = self.linear2(A1)
        return y_pred

    
# M是样本数量，input_size是输入层大小
# hidden_size是隐含层大小，output_size是输出层大小
M, input_size, hidden_size, output_size = 64, 1000, 100, 10

# 生成随机数当作样本，同时用Variable 来包装这些数据，设置 requires_grad=False 表示在方向传播的时候，
# 我们不需要求这几个 Variable 的导数
x = Variable(torch.randn(M, input_size))
y = Variable(torch.randn(M, output_size))


model = TwoLayerNet(input_size, hidden_size, output_size)

# 定义损失函数
loss_fn = nn.MSELoss(size_average=False)

## 设置超参数 ##
learning_rate = 1e-4
EPOCH = 300

# 使用optim包来定义优化算法，可以自动的帮我们对模型的参数进行梯度更新。这里我们使用的是随机梯度下降法。
# 第一个传入的参数是告诉优化器，我们需要进行梯度更新的Variable 是哪些，
# 第二个参数就是学习速率了。
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

## 开始训练 ##
for t in range(EPOCH):    
    
    # 向前传播
    y_pred= model(x)
    
    # 计算损失
    loss = loss_fn(y_pred, y)
    
    # 显示损失
    if (t+1) % 50 == 0:
        print(loss.data[0])
    
    # 在我们进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度。
    optimizer.zero_grad()
    
    # 计算梯度
    loss.backward()
    
    # 更新梯度
    optimizer.step()



# ----------------------------------------------------
# ----------------------------------------------------
# ----------------------------------------------------


###   这里我们展示两种定义模型的写法: 第二种如下 （推荐） ###

import torch
from torch.autograd import Variable

# 一定要继承 nn.Module
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
            我们在构建模型的时候，能够使用nn.Sequential的地方，尽量使用它，因为这样可以让结构更加清晰
        """
        super(TwoLayerNet, self).__init__()
        self.twolayernet = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        """
        在forward函数中，我们会接受一个Variable，然后我们也会返回一个Varible
        """
        y_pred = self.twolayernet(x)
        return y_pred

    
# M是样本数量，input_size是输入层大小
# hidden_size是隐含层大小，output_size是输出层大小
M, input_size, hidden_size, output_size = 64, 1000, 100, 10

# 生成随机数当作样本，同时用Variable 来包装这些数据，设置 requires_grad=False 表示在方向传播的时候，
# 我们不需要求这几个 Variable 的导数
x = Variable(torch.randn(M, input_size))
y = Variable(torch.randn(M, output_size))


model = TwoLayerNet(input_size, hidden_size, output_size)

# 定义损失函数
loss_fn = nn.MSELoss(size_average=False)

## 设置超参数 ##
learning_rate = 1e-4
EPOCH = 300

# 使用optim包来定义优化算法，可以自动的帮我们对模型的参数进行梯度更新。这里我们使用的是随机梯度下降法。
# 第一个传入的参数是告诉优化器，我们需要进行梯度更新的Variable 是哪些，
# 第二个参数就是学习速率了。
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

## 开始训练 ##
for t in range(EPOCH):    
    
    # 向前传播
    y_pred= model(x)
    
    # 计算损失
    loss = loss_fn(y_pred, y)
    
    # 显示损失
    if (t+1) % 50 == 0:
        print(loss.data[0])
    
    # 在我们进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度。
    optimizer.zero_grad()
    
    # 计算梯度
    loss.backward()
    
    # 更新梯度
    optimizer.step()