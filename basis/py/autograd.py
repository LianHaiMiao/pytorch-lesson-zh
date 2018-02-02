# 把必要的包给导入
import torch
from torch.autograd import Variable

# 如果我们想计算  f=3×x3+4×x2+6f=3×x3+4×x2+6  的导数，该如何做呢？

def fn(x):
    y = 3 * x.pow(3) + 4 * x.pow(2) + 6
    return y

x1 = Variable(torch.Tensor([1]), requires_grad=True)

y1 = fn(x1)

print(y1)

y1.backward() # 自动求导

x1.grad # 查看梯度


# 通过调用 backward() 函数，我们自动求出了在 x = 1 的时候的导数
# 需要注意的一点是：如果我们输入的 Tensor 不是一个标量，而是矢量（多个值）。
# 那么，我们在调用backward()之前，需要让结果变成标量 才能求出导数。
# 也就是说如果不将 Y 的值变成标量，就会报错。（可以尝试把mean()给取消，看看是不是报错了）

x2 = Variable(torch.Tensor([[1, 2], [3, 4]]), requires_grad=True)

y2 = fn(x2).mean() # 将结果变成标量，这样就不会报错了

y2.backward() # 自动求导

x2.grad # 查看梯度