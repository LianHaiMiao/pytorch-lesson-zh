import torch # 导入torch包
import numpy as np # 导入numpy包

# 创建一个随机的二维数组（矩阵）
exam1 = torch.randn(2, 3)
print(exam1)

# 跟numpy一样，我们可以构造初始化为0、1的数组
exam2 = torch.zeros(2, 3)

exam3 = torch.ones(2, 3)

print(exam2)

print(exam3)


# 当然，我们也可以直接从python的数组直接构造
exam4 = torch.Tensor([[1, 2, 4], [2, 3, 6] ])
print(exam4)


# numpy 通常是通过 .shape 来获取数组的形状，但是对于torch.Tensor，我们使用的是 .size()
# 对得到的变量进行访问，可以采取访问列表的方式

shape = exam4.size()
print(type(shape))
print(shape[0])
print(shape[1])

# 有时候，我们需要对数组形状进行改变，我们可以采用 .view() 的方式
exam5 = exam4.view(3, 2)

# -1表示的是系统自动补齐
exam6 = exam4.view(1, -1)

print(exam5)
print(exam6)

# torch.Tensor 支持大量的数学操作符 + , - , * , / 都是可以用的。
# 当然也可以用Tensor内置的 add() 等, 这里需要提一下的就是 add 和 add_ 的区别
# 使用add函数会生成一个新的Tensor变量， add_ 函数会直接再当前Tensor变量上进行操作
# 所以，对于函数名末尾带有"_" 的函数都是会对Tensor变量本身进行操作的

exam1.add(20)
print(exam1)

exam1.add_(20)
print(exam1)


# 对于常用的矩阵运算Tensor也有很好的支持

exam7 = torch.Tensor([[1, 2, 3], [4, 5, 6]])
exam8 = torch.randn(2, 3)

print("exam7: " , exam7)
print("exam8: " , exam8)


# 矩阵乘法, 其中 t() 表示取转置
torch.mm(exam7, exam8.t())


# 矩阵对应元素相乘
exam7 * exam8

# 跟numpy一样，再Tensor中，也存在Broadcasting
# 当二元操作符左右两边Tensor形状不一样的时候，
# 系统会尝试将其复制到一个共同的形状。例如a的第0维是3, b的第0维是1，那么 a + b这个操作会将b沿着第0维复制3遍。


a = torch.arange(0, 3).view(3, 1)
b = torch.arange(0, 2).view(1, 2)
print("a:", a)
print("b:", b)
print("a+b:", a + b)


# Tensor和Numpy的相互转换


x = np.ones((2, 3))
y = torch.from_numpy(x) # 从numpy -> torch.Tensor
print(y)
z = y.numpy() # 从torch.Tensor -> numpy
print(z)

# 常用操作
# unsqueeze() 可以让我们把一个向量变成矩阵，非常有用~

x_u = torch.Tensor([1, 2, 3, 4])
print(x_u)
x_u_1 = torch.unsqueeze(x_u, 0)
print(x_u_1)
x_u_2 = torch.unsqueeze(x_u, 1)
print(x_u_2)


# 最后，这里只是一个引子，还有很多很多的操作，可以在pyTorch官方文档上查阅 torch.Tensor API
