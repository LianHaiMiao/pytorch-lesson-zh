import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, in_dim, n_class):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)
        
        # 参数初始化函数
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

# 实例化模型
lenet = LeNet5(224, 10)

# 让我们假设，经过了一连串的训练
# 这时候的模型已经被我们训练的
# 十分完美了。

# 保存路径
PATH = "./test.pkl"

# 第一种：保存模型的参数和结构

# 保存
torch.save(lenet, PATH)

# 加载
model = torch.load(PATH)


# 第二种：仅保存模型的参数

torch.save(lenet.state_dict(), PATH)

model2 = LeNet5(224, 10) # 实例化模型
model2.load_state_dict(torch.load(PATH)) # 加载模型参数