import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# 各位读者大大请注意，这段代码的运行环境是 pytorch 1.0；
# 0.3.0 及以下的版本会需要加上 Variable 变量。

# 数据准备工作，我们使用 karate 俱乐部数据集
# 前期的准备工作，其实就两个：  1. 构建邻接矩阵；  2. 求出度数矩阵
# 对于大规模数据集，我们可以使用 稀疏矩阵的形式去完成。
def construct_graph_data(path, num):
    # 由于 GCN 中真正使用到的数据就是 X \in R^{N x C} 和 A \in R^{N x N} 和 D \in R^{N x C}
    # 所以，给定图之间节点的连接信息，我们输出 A 和 D
    # 我们需要提前知道图中的节点数目
    N = num
    A = np.zeros((N, N), dtype=int)
    with open(path, "r") as fr:
        # 第一行不要
        line = fr.readline()
        line = fr.readline()
        while line != None and line != "":
            arr = line.strip().split(" ")
            sou, tar = int(arr[0]), int(arr[1])
            A[sou-1, tar-1] = 1
            line = fr.readline()
    # 得到了邻接矩阵 A，根据邻接矩阵去求度数矩阵 D，我们先让邻接矩阵加上一个单位阵
    I_N = np.eye(N, dtype=int)
    A_tilde = A + I_N
    # 接着根据 A_tilde 去求 D_tilde
    each_node_degree = np.sum(A_tilde, axis=0)
    D = np.power(each_node_degree, -0.5)
    D_05 = np.power(D, -0.5)
    D_tilde = np.diag(D_05)
    return A_tilde, D_tilde

def get_train_test(path, train_num):
    all_data = []
    with open(path, "r") as fr:
        # 第一行不要
        line = fr.readline()
        line = fr.readline()
        while line != None and line != "":
            arr = line.strip().split(" ")
            all_data.append((int(arr[0]), int(arr[1])))
            line = fr.readline()
    data = [temp[0] for temp in all_data]
    label = [temp[1] for temp in all_data]
    return data, label

# 开始准备构建模型
class GCN(nn.Module):
    def __init__(self, A_tilde, D_tilde, input_dim, output_class):
        super(GCN, self).__init__()
        self.A_hat = D_tilde.mm(A_tilde).mm(D_tilde)  # N*N
        self.fc1 = nn.Linear(input_dim, input_dim // 2, bias=False)   # 这里遵循原文的写法，不加 bias
        self.fc2 = nn.Linear(input_dim // 2, input_dim // 4, bias=False)  # 感觉加不加结果没啥改变...当然有可能是因为数据集太小的原因
        # 输出结果的分类
        self.output_layer = nn.Linear(input_dim // 4, output_class, bias=False)

    def forward(self, x):
        # 两层 GCN
        h1 = F.relu(self.fc1(self.A_hat.mm(x)))
        h2 = F.relu(self.fc2(self.A_hat.mm(h1)))
        res = self.output_layer(self.A_hat.mm(h2))
        return res


if __name__ == '__main__':
    # 构造图结构
    graph_path = "../data/karate.txt"
    node_num = 34
    input_dim = node_num
    output_class = 2
    A_tilde, D_tilde = construct_graph_data(graph_path, node_num)
    # 从 numpy 格式转换成 tensor 格式
    A = torch.tensor(A_tilde, dtype=torch.float32)
    D = torch.tensor(D_tilde, dtype=torch.float32)
    # 我们将 label.txt 数据处理一下，得到数据和标签。
    label_path = "../data/label.txt"
    all_data, all_label = get_train_test(label_path, train_num=24)
    # 简单点，直接使用 one-hot encoding 来表示特征
    X_features = torch.eye(node_num, node_num)  # N*N
    # 用于比较的数据
    torch_all_label = torch.tensor(all_label, dtype=torch.int64)
    # 构建模型
    gcn = GCN(A, D, input_dim, output_class)
    # 确定优化器
    optimizer = torch.optim.Adam(gcn.parameters())
    # 确定损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 开始训练
    for epoch in range(200):
        y_pred = F.softmax(gcn(X_features), dim=1)
        loss = loss_fn(y_pred, torch_all_label)
        # 计算准确率
        _, pred = torch.max(y_pred, 1)
        num_correct = (pred == torch_all_label).sum()
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 每隔 10 次， output 一次
        if epoch % 10 == 0:
            print('Accuracy is {:.4f}'.format(num_correct.item() / node_num))

    print("Done!")

