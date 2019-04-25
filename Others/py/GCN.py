import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 数据准备工作，我们使用 karate 俱乐部数据集
# 前期的准备工作，其实就两个：  1. 构建邻接矩阵；  2. 求出度数矩阵
# 对于大规模数据集，我们可以使用 稀疏矩阵的形式去完成。
def read_data(path, num):
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
    # 接着根据 A_tilde 去求 D
    each_node_degree = np.sum(A_tilde, axis=0)
    D = np.diag(each_node_degree)
    return A_tilde, D






path = "./data/karate.txt"
read_data(path, 34)
