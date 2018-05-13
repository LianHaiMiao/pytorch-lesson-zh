import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from torch.nn.parameter import Parameter
import numpy as np

# 这里不做 mask，同时，我们默认 dk = dv
class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_model):
        '''scaled-dot-product-attention
            parameters: 
                d_model: A scalar. attention size
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
    
    def forward(self, Q, K, V):
        ''' forward step
            parameters: 
                Q (batch*n*dk)
                K (batch*m*dk)
                V (batch*m*dv)
            note: dv == dk
        '''
        qk = torch.bmm(Q, K.transpose(1, 2)) # (batch*n*dk) x (batch*dk*m) -> batch*n*m
        weight = F.softmax(qk / self.temper, dim=1) # batch*n*m -> batch*n*m
        attention_V = torch.matmul(weight, V) # (batch*n*m) x (batch*m*dv) -> batch*n*dv
        return attention_V




# 
# code reference from https://github.com/jadore801120/attention-is-all-you-need-pytorch
# 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from torch.nn.parameter import Parameter
import numpy as np
import math

# 使用残差进行链接, no mask
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_k_hat, d_v_hat, n_head=8, dropout_rate=0, mask=False):
        '''multi-head-attention.
            parameters:
                d_model: A scalar. attention size.
                d_k_hat: A scalar. linear project dimension of k.
                d_v_hat: A scalar. linear project dimension of v.
                num_heads: An int. Number of heads.
                dropout_rate: A floating point number. drop_ou
        '''
        super(MultiHeadAttention, self).__init__()
        
        self.n_head = n_head
        self.d_k_hat = d_k_hat # 通常 d_k_hat = d_model / n_head
        self.d_v_hat = d_v_hat # 通常 d_v_hat = d_model / n_head
        
        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k_hat))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k_hat))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v_hat))
        
        self.attention_net = ScaledDotProductAttention(d_model)
        
        self.linear_proj = torch.nn.Linear(n_head*d_v_hat, d_model)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.mask = mask

    def forward(self, Q, K, V):
        ''' forward step
            parameters: Q (batch*n*d_model), K(batch*m*d_model), V(batch*m*d_model)
        '''
        d_k_hat, d_v_hat = self.d_k_hat, self.d_v_hat
        
        residual = Q # batch_size x len_q x d_model
        
        n_head = self.n_head
        
        batch_size, len_q, d_model = Q.size()
        batch_size, len_k, d_model = K.size()
        batch_size, len_v, d_model = V.size()
        
        # 重复 multi-head 次，方便之后进行线性变换
        q_s = Q.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head*(batch_size*len_q)*d_model
        k_s = K.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head*(mb_size*len_k)*d_model
        v_s = V.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head*(mb_size*len_v)*d_model
        
        # 线性变换
        # bmm: (n_head*(batch_size*len_q)*d_model) x (n_head*d_model*d_k_hat) -> n_head*(batch_size*len_q)*d_k_hat
        # view: n_head*(batch_size*len_q)*d_k_hat -> (n_head*batch_size)*len_q*d_k_hat
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k_hat) 
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k_hat)
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v_hat)
        
        # 扔进 Attention network 中
        outputs = self.attention_net(q_s, k_s, v_s) # (n_head*batch_size)*len_q*d_v_hat
        
        # concatenate 操作，复原到  batch_size x len_q x (n_head*d_v_hat)
        # split: (n_head*batch_size)*len_q*d_v_hat ->  n_head 个 [batch_size*len_q*d_v_hat]
        # cat: n_head 个 [batch_size*len_q*d_v_hat] -> batch_size x len_q x (n_head*d_v_hat)
        outputs = torch.cat(torch.split(outputs, batch_size, dim=0), dim=-1)
        
        # 最后一个 linear layer
        outputs = self.linear_proj(outputs) # batch_size x len_q x (n_head*d_v_hat) -> batch_size x len_q x d_model
        outputs = self.dropout(outputs)
        
        # 残差
        return outputs + residual