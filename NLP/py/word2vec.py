import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# 构建 skip-grams 模型

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        """
            vocab_size: 语料库中的单词的数量
            emb_dim: 向量的维度
        """
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.embedding_v = nn.Embedding(vocab_size, emb_dim) # center word embedding
        self.embedding_u = nn.Embedding(vocab_size, emb_dim) # out word embedding
        self.init_emb()
    
    def forward(center_words, target_words, outer_words):
        center_embeds = self.embedding_v(center_words) # B x 1 x D
        target_embeds = self.embedding_u(target_words) # B x 1 x D
        outer_embeds = self.embedding_u(outer_words) # B x V x D
        
        scores = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # Bx1xD * BxDx1 => Bx1 
        norm_scores = outer_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # BxVxD * BxDx1 => BxV
        
        P_oc = torch.exp(scores) / torch.sum(torch.exp(norm_scores), 1).unsqueeze(1) # Bx1 * Bx1 => B*1 表示 B个单词的 P(o|c)
        nll = -torch.mean(torch.log(P_oc)) # 求 这 B 个单词的 negative log likelihood

        return nll 
    
    def init_emb(self):
        """
            初始化网络权重
        """
        initrange = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)
        return 
    
    def prediction(self, inputs):
        """
            给出 word 的 vector, 这里，我们没有用到后面的 embedding_u，其实这里还有几种做法：
            1、embedding_v + embedding_u； 
            2、embedding_v[:, N/2; :] + embedding_u[:; :, N/2] 也就是 embedding_v 取前一半， embedding_u取后一半，两者拼接。
        """
        embeds = self.embedding_v(inputs)
        return embeds 



# skip-grams模型 + Negative sampling

class SkipgramNegSampling(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(SkipgramNegSampling, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.embedding_v = nn.Embedding(vocab_size, emb_dim) # center word embedding
        self.embedding_u = nn.Embedding(vocab_size, emb_dim) # out word embedding
        self.logsigmoid = nn.LogSigmoid()
        self.init_emb()
        
    def forward(self, center_words, target_words, negative_words):
        center_embeds = self.embedding_v(center_words) # B x 1 x D
        target_embeds = self.embedding_u(target_words) # B x 1 x D
        neg_embeds = -self.embedding_u(negative_words) # K个负样本 B x K x D  
        
        positive_score = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # Bx1xD * BxDx1 -> Bx1
        
        negative_score = neg_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # BxKxD * BxDx1 -> BxK
        
        negative_score = torch.sum(negative_score, 1).view(negs.size(0), -1) # BxK -> Bx1
        
        loss = self.logsigmoid(positive_score) + self.logsigmoid(negative_score)
        
        return -torch.mean(loss) # 原目标函数是最大化的，加一个负号，变成 min 然后就可以使用SGD等优化算法了。
    
    
    def prediction(self, inputs):
        embeds = self.embedding_v(inputs)
        return embeds

    
    def init_emb(self):
        initrange = (2.0 / (self.vocab_size + self.emb_dim))**0.5 # Xavier init
        self.embedding_v.weight.data.uniform_(-initrange, initrange) # init
        self.embedding_u.weight.data.uniform_(-0.0, 0.0) # init
        return 