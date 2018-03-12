import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np



class Corpus(object):
    """
        构建语料库的类
        path: 文件路径
    """
    def __init__(self, path):
        self.path = path
        self.char2id = {}
        self.id2char = {}
        self.corpus_indices = None
    def get_data(self):
        with open(self.path, 'r', encoding='utf8') as f:
            chars = f.read()
        chars_list = chars.replace('\n', ' ').replace('\r', ' ')
        # 开始创建索引 word 2 id
        idx = 0
        for char in chars_list:
            if not char in self.char2id:
                self.char2id[char] = idx
                self.id2char[idx] = char
                idx += 1
        # 将 corpus 里面的 char 用 index表示
        self.corpus_indices = [self.char2id[char] for char in chars_list]
    
    # 获取 corpus 的长度
    def __len__(self):
        return len(self.char2id)


# 构建 Config 类，用于控制超参数
class Config(object):
    def __init__(self):
        self.embed_size = 128 # embedding size
        self.hidden_size = 1024 # RNN中隐含层的 size
        self.num_layers = 1 # RNN 中的隐含层有几层，我们默认设置为 1层
        self.epoch_num = 50 # 训练迭代次数
        self.sample_num = 10 # 随机采样
        self.batch_size = 32 # batch size
        self.seq_length = 35 # seq length
        self.lr = 0.002 #learning rate
        self.path = "./LSTM/jaychou_lyrics.txt" # 歌词数据集
        self.prefix = ['分开', '战争中', '我想'] # 测试阶段，给定的前缀，我们用它来生成歌词
        self.pred_len = 50 # 预测的字符长度
        self.use_gpu = True
        
config = Config()

# 这里简单一些，我们直接用一个函数来作为迭代器生成训练样本
def getBatch(corpus_indices, batch_size, seq_length, config):
    data_len = len(corpus_indices)
    batch_len = data_len // config.batch_size
    corpus_indices = torch.LongTensor(corpus_indices)
    # 将训练数据的 size 变成 batch_size x seq_length
    indices = corpus_indices[0: batch_size * batch_len].view(batch_size, batch_len)
    for i in range(0, indices.size(1) - seq_length, seq_length):
        input_data = Variable(indices[:, i: i + seq_length])
        target_data = Variable(indices[:, (i + 1): (i + 1) + seq_length].contiguous())
        # use GPU to train the model
        if config.use_gpu:
            input_data = input_data.cuda()
            target_data = target_data.cuda()
        yield(input_data, target_data)


# 将当前的状态从计算图中分离，加快训练速度
def detach(states):
    return [state.detach() for state in states] 


# 定义 LSTM 模型
class lstm(nn.Module):
    # input: 
    # x: 尺寸为 batch_size * seq_length 矩阵。
    # hidden: 尺寸为 batch_size * hidden_dim 矩阵。
    # output:
    # out: 尺寸为 batch_size * vocab_size 矩阵。
    # h: 尺寸为 batch_size * hidden_dim 矩阵。
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(lstm, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        
    def forward(self, x, hidden):
        embeds = self.embed(x)
        
        out, hidden = self.rnn(embeds, hidden)

        out = out.contiguous().view(out.size(0)*out.size(1), -1) # out 的 size 变成 (batch_size*sequence_length, hidden_size)
        
        out = self.linear(out) # (batch_size*sequence_length, hidden_size) -> (batch_size*sequence_length, vocab_size)
        return out, hidden
    
    def init_weights(self):
        self.embed.weight = nn.init.xavier_uniform(self.embed.weight)
        self.linear.bias.data.fill_(0)
        self.linear.weight = nn.init.xavier_uniform(self.linear.weight)


# 构建语料库
corpus = Corpus(config.path)
# 处理 data
corpus.get_data()
# 模型初始化
lstm = lstm(len(corpus), config.embed_size, config.hidden_size, config.num_layers)
# 使用 gpu
if config.use_gpu:
    lstm = lstm.cuda()


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=config.lr)



# 使用训练好的 LSTM 在给定前缀的前提下，自动生成歌词。
def predict(model, prefix, config, corpus):
    """
        model 是模型， prefix是前缀， config 是参数类， corpus是语料库类
    """
    state_h = Variable(torch.zeros(config.num_layers, 1, config.hidden_size)) # 起始的hidden status
    state_c = Variable(torch.zeros(config.num_layers, 1, config.hidden_size)) # 起始的cell status
    
    # use gpu
    if config.use_gpu:
        state_h = state_h.cuda()
        state_c = state_c.cuda()
    # become a tuple
    state = (state_h, state_c)
    output = [corpus.char2id[prefix[0]]]
    for i in range(config.pred_len + len(prefix)):
        X = Variable(torch.LongTensor(output)).unsqueeze(0)
        # use gpu
        if config.use_gpu:
            X = X.cuda()
        Y, state = model(X, state)
        # 我们将结果变成概率，选择其中概率最大的作为预测下一个字符
        prob = Y.data[0].exp()
        word_id = torch.multinomial(prob, 1)[0]
        if i < len(prefix) - 1:
            next_char = corpus.char2id[prefix[i+1]]
        else:
            next_char = int(word_id)
        output.append(next_char)
    print("".join([corpus.id2char[id] for id in output]))
    return 


# 开始训练
for epoch in range(config.epoch_num):
    # 由于使用的是 lstm，我们初始化 hidden status 和 cell
    state_h = Variable(torch.zeros(config.num_layers, config.batch_size, config.hidden_size)) # 起始的hidden status
    state_c = Variable(torch.zeros(config.num_layers, config.batch_size, config.hidden_size)) # 起始的cell status
    # use gpu
    if config.use_gpu:
        state_h = state_h.cuda()
        state_c = state_c.cuda()
    
    hidden = (state_h, state_c)
    
    train_loss = [] # 训练的总误差
    
    for i,batch in enumerate(getBatch(corpus.corpus_indices, config.batch_size, config.seq_length, config)):
        inputs, targets = batch
        # Forward + Backward + Optimize
        lstm.zero_grad()
        hidden = detach(hidden)
        
        outputs, hidden = lstm(inputs, hidden)

        loss = criterion(outputs, targets.view(-1))
        train_loss.append(loss.data[0])
        loss.backward()
        torch.nn.utils.clip_grad_norm(lstm.parameters(), 0.5) # 梯度剪裁
        optimizer.step()
    # 采样，进行预测
    if epoch % config.sample_num == 0:
        print("Epoch %d. Perplexity %f" % (epoch, np.exp(np.mean(train_loss))))
        # 对给定的歌词开头，我们自动生成歌词
        for preseq in config.prefix:
            predict(lstm, preseq, config, corpus)

