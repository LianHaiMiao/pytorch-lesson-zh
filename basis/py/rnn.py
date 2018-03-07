import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np


# 构建 config 类来 控制所有参数
class Config(object):
    def __init__(self):
        self.path = '../dataSet/dinos.txt'
        self.data_size = None
        self.vocab_size = None
        self.char_to_ix = None
        self.ix_to_char = None
        self.output_size = None
        self.epoch = 100
        self.lr = 0.01
        self.hidden_size = 50
        self.batch_size = 1
        self.maxValue = 10 # 防止梯度爆炸所使用
    
config = Config()

# 读取数据
data = open(config.path, 'r').read()
# 字符全部小写
data = data.lower()
# 查看有多少种字符
chars = list(set(data))

# 27个字符，所以输出也是 27
config.data_size, config.vocab_size, config.output_size = len(data), len(chars), len(chars)

print('数据集中一共有 %d 个字符，不同的字符一共有 %d 个。' % (config.data_size, config.vocab_size))

config.char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) } # 字符到索引的映射
config.ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) } # 索引到字符的映射
print(config.ix_to_char)


# one hot encoding
def one_hot(ids, vocab_size):
    """
        ids: list
    """
    ids = torch.LongTensor(ids).view(-1, 1)
    out = torch.zeros(ids.size()[0], vocab_size).scatter_(1, ids, 1)
    return out


# 训练数据集
def train_dataset(path, char_to_ix, vocab_size):
    with open(path, 'r') as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    for index in range(len(examples)):
        X = [char_to_ix[ch] for ch in examples[index]] 
        Y = X[1:] + [char_to_ix["\n"]]
        i_d = one_hot(X, vocab_size)
        input_data = Variable(i_d.view(1, i_d.size()[0], i_d.size()[1])) # batch*seq_len*input_size
        target_data = Variable(torch.LongTensor(Y))
        yield input_data, target_data


# 构造模型
class DinosaurModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, num_layers=1):
        super(DinosaurModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        # 就是一行就构建了 RNN 模型
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # 输出层，用于预测字符
        self.fc  = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0):
        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # hidden: (batch, num_layers * num_directions, hidden_size)
        # out: (batch, seq_len, hidden_size)
        out, hidden = self.rnn(x, h0)

        # 把 rnn 的结果堆成 (batch*seq_len, hidden_size) 大小
        out = out.view(out.size()[0]*out.size()[1], self.hidden_size)
        y = self.fc(out)
        return y
    
    def init_hidden(self):
        # 初始化隐含层
        return Variable(torch.zeros(self.batch_size, self.num_layers, self.hidden_size))

# 构建更新梯度和梯度剪裁的函数¶
def clip_and_update(parameters, lr, maxValue):
    for p in parameters:
        gradients = torch.clamp(p.grad.data, min=-maxValue, max=maxValue)
        p.data.add_(-lr, gradients)
    return


def sample(model, char_to_ix, ix_to_char, vocab_size):
    # 初始值
    random_int = random.randint(0, vocab_size)
    i_d = one_hot([random_int], vocab_size)
    a_input = Variable(i_d).view(1, 1, vocab_size)
    indices = []
    idx = -1
    counter = 0
    eos = char_to_ix['\n']
    
    while (idx != eos and counter != 30):
        h0 = model.init_hidden()
        out = model(a_input, h0)
        # 通过 softmax 求出每个字符的概率
        p = F.softmax(out)
        # 取出概率最大的字符的位置
        val, ids = torch.max(p, 1)
        # 加入预测结果数组里面
        idx = ids.data[0]
        indices.append(idx)
        a_input = Variable(one_hot([idx], vocab_size).view(1, 1, vocab_size))
        counter += 1
    if (counter == 30):
        indices.append(char_to_ix['\n'])
    strl = [ix_to_char[i] for i in indices]
    return "".join(strl)


dinos = DinosaurModel(config.vocab_size, config.hidden_size, config.output_size, config.batch_size)

# 有的朋友可能很好奇，明明就是softmax + Negtive log-likelihood function 为啥要用 CrossEntropyLoss 呢？
# 因为 PyTorch 就是这样设计的啊~
loss_fn = nn.CrossEntropyLoss()

for i in range(config.epoch):    
    # 数据准备
    total_loss = []
    for input_data, target_data in train_dataset(config.path, config.char_to_ix, config.vocab_size):
        dinos.zero_grad()
        # 隐含层初始化
        h0 = dinos.init_hidden()
        # 数据扔进模型里面
        y_pred = dinos(input_data, h0)
        # 求 loss
        loss = loss_fn(y_pred, target_data)
        total_loss.append(loss.data[0])
        # 反向传播
        loss.backward()
        # 梯度剪裁同时更新模型
        clip_and_update(dinos.parameters(), config.lr, config.maxValue)

    if (i+1) % 20 == 0:
        print("loss is %.6f" %(np.mean(total_loss)))
        # 采样
        strl = sample(dinos, config.char_to_ix, config.ix_to_char, config.vocab_size)
        print(strl)