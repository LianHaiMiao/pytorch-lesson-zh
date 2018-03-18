import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import unicodedata, string, re, random, time, math



class Config():
    def __init__(self):
        self.data_path = "../data/cmn-eng/cmn.txt"
        self.use_gpu = True
        self.hidden_size = 256
        self.encoder_lr = 5*1e-5
        self.decoder_lr = 5*1e-5
        self.train_num = 100000 # 训练数据集的数目
        self.print_epoch = 10000
        self.MAX_Len = 15
config = Config()


SOS_token = 0
EOS_token = 1

class Lang():
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.word2count = {}
        self.n_words = 2  # Count SOS and EOS
    
    def addSentence(self, sentence):
        if self.name == "Chinese":
            for word in sentence:
                self.addWord(word)
        else:
            for word in sentence.split(' '):
                self.addWord(word)
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s



def readLangs(lang1, lang2, pairs_file, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(pairs_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = []
    for l in lines:
        temp = l.split('\t')
        eng_unit = normalizeString(temp[0])
        chinese_unit = temp[1]
        pairs.append([eng_unit, chinese_unit])
    
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
        
    return input_lang, output_lang, pairs



MAX_LENGTH = config.MAX_Len  # 长度大于15的我们统统舍弃

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re ",
    "i", "he", 'you', 'she', 'we',
    'they', 'it'
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1]) < MAX_LENGTH and \
        p[0].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]



def prepareData(lang1, lang2, pairs_file, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, pairs_file, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, "字典的大小为", str(input_lang.n_words))
    print(output_lang.name, "字典的大小为", str(output_lang.n_words))
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('Eng', 'Chinese', config.data_path)
print(random.choice(pairs))


# 到目前为止，我们已经把字典构建好了，接下来就是构建训练集

def indexesFromSentence(lang, sentence):
    if lang.name == "Chinese":
        return [lang.word2index[word] for word in sentence]
    else:
        return [lang.word2index[word] for word in sentence.split(' ')]

def variableFromSentence(lang, sentence, use_gpu):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1)) # seq*1
    if use_gpu:
        return result.cuda()
    else:
        return result

def variablesFromPair(pair, use_gpu):
    input_variable = variableFromSentence(input_lang, pair[0], use_gpu)
    target_variable = variableFromSentence(output_lang, pair[1], use_gpu)
    return (input_variable, target_variable)




# 随机获取2个训练数据集， 这里我们依旧不用进行 batch 处理，下一章节 attention 机制中，我们再进行 batch 处理
example_pairs = [variablesFromPair(random.choice(pairs), config.use_gpu)
                      for i in range(2)]
print(example_pairs)





# 建模

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
    
    def forward(self, x, hidden):
        embedded = self.embedding(x).view(1, x.size()[0], -1)
        output = embedded  # batch*seq*feature
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def initHidden(self, use_gpu):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_gpu:
            return result.cuda()
        else:
            return result


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, x, hidden):
        output = self.embedding(x).view(1, 1, -1)
        output = F.relu(output)  
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, use_gpu):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_gpu:
            return result.cuda()
        else:
            return result


# 实例化模型

encoder = Encoder(input_lang.n_words, config.hidden_size)
encoder = encoder.cuda() if config.use_gpu else encoder

decoder = Decoder(config.hidden_size, input_lang.n_words)
decoder = decoder.cuda() if config.use_gpu else decoder

# 定义优化器

encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.encoder_lr)

decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.decoder_lr)


# 定义损失函数

fn_loss = nn.NLLLoss()

training_pairs = [variablesFromPair(random.choice(pairs), config.use_gpu)
                      for i in range(config.train_num)]


# 开始训练
for iter in range(1, config.train_num+1):
    training_pair = training_pairs[iter - 1]
    input_variable = training_pair[0]  # seq_len * 1
    target_variable = training_pair[1]  # seq_len * 1
    
    loss = 0
    
    # 训练过程
    encoder_hidden = encoder.initHidden(config.use_gpu)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    
    # 传入 encoder
    encoder_output, encoder_hidden = encoder(input_variable, encoder_hidden)
    
    # decoder 起始
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if config.use_gpu else decoder_input
    
    decoder_hidden = encoder_hidden
    
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)          
        targ = target_variable[di]
        loss += fn_loss(decoder_output, targ)
        decoder_input = targ
    
    # 反向求导
    loss.backward()
    # 更新梯度
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    print_loss = loss.data[0] / target_length
    
    if iter % config.print_epoch == 0:
        print("loss is: %.4f" % (print_loss))


def sampling(encoder, decoder):
    # 随机选择一个句子
    pair = random.choice(pairs)
    print('>', pair[0])
    print('=', pair[1])
    # 扔进模型中，进行翻译
    input_variable = variableFromSentence(input_lang, pair[0], config.use_gpu)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden(config.use_gpu)
    encoder_output, encoder_hidden = encoder(input_variable, encoder_hidden)
    
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if config.use_gpu else decoder_input
    decoder_hidden = encoder_hidden
    
    decoded_words = []
    
    for di in range(config.MAX_Len):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
        # 把当前的输出当做输入
        decoder_input = Variable(torch.LongTensor([ni]))
        decoder_input = decoder_input.cuda() if config.use_gpu else decoder_input
        
    # 对 decoded_words 进行连接，输出结果
    output_sentence = ' '.join(decoded_words)
    print('<', output_sentence)
    print('')




for i in range(10):
    sampling(encoder, decoder)