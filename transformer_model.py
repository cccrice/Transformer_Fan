import math

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 此处试验数据集是两对德语----->英语的句子
# 此处将每个词的索引手动编码，主要是为了降低代码的阅读难度

# S：Decoder输入的起始标志符
# E：Decoder输出的结束标志符
# P：padding，没有意义的符号
sentences = [['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
             ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']]
# 德语和英语要分别建立词库
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
src_idx2word = {i:w for i,w in enumerate(src_vocab)}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
tgt_idx2word = {i:w for i,w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 5
tgt_len = 6

'''Transformer模型中的一些参数'''
d_model = 512 # Embedding Size（token embedding和position编码的维度）
d_ff = 2048 # FeedForward dimension (两次线性层中的隐藏层 512->2048->512，线性层是用来做特征提取的），当然最后会再接一个projection层
d_k, d_v = 64, 64 # dimension of K(=Q), V（Q和K的维度需要相同，这里为了方便让K=V）
n_layers = 6 # Encoder和Decoder Block的个数
n_heads = 8 # 多头注意力结构有几个头

'''构建数据: 把单词序列转换为数字序列'''
def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    # 遍历输入, sentences[i][0]是输入的第i个句子的Encoder input, sentences[i][1]是Decoder input
    # sentences[i][2]是Decoder output(Ground Truth)
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]  # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]] # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]] # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]
        # python中List.append是添加一个对象，List.extend是添加一个列表
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

# 获取编码后的输入
enc_inputs, dec_inputs, dec_outputs = make_data(sentences)

'''自定义Dataset'''
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
    def __getitem__(self, index):
        return self.enc_inputs[index], self.dec_inputs[index], self.dec_outputs[index]
    def __len__(self):
        return self.enc_inputs.shape[0]

# 实例化dataloader对象
loader = Data.DataLoader(MyDataSet(enc_inputs,dec_inputs,dec_outputs), batch_size=2, shuffle=True)


'''位置编码'''
'''
PE(pos, 2i) = sin(pos/(10000^(2i/d_model)))
PE(pos, 2i + 1) = cos(pos/(10000^(2i/d_model)))
为什么Embedding要用dropout？？？
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 初始化positional encoding矩阵 ------> [5000, 512]
        pe = torch.zeros(max_len, d_model)
        # 位置编码公式中的i ------> [5000, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 这一步求出 1/(10000^(2i/d_model)) ------> [256]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # 奇数位执行cos函数, 偶数位执行sin函数
        # 这里的position * div_term利用了广播机制：[5000, 1] * [256] -------> [5000, 256]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [5000, 512] --------> [5000, 1, 512], 扩充的这一维度是给batch_size的
        pe = pe.unsqueeze(0).transpose(0,1)
        # 使用了self.register_buffer('name', Tensor)
        # 此方法的作用是使这组参数在模型训练时不会更新(即调用optimizer.step()后该组参数不会变化，只可人为地改变他们的值)
        # 但是保存模型时，该组参数又作为模型参数的一部分被保存
        self.register_buffer('pe', pe)

    def forward(self,x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        # 因为pe的第一维有5000维，但是x的第一维一般没有那么长，所以仅取pe的前x.size(0)个位置编码进行对位相加
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

''' 
在注意力部分，计算出QK^T/根号dk之后，会得到一个shape为[len_input, len_input]的矩阵，代表每个单词对其余(包含自己)单词的影响力
所以这里需要一个同等大小的矩阵，告诉我哪个位置是PAD部分，在计算softmax之前把这些位置的值设为无穷小
需要注意的是这里得到的矩阵形状是[batch_size, len_q, len_k]，此处是对seq_k做标识
seq_q 和 seq_k 在自注意力机制里是一致的，但是在交互注意力部分，q来自Decoder，k来自Encoder
所以告诉Encoder这边pad符号信息就可以，Decoder的pad信息在交互注意力层是没有用到的
'''
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # pad_attn_mask的shape为[batch_size, 1, len_k]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # 此时返回的值的shape是[batch_szie, len_q, len_k],
    # 这里返回值和pad_attn_mask不共享内存, 也就是tensor.expand()会创建一个新的对象而不会改变tensor本身
    return pad_attn_mask.expand(batch_size, len_q, len_k)

'''用于Decoder的自注意力层, 用一个mask来防止当前输入看到未来的信息'''
def get_attn_subsequence_mask(seq):
    '''
    seq: Decoder的输入
         [batch_size, tgt_len]
    '''
    # attn_shape: [batch_size, tgt_len, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # mask显示了每个tgt词(行)被允许看的位置(列)
    # np.triu(k=1)会将矩阵的左下部分(包含对角线)置为0，这一部分是允许被看到的部分，包含了当前输入词之前的词和他本身
    # 右上部分(对角线以右)保持为1，这一部分是不允许被看到的未来的词
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    # numpy.ndarray ----> torch.ByteTensor
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    # subsequence_mask: [batch_size, tgt_len, tgt_len]
    return subsequence_mask


'''论文中的Feed Forward层，包含了残差连接'''
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias = False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        # inputs的shape为(batch_size, seq_len, d_model)
        residual = inputs
        output = self.fc(inputs)
        # 输出的shape为(batch_size, seq_len, d_model)
        return nn.LayerNorm(d_model).to(device)(output + residual)

'''用于计算softmax(QK^T/根号dk)*V'''
class ScaleDotProdectAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProdectAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # 输入进来的数据shape：
        # Q:(batch_size, n_heads, len_q, d_model), K:(batch_size, n_heads, len_k, d_model), V:(batch_size, n_heads, len_k, d_model)
        # 先计算QK^T/根号dk, 得到的shape为(batch_szie, n_heads, len_q, len_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        # 通过attn_mask将scores中是pad符号的地方设置为无限小，softmax之后基本就是0，对q的单词不起作用
        # 这里使用了tensor.masked_fill_方法
        # attn_mask的shape应该和scores的shape一致，其类型为torch.BoolTensor，pad的地方值为1，不是pad的地方值为0
        scores.masked_fill_(attn_mask, -1e9)
        # 对最后一维进行softmax运算，每一行就是一个q与其他所有k计算得到的权重
        attn = nn.Softmax(dim=-1)(scores)
        # 将权重与V进行相乘得到自注意力层的输出
        context = torch.matmul(attn, V)
        return context, attn

'''多头注意力结构'''
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # 输入进来的QKV是完全相等的，使用nn.Linear将QKV映射得到参数矩阵Wq, Wk, Wv
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    # 这里的QKV就是把Encoder inputs(也就是每个输入值embedding之后的向量)复制三次
    # 为什么不直接把Encoder inputs复制三次作为一个矩阵输入呢？是因为在交互注意力层Q来自Decoder，K,V来自Encoder，就不是相等的了，所以要分开输入
    def forward(self, Q, K, V, attn_mask):
        # 首先映射分头，然后计算atten_scores，然后计算atten_value
        # 输入进来的数据shape：Q:(batch_size, len_q, d_model), K:(batch_size, len_k, d_model), V:(batch_size, len_k, d_model)
        residual, batch_size = Q, Q.size(0)

        '''将QKV矩阵变成多头的'''
        # 映射：(batch_size, len_q, d_model)  ------>  (batch_size, len_k, d_k * n_heads)
        # 分头：(batch_size, len_k, d_k * n_heads)  ------>  (batch_size, len_k, n_heads, d_k)
        # 转置: (batch_size, len_k, n_heads, d_k)  ------>  (batch_size, n_heads, len_k, d_k)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)

        '''这一步就是把pad信息重复到了n头上'''
        # 输入的attn_mask的shape是(batch_size, len_q, len_k),经过以下步骤得到新的attn_mask: (batch_szie, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        '''进行自注意力层的计算'''
        # 得到计算结果: context是公式softmax(QK^T/根号dk)V最后的输出，attn是softmax得到的权重(也就是乘V之前的结果)
        # 这里得到的context的shape是(batch_szie, n_heads, len_q, d_v), attn的shape是(batch_szie, n_heads, len_q, len_k)
        context, attn = ScaleDotProdectAttention()(q_s, k_s, v_s, attn_mask)

        '''这一步将多头结果融合得到最终结果'''
        # tensor.contiguous是用于将tensor变成在内存中连续分布的形式，因为有些tensor并不是在内存中连续分布的，而view操作需要tensor的内存是整块的
        # context的shape变为(batch_szie, len_q, n_heads * d_v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        # 接一个线性层，将其最后一个维度变回d_model
        output = self.linear(context)
        # 残差连接之后接一个LN
        # 输出结果的shape为(batch_szie, len_q, d_model)
        return self.layer_norm(output + residual), attn

'''Encoder一个Block的结构，包含了多头注意力结构和Feed Forward层'''
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # 进行多头自注意力层的计算，最初始的QKV矩阵是和输入相等的
        # enc_inputs的shape为(batch_size, seq_len_q, d_model)
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # 进行Feed Forward层的计算
        # enc_outputs的shape是(batch_size, len_q, d_model)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

'''Encoder完整结构'''
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 定义一个生成矩阵的方法，矩阵的shape是[src_vocab_size, d_model]
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # 实例化PositionalEncoding类
        self.pos_emb = PositionalEncoding(d_model)
        # 使用ModuleList对多个encoder进行堆叠，因为后续的encoder没有使用词向量和位置编码，所以抽离出来
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        # 传入的enc_inputs的shape是[batch_size, source_len]
        '''编码部分'''
        # 通过nn.Embedding将输入进行编码, enc_outputs的shape是[batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs)
        # 将编码好的输入传入PositionalEncoding类添加位置编码, enc_outputs的shape是[batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1)

        ''''''
        # 输入的句子不可能长度都一致，为了将一个batch构造成一个矩阵方便网络计算，会根据最大的句子长度来构造矩阵，大于最大长度的
        # 直接截断不要，小于最大长度的用pad符号填充
        # get_attn_pad_mask是为了得到句子中pad的位置信息，给到模型后面，在计算自注意力和交互注意力的时候去掉pad符号的影响
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        # 在计算中用不到，这个列表用于保存得到的权重值，主要是为了画热力图，看各个词之间的关系之类的
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

'''Dcoder的一个Block的结构，包含了多头自注意力结构、交互注意力结构和Feed Forward结构'''
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        # Decoder中的自注意力层
        self.dec_self_attn = MultiHeadAttention()
        # Decoder和Encoder的交互注意力层
        self.dec_enc_attn = MultiHeadAttention()
        # Feed Forward神经网络
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: Decoder的输入
                    [batch_size, tgt_len, d_model]
        enc_outputs: Encoder的输出
                     [batch_size, src_len, d_model]
        dec_self_attn_mask: Decoder中自注意力层的mask，用于防止注意力层看到当前位置后面的词
                            [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: Decoder和Encoder之间的交互注意力层中的mask，用于QKV计算时识别pad符号所在的位置
                           [batch_size, tgt_len, src_len]
        '''

        # Decoder多头自注意力层的输入QKV是相等的，都等于Decoder的输入
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn(权重矩阵): [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # Decoder和Encoder的交互注意力层的输入中的Q来自Decoder的计算结果，KV来自Encoder的计算结果
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn(softmax之后的权重矩阵): [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        # Feed Forward层
        # dec_outputs: [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_ffn(dec_outputs)
        # 输出结果和两个权重矩阵
        return dec_outputs, dec_self_attn, dec_enc_attn

'''Decoder完整结构'''
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # 将decoder的输入进行编码
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        # 添加位置编码
        self.pos_emb = PositionalEncoding(d_model)
        # 将Decoder中的block复制n_layers次
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''

        dec_inputs: Decoder的输入
                    [batch_size, tgt_len]
        enc_inputs: Encoder的输入
                    [batch_size, src_len]
        enc_outputs: Encoder的输出，也就是Encoder-Decoder交互注意力层中的K,V输入，K=V=enc_outputs
                     [batch_size, src_len, d_model]
        '''

        # 将decoder的输入进行编码
        dec_outputs = self.tgt_emb(dec_inputs)
        # 添加位置编码
        # dec_outputs: [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(device)
        # 自注意力层的pad部分
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(device)
        # 自注意力层的mask部分，用一个上三角mask使自注意力层看不到当前单词之后的词
        dec_self_attn_subsequent_mask = get_attn_subsequence_mask(dec_inputs).to(device)
        # 两个矩阵相加，大于0的为1，不大于0的为0，为1的在之后就会被设为无穷小
        # torch.gt(a, value)的作用是将a中各个位置上的元素与value比较，若大于value则该位置取1，否则取0
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0).to(device)
        # 交互注意力层的mask矩阵，enc_inputs是k，去看k里面有哪些是pad符号，给到后面的模型
        # 注意！dec_inputs是q，里面也是有pad符号的，但是不在意
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        # 设置两个列表来存放权重矩阵
        dec_self_attns, dec_enc_attns = [], []
        # 循环n_layers次
        for layer in self.layers:
            # 通过DecoderLayer()得到最后的输出和两个权重矩阵
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            # 将两个权重矩阵分别加入列表
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # 实例化Encode类
        self.encoder = Encoder()
        # 实例化Decoder类
        self.decoder = Decoder()
        # 最后的全连接层，tgt_vocab_size是词库的len，也就是每个单词有tgt_vocab_size个概率得分，取概率最大的那个作为输出
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias = False)

    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: Encoder的输入
                    [batch_size, src_len]
        dec_inputs: Decoder的输入
                    [batch_size, tgt_len]
        '''
        # enc_outputs:[batch_size, src_len, d_model], enc_self_attns:[n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outputs:[batch_size, tgt_len, d_model], dec_self_attns:[n_layers, batch_size, n_heads, tgt_len, tgt_len]
        # dec_enc_attns:[n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # 全连接层，dec_logits的shape是[batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)
        # 返回的dec_logits是二维数组，第一维大小是batch_size*tgt_len，代表了一个batch中一共有多少个单词
        # 第二维是tgt_vocab_size，代表了每个单词对应于词库每个单词的概率
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

def train(model, epoch, dataloader, optimizer, criterion, device):
    model.to(device)
    model.train()
    for epoch in range(epoch):
        for enc_inputs, dec_inputs, dec_outputs in dataloader:
            '''
            enc_inputs:[batch_size, src_len]
            dec_inputs:[batch_size, tgt_len]
            dec_outputs:[batch_size, tgt_len]
            '''
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            # outputs:[batch_size * tgt_len, tgt_vocab_size]
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            # 这里因为outputs的第一维是batch_size乘以句子长度，所以相应的dec_outputs也要变成一维的和outputs一一对应
            # dec_outputs.view(-1):[batch_size * tgt_len * tgt_vocab_size]
            loss = criterion(outputs, dec_outputs.view(-1))
            print('Epoch:','%04d'%(epoch+1),'loss =', '{:.6f}'.format(loss))
            # 梯度回传并更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def dec(model, dataloader, device):
    model.eval()
    # 在测试的时候只有enc_inputs，没有dec_inputs和dec_outputs
    enc_inputs, _, _ = next(iter(dataloader))
    for i in range(len(enc_inputs)):
        greedy_dec_predict = greedy_decoder(model, enc_inputs[i].view(1,-1).to(device), start_symbol=tgt_vocab['S'])
        print(enc_inputs[i], '----->', greedy_dec_predict.squeeze())
        print([src_idx2word[t.item()] for t in enc_inputs[i]], '------>',
              [tgt_idx2word[n.item()] for n in greedy_dec_predict.squeeze()])
'''
在预测时，我们不知道Decoder的input，所以采用一个一个词进行输入，将上一个词的预测结果当做下一个预测的输入
贪心解码：每一步都选择概率最大的词，相当于Beam Search中K=1
'''
def greedy_decoder(model, enc_input, start_symbol):
    '''
    model: Transformer model
    enc_input: Encoder的输入
               [batch_size, src_len]
    start_symbol: 起始符号，在这里起始符号是'S'，在tgt_vocab里面对应的编号就是6
    '''
    # 先获取Encoder的输出
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    # 用0来初始化Decoder的输入，torch.zeros(1, 0)代表创建一个长度为1的空tensor：[]
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    # terminal表示是否终止预测，只有在预测得到结束符号的时候terminal才会被设为True
    terminal = False
    # 下一个输入就是上一个输出
    next_symbol = start_symbol
    while not terminal:
        # dec_input:[batch_size, tgt_len]
        # 随着预测的进行，dec_input会将每次新预测出来的单词添加进去，比如添加了起始符号进去之后dec_inputs=[[6]]，添加了第一个词进去之后是[[6,1]]
        dec_input = torch.cat([dec_input.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)], -1)
        # dec_outputs:[batch_size, tgt_len, d_model]，预测中batch_size一直为1，tgt_len就是到当前为止出现的单词个数，d_model就是512维
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        # projected:[batch_size, tgt_len, vocab_len]，将d_model投影到词库的长度
        projected = model.projection(dec_outputs)
        # prob:[batch_size, tgt_len]，得到每个单词对应唯一的预测输出
        prob = projected.squeeze(0).max(dim=-1, keepdim = False)[1]
        # 增量更新（我们希望重复单词预测结果是一样的）
        # 我们在预测时会选择性忽略重复的预测的词，只摘取最新预测的单词拼接到输入序列中
        # 拿出当前预测的词（数字），next_word:[tgt_len]
        next_word = prob.data[-1]
        next_symbol = next_word
        # 如果预测到了结束符号，则将terminal设为True
        if next_symbol == tgt_vocab['E']:
            terminal = True
    # greedy_dec_predict:[batch_size, len_tgt-1]，注意这里不输出起始符号
    greedy_dec_predict = dec_input[:,1:]
    return greedy_dec_predict


if __name__ == '__main__':
    model = Transformer()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    train(model, 10, loader, optimizer, criterion, device)
    dec(model, loader, device)
