import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math
import torchvision
from torch.autograd import Variable


# Standard 2 layerd FFN of transformer
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.3):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        # print(f'd_model={d_model}, d_ff={d_ff}')
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

        nn.init.normal_(self.linear_1.weight, std=0.001)
        nn.init.normal_(self.linear_2.weight, std=0.001)

    def forward(self, x):
        # print(f'FF: x size={x.size()}')
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


# standard NORM layer of Transformer
class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6, trainable=True):
        super(Norm, self).__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        if trainable:
            self.alpha = nn.Parameter(torch.ones(self.size))
            self.bias = nn.Parameter(torch.zeros(self.size))
        else:
            self.alpha = nn.Parameter(torch.ones(self.size), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(self.size), requires_grad=False)
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


# Standard positional encoding (addition/ concat both are valid)
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        batch_size = x.size(0)
        seq_len = x.size(1)
        num_feature = x.size(2)

        # print(self.pe.size())
        z = Variable(self.pe[:, :seq_len], requires_grad=False)
        #z = z.unsqueeze(-1).unsqueeze(-1)
        # print(z.size())
        # print((batch_size, seq_len, num_feature))
        z = z.expand(batch_size, seq_len, num_feature)
        x = x + z
        return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(d_k)
    if mask is not None:
        attn = attn.masked_fill(mask == 0, -1e9)
    if dropout:
        attn = dropout(F.softmax(attn, dim=-1))
    output = torch.matmul(attn, v)

    return output
'''
# standard attention layer
def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.sum(q * k, -1) / math.sqrt(d_k)
    # scores : b, t 
    scores = F.softmax(scores, dim=-1)
    scores = scores.unsqueeze(-1).expand(scores.size(0), scores.size(1), v.size(-1))
    # scores : b, t, dim 
    output = scores * v
    output = torch.sum(output, 1)
    if dropout:
        output = dropout(output)
    return output
'''
class Fusion(nn.Module):
    def __init__(self, d_x=16, d_k=16, dropout=0.3):
        super(Fusion, self).__init__()
        self.d_x = d_x
        self.d_k = d_k
        # no of head has been modified to encompass : 1024 dimension
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_x = Norm(d_x)
        self.norm_d = Norm(d_k)
        self.norm_2 = Norm(d_x+d_k)
        self.ff = FeedForward(d_x+d_k, d_ff=int((d_x+d_k)//8))#d_ff=int(d_k / 2))
        self.linear_qx = nn.Linear(self.d_x, self.d_x, bias=False)
        self.linear_kx = nn.Linear(self.d_x, self.d_k, bias=False)
        self.linear_vx = nn.Linear(self.d_x, self.d_k, bias=False)
        self.linear_qd = nn.Linear(self.d_k, self.d_k, bias=False)
        self.linear_kd = nn.Linear(self.d_k, self.d_x, bias=False)
        self.linear_vd = nn.Linear(self.d_k, self.d_x, bias=False)

        nn.init.normal_(self.linear_qx.weight, std=0.001)
        nn.init.normal_(self.linear_kx.weight, std=0.001)
        nn.init.normal_(self.linear_vx.weight, std=0.001)
        nn.init.normal_(self.linear_qd.weight, std=0.001)
        nn.init.normal_(self.linear_kd.weight, std=0.001)
        nn.init.normal_(self.linear_vd.weight, std=0.001)

    def forward(self, x, d, mask=None):
        # q_x = self.linear_qx(x)
        # v_x = self.linear_vx(x)  # value
        # k_x = self.linear_kx(x)  # key
        # q_d = self.linear_qd(d)
        # v_d = self.linear_vd(d)  # value
        # k_d = self.linear_kd(d)  # key
        # # q: (b , dim )
        # # q,k,v : (b, t , d_model=1024 // 16 )
        # A_x = attention(q_x, k_d, v_d, self.d_k, mask, self.dropout)
        # A_d = attention(q_d, k_x, v_x, self.d_x, mask, self.dropout)
        # # A : (b , d_model=1024 // 16 )
        # x = self.norm_x(A_x + x)
        # #d = self.norm_d(A_d + d)
        # d = A_d + d
        xd = torch.cat([x, d], dim=-1)
        out = self.norm_2(xd + self.dropout_2(self.ff(xd)))
        return out


class Tail(nn.Module):
    def __init__(self, num_classes, num_frames):
        super(Tail, self).__init__()
        self.num_frames = num_frames
        #self.bn1 = nn.BatchNorm2d(self.num_features)
        #self.bn2 = Norm(self.d_model, trainable=False)

        self.pos_embd_x = PositionalEncoder(1024, self.num_frames) #128
        self.pos_embd_k = PositionalEncoder(256, self.num_frames) #128
        #self.Qpr = nn.Conv2d(self.num_features, self.d_model, kernel_size=(7, 7), stride=1, padding=0, bias=False)

        # self.lstm_x = nn.LSTM(512, 128, batch_first=True)
        # self.lstm_k = nn.LSTM(16, 128, batch_first=True)

        self.list_layers = Fusion(1024, 256) #128.128

        self.classifier = nn.Linear(1024+256, num_classes) #256
        # resnet style initialization
        #nn.init.kaiming_normal_(self.Qpr.weight, mode='fan_out')
        nn.init.normal_(self.classifier.weight, std=0.001)
        # nn.init.xavier_normal_(self.lstm_x.all_weights[0][0])
        # nn.init.xavier_normal_(self.lstm_x.all_weights[0][1])
        # nn.init.xavier_normal_(self.lstm_k.all_weights[0][0])
        # nn.init.xavier_normal_(self.lstm_k.all_weights[0][1])
        # nn.init.constant(self.classifier.bias, 0)

        #nn.init.constant_(self.bn1.weight, 1)
        #nn.init.constant_(self.bn1.bias, 0)

    def forward(self, x, k):
        # print('Tail.forward', x.size())
        t = self.num_frames
        # print(f'x size={x.size()}')
        # print(f'k size={k.size()}')
        b = x.size(0) // t
        #x = self.bn1(x)
        x = x.view(b, t, -1)
        # print(f'x size={x.size()}')
        # self.lstm_x.flatten_parameters()
        # f_x, _ = self.lstm_x(x)
        k = k.view(b, t, -1)
        # self.lstm_k.flatten_parameters()
        # f_k, _ = self.lstm_k(k)
        #f_x = f_x.contiguous().view(b, t, -1)
        #f_k = f_k.contiguous().view(b, t, -1)
        # stabilizes the learning
        # f_x = self.pos_embd_x(f_x)
        # f_k = self.pos_embd_k(f_k)
        f_x = self.pos_embd_x(x)
        f_k = self.pos_embd_k(k)

        f = self.list_layers(f_x, f_k)
        f = F.normalize(f, p=2, dim=2)
        f = f.view(b*t, -1)

        # if not self.training:
        # return f
        y = self.classifier(f)  # [b/s,n_c]
        return y, f


# base is resnet
# Tail is the main transformer network
class Transformer(nn.Module):
    def __init__(self, num_classes, seq_len):
        super(Transformer, self).__init__()
        self.seq_len = seq_len
        resnet34 = torchvision.models.resnet34(pretrained=True)
        self.base = nn.Sequential(*list(resnet34.children())[:-1])  # -2
        self.tail = Tail(num_classes, seq_len)

    def forward(self, x, k):
        # x = x.view(b*t, x.size(2), x.size(3))
        print(f"1 resnet video {x.size()}")
        x = self.base(x)
        print(f"2 resnet video {x.size()}")
        x = x[:, :, 0, 0]
        print(f"3 resnet video {x.size()}")
        x = self.tail(x, k)
        return x






