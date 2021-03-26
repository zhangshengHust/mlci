import torch
import torch.nn as nn
from fc import FCNet
from torch.nn.utils.weight_norm import weight_norm

class TCNet(nn.Module):
    def __init__(self, v_dim, q_dim, a_dim, h_dim, h_out, rank, glimpse, act='ReLU', dropout=[.2, .5], k=1):
        super(TCNet, self).__init__()

        self.v_dim = v_dim
        self.q_dim = q_dim
        self.a_dim = a_dim
        self.h_out = h_out
        self.rank  = rank
        self.h_dim = h_dim*k
        self.hv_dim = int(h_dim/rank)
        self.hq_dim = int(h_dim/rank)
        self.ha_dim = int(h_dim/rank)


        self.v_tucker = FCNet([v_dim, self.h_dim], act=act, dropout=dropout[1])
        self.q_tucker = FCNet([q_dim, self.h_dim], act=act, dropout=dropout[0])
        self.a_tucker = FCNet([a_dim, self.h_dim], act=act, dropout=dropout[0])

        if self.h_dim < 1024:
            self.a_tucker = FCNet([a_dim, self.h_dim], act=act, dropout=dropout[0])
            self.v_net = nn.ModuleList([FCNet([self.h_dim, self.hv_dim], act=act, dropout=dropout[1]) for _ in range(rank)])
            self.q_net = nn.ModuleList([FCNet([self.h_dim, self.hq_dim], act=act, dropout=dropout[0]) for _ in range(rank)])
            self.a_net = nn.ModuleList([FCNet([self.h_dim, self.ha_dim], act=act, dropout=dropout[0]) for _ in range(rank)])

            if h_out > 1:
                self.ho_dim = int(h_out / rank)
                h_out = self.ho_dim

            self.T_g = nn.Parameter(torch.Tensor(1, rank, self.hv_dim, self.hq_dim, self.ha_dim, glimpse, h_out).normal_())
        self.dropout = nn.Dropout(dropout[1])


    def forward(self, v, q, a):
        f_emb = 0
        v_tucker = self.v_tucker(v)
        q_tucker = self.q_tucker(q)
        a_tucker = self.a_tucker(a)
        for r in range(self.rank):
            v_ = self.v_net[r](v_tucker)
            q_ = self.q_net[r](q_tucker)
            a_ = self.a_net[r](a_tucker)
            f_emb = ModeProduct(self.T_g[:, r, :, :, :, :, :], v_, q_, a_, None) + f_emb

        return f_emb.squeeze(4)

    def forward_with_weights(self, v, q ,a ,w):
        v_ = self.v_tucker(v).transpose(2, 1) #b x d x v
        q_ = self.q_tucker(q).transpose(2, 1).unsqueeze(3) #b x d x q x 1
        a_ = self.a_tucker(a).transpose(2, 1).unsqueeze(3) #b x d x a
        logits = torch.einsum('bdv,bvqa,bdqi,bdaj->bdij',[v_,w,q_,a_])
        logits = logits.squeeze(3).squeeze(2)
        return logits
    
class TriAttention(nn.Module):
    def __init__(self, v_dim, q_dim, a_dim, h_dim, h_out, rank, glimpse, k, dropout=[.2, .5]):
        super(TriAttention, self).__init__()
        self.glimpse = glimpse
        self.TriAtt = TCNet(v_dim, q_dim, a_dim, h_dim, h_out, rank, glimpse, dropout=dropout, k=k)

    def forward(self, v, q, a):
        v_num = v.size(1)
        q_num = q.size(1)
        a_num = a.size(1)
        logits = self.TriAtt(v, q, a)
        if logits.dim() == 4:
            logits = logits.unsqueeze(dim=4)
        mask = (0 == v.abs().sum(2)).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(logits.size())
        logits.data.masked_fill_(mask.data, -float('inf'))

        p = torch.softmax(logits.contiguous().view(-1, v_num*q_num*a_num, self.glimpse), 1)
        return p.view(-1, v_num, q_num, a_num, self.glimpse), logits

def ModeProduct(tensor, matrix_1, matrix_2, matrix_3, matrix_4, n_way=3):

    #mode-1 tensor product
    tensor_1 = tensor.transpose(3,2).contiguous().view(tensor.size(0), tensor.size(1), tensor.size(2)*tensor.size(3)*tensor.size(4))
    tensor_product = torch.matmul(matrix_1, tensor_1)
    tensor_1 = tensor_product.view(-1, tensor_product.size(1),tensor.size(4), tensor.size(3), tensor.size(2)).transpose(4,2)

    #mode-2 tensor product
    tensor_2 = tensor_1.transpose(2,1).transpose(4,2).contiguous().view(-1, tensor_1.size(2), tensor_1.size(1)*tensor_1.size(3)*tensor_1.size(4))
    tensor_product = torch.matmul(matrix_2, tensor_2.float())
    tensor_2 = tensor_product.view(-1, tensor_product.size(1), tensor_1.size(4), tensor_1.size(3), tensor_1.size(1)).transpose(4,1).transpose(4,2)
    tensor_product = tensor_2
    if n_way > 2:
        #mode-3 tensor product
        tensor_3 = tensor_2.transpose(3,1).transpose(4,2).transpose(4,3).contiguous().view(-1, tensor_2.size(3), tensor_2.size(2)*tensor_2.size(1)*tensor_2.size(4))
        tensor_product = torch.matmul(matrix_3, tensor_3.float())
        tensor_3 = tensor_product.view(-1, tensor_product.size(1), tensor_2.size(4), tensor_2.size(2), tensor_2.size(1)).transpose(1,4).transpose(4,2).transpose(3,2)
        tensor_product = tensor_3
    if n_way > 3:
    #mode-4 tensor product
        tensor_4 = tensor_3.transpose(4,1).transpose(3,2).contiguous().view(-1, tensor_3.size(4), tensor_3.size(3)*tensor_3.size(2)*tensor_3.size(1))
        tensor_product = torch.matmul(matrix_4, tensor_4)
        tensor_4 = tensor_product.view(-1, tensor_product.size(1), tensor_3.size(3), tensor_3.size(2), tensor_3.size(1)).transpose(4,1).transpose(3,2)
        tensor_product = tensor_4

    return tensor_product