import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.weight_norm as weight_norm
from fc import FCNet


class BlFusion(nn.Module):

    def __init__(self, v_dim=2048, q_dim=1280, h_dim=1280, dropout = [.2, .0], k=5):
        super(BlFusion, self).__init__()
        
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.v_net = FCNet([v_dim,h_dim*k], dropout=dropout[0])
        self.q_net = FCNet([q_dim,h_dim*k], dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])
        if k>1:
            self.p_net = nn.AvgPool1d(k,stride = k)

    def forward(self, v, q):
        
        v_emb = self.dropout(self.v_net(v))
#         v_emb = self.v_net(v)
        q_emb = self.q_net(q)
        
        logits = v_emb*q_emb
        
        if self.k>1:
            logits = logits.unsqueeze(dim=1)
            logits = self.p_net(logits).squeeze(dim=1)*self.k
        
        return logits

class BlFusion2d(nn.Module):
    
    def __init__(self, v_dim=2048, q_dim=1280, h_dim=1280, dropout = [.2, .0], k=5):
        super(BlFusion2d, self).__init__()
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.blf = BlFusion(v_dim, q_dim, h_dim, dropout, k)
        
    def forward(self, v, q):
        # v and q have same dimension
        
        bs = v.size(0)
        num = v.size(1)
        
        if not v.is_contiguous():
            v = v.contiguous()
        if not q.is_contiguous():
            q = q.contiguous()
        
        x_v = v.view(bs * num, self.v_dim)
        x_q = q.view(bs * num, self.q_dim)
        x_mm = self.blf(x_v, x_q)
        x_mm = x_mm.view(bs, num, self.h_dim)
        
        return x_mm

class Bfusion(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, com=False):
        self.q_net = FCNet([q_dim, num_hid])
        self.v_net = FCNet([v_dim, num_hid])
        self.com = com
        if com:
            layers = [
            nn.Dropout(0.2, inplace=True),
            weight_norm(nn.Linear(num_hid, num_hid), dim=None)#,
            #nn.ReLU()
        ]
            self.f = nn.Sequential(*layers)
            
    def forward(self,v,q):
        v_emb = self.v_net(v)
        q_emb = self.q_net(q)
        jion_emb = v_emb*q_emb
        if com:
            jion_emb = self.f(jion_emb)
        return jion_emb