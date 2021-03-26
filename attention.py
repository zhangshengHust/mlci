import torch
import torch.nn as nn
from layers import NonLinearElementMultiply, TransformLayer
from fc import FCNet
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
from fusion import BlFusion
import math

class AttentionTextEmbedding(nn.Module):
    def __init__(self, hidden_dim = 1024, dropout_ = 0.2, **kwargs):
        super(AttentionTextEmbedding, self).__init__()

        self.text_out_dim = hidden_dim * kwargs["conv2_out"]

        self.dropout = nn.Dropout(p=dropout_)

        conv1_out = kwargs["conv1_out"]
        conv2_out = kwargs["conv2_out"]

        self.transform = FCNet([hidden_dim, conv1_out], dropout=0.2)
        
        layers = [
            nn.Dropout(0.2, inplace=False),
            weight_norm(nn.Linear(conv1_out, conv2_out), dim=None)
        ]
       
        # self.proj = nn.Linear(conv2_out * hidden_dim, self.text_out_dim)
        self.atte = nn.Sequential(*layers)

    def forward(self, x, mask):
        batch_size = x.size(0)

        qatt_f1 = self.transform(x)        # N x T x conv1_out
        attention = self.atte(qatt_f1)           # N x T x conv2_out

        attention.data.masked_fill_(mask.unsqueeze(-1).expand_as(attention).data,-float('inf'))
        atte_softmax = nn.functional.softmax(attention, dim=1)
        qtt_feature = torch.bmm(atte_softmax.transpose(-1,-2), x)   # 2XT  T*hidden_dim
        
        qtt_feature_concat = qtt_feature.view(batch_size, -1)       # g X hidden_dim
#         qtt_feature_concat = qtt_feature.sum(dim = 1)       # g X hidden_dim
#         qtt_feature_concat = self.proj(qtt_feature_concat)
        return qtt_feature_concat

class Channel_attention(nn.Module):
    def __init__(self, dim_x, dim_y, h_dim, f_dropout=[0.2, 0], dropout = 0.2, k=1, **arg):
        super(Channel_attention, self).__init__()
        self.fusion = BlFusion(dim_x, dim_y, h_dim, f_dropout, k)

        layersx = [
            nn.Dropout(dropout, inplace=False),
            weight_norm(nn.Linear(h_dim, dim_x), dim=None),
            nn.Sigmoid()  ## softmax
        ]
        
        layersy = [
            nn.Dropout(dropout, inplace=False),
            weight_norm(nn.Linear(h_dim, dim_y), dim=None),
            nn.Sigmoid()  ## softmax
        ]
        
        self.x_attention = nn.Sequential(*layersx)
        self.y_attention = nn.Sequential(*layersy)

    def forward(self, x, y):
        '''
        x : b, n, dim1
        y : b, m, dim2
        '''
        x_channel_satistic = F.avg_pool1d(x.transpose(-1,-2),kernel_size=x.size(1)).squeeze(dim=2)
        y_channel_satistic = F.avg_pool1d(y.transpose(-1,-2),kernel_size=y.size(1)).squeeze(dim=2)     
        f_feature = self.fusion(x_channel_satistic, y_channel_satistic)
        x_atte = self.x_attention(f_feature) # b dim1
        y_atte = self.y_attention(f_feature) # b dim2
        x_channel_feature = x_atte.unsqueeze(dim=1)*x
        y_channel_feature = y_atte.unsqueeze(dim=1)*y
        
        return x_channel_feature, y_channel_feature

class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    """
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2,.5], k=1):
        super(BCNet, self).__init__()
        
        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim 
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1]) # attention
        
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)
        
        if None == h_out:
            pass
        elif h_out <= self.c:     #  小于 32       1, 2, 1, h_dim * self.k
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v).transpose(1,2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1,2).unsqueeze(2)
            d_ = torch.matmul(v_, q_) # b x h_dim x v x q
            logits = d_.transpose(1,2).transpose(2,3) # b x v x q x h_dim
            return logits

        # broadcast Hadamard product, matrix-matrix production
        # fast computation but memory inefficient
        # epoch 1, time: 157.84
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v)).unsqueeze(1)
            q_ = self.q_net(q)
            h_ = v_ * self.h_mat # broadcast, b x h_out x v x h_dim
            logits = torch.matmul(h_, q_.unsqueeze(1).transpose(2,3)) # b x h_out x v x q
            logits = logits + self.h_bias
            return logits # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        # epoch 1, time: 304.87
        else: 
            v_ = self.dropout(self.v_net(v)).transpose(1,2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1,2).unsqueeze(2)
            d_ = torch.matmul(v_, q_) # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1,2).transpose(2,3)) # b x v x q x h_out
            return logits.transpose(2,3).transpose(1,2) # b x h_out x v x q

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v).transpose(1,2).unsqueeze(2) # b x d x 1 x v
        q_ = self.q_net(q).transpose(1,2).unsqueeze(3) # b x d x q x 1
        logits = torch.matmul(torch.matmul(v_, w.unsqueeze(1)), q_) # b x d x 1 x 1
        logits = logits.squeeze(3).squeeze(2)
        if 1 < self.k:
            logits = logits.unsqueeze(1) # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k # sum-pooling
        return logits
    
class NormalSubLayer(nn.Module):

    def __init__(self, dim1, dim2, num_attn, num_none1, num_none2, dropout, is_multi_head = False, gated = False, dropattn=0.0, **arg):
        super(NormalSubLayer, self).__init__()
        self.dense_coattn = DenseCoAttn(dim1, dim2, num_attn, num_none1, num_none2, dropattn, is_multi_head = is_multi_head, gated = gated)
        if not is_multi_head:
            dim1_add = dim2
            dim2_add = dim1
        else:
            dim1_add = self.dense_coattn.dim
            dim2_add = self.dense_coattn.dim
        self.linears = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Linear(dim1 + dim1_add, dim1), dim=None),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ),
            nn.Sequential(
                weight_norm(nn.Linear(dim2 + dim2_add, dim2), dim=None),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
        ])

    def forward(self, data1, data2, mask1, mask2):
#         print("mask1:", mask1)
#         print("mask2:", mask2)
        weighted1, weighted2 = self.dense_coattn(data1, data2, mask1, mask2)
        weighted1 = weighted1.view(weighted1.size(0), weighted1.size(1), -1)
        weighted2 = weighted2.view(weighted2.size(0), weighted2.size(1), -1)
        
#         print("weighted1", weighted1.shape,"weighted2", weighted2.shape)
#         print(self.linears)
        data1 = data1 + self.linears[0](torch.cat([data1, weighted2], dim=2))
        data2 = data2 + self.linears[1](torch.cat([data2, weighted1], dim=2))

        return data1, data2

class DenseCoAttn(nn.Module):

    def __init__(self, dim1, dim2, num_attn, num_none1, num_none2, dropout, is_multi_head = False, gated = True):
        super(DenseCoAttn, self).__init__()
        self.dim =  1280 # min(dim1, dim2)
        self.linears = nn.ModuleList([nn.Linear(dim1, self.dim, bias=False),
                                      nn.Linear(dim2, self.dim, bias=False)])
        
        self.nones = nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_none1, dim1))),
                                       nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_none2, dim2)))])
        self.d_k = self.dim // num_attn     # glimpse
        self.h = num_attn
        self.num_none1 = num_none1
        self.num_none2 = num_none2
        
        self.is_multi_head = is_multi_head
        self.attn = None
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(2)])
        
        self.gated = gated
        if gated:
            self.gated_atte = BCNet(self.d_k, self.d_k, self.d_k, 2)

    def forward(self, value1, value2, mask1=None, mask2=None):
        batch = value1.size(0)
        dim1, dim2 = value1.size(-1), value2.size(-1)
        value1 = torch.cat([self.nones[0].unsqueeze(0).expand(batch, self.num_none1, dim1), value1], dim=1)
        value2 = torch.cat([self.nones[1].unsqueeze(0).expand(batch, self.num_none2, dim2), value2], dim=1)
        none_mask1 = value1.new_ones((batch, self.num_none1))
        none_mask2 = value1.new_ones((batch, self.num_none2))

        if mask1 is not None:
            mask1 = torch.cat([none_mask1, mask1], dim=1)
            mask1 = mask1.unsqueeze(1).unsqueeze(2)  # b, 1, 1, n
        if mask2 is not None:
            mask2 = torch.cat([none_mask2, mask2], dim=1)
            mask2 = mask2.unsqueeze(1).unsqueeze(2)  # b, 1, 1, n
        
        # b, n, g, d_k   --> b, g, n, d_k                                           
        query1, query2 = [l(x).view(batch, -1, self.h, self.d_k).transpose(1, 2) 
            for l, x in zip(self.linears, (value1, value2))] 

        if self.is_multi_head:
            weighted1, attn1 = self.qkv_attention(query2, query1, query1, mask_key=mask1, mask_query=mask2, dropout=self.dropouts[0])
            weighted1 = weighted1.transpose(1, 2).contiguous()[:, self.num_none2:, :]                         # 去掉了加的虚拟点的部分
            weighted2, attn2 = self.qkv_attention(query1, query2, query2, mask_key=mask2, mask_query=mask1, dropout=self.dropouts[1])
            weighted2 = weighted2.transpose(1, 2).contiguous()[:, self.num_none1:, :]                         # 去掉了加的虚拟点的部分
        
        else:
            weighted1, attn1 = self.qkv_attention(query2, query1, value1.unsqueeze(1), mask_key=mask1, mask_query=mask2, 
                dropout=self.dropouts[0])
            weighted1 = weighted1.mean(dim=1)[:, self.num_none2:, :]
            weighted2, attn2 = self.qkv_attention(query1, query2, value2.unsqueeze(1), mask_key=mask2, mask_query=mask1, 
                dropout=self.dropouts[1])
            weighted2 = weighted2.mean(dim=1)[:, self.num_none1:, :]

        self.attn = [attn1[:,:,self.num_none2:,self.num_none1:], attn2[:,:,self.num_none1:,self.num_none2:]]

        return weighted1, weighted2
    
    def qkv_attention(self, query, key, value, mask_key=None, mask_query=None, dropout=None):
        
        d_k = query.size(-1)
#         print("query size", query.size())
#         print("key size", key.size())        
        ## 计算 gated attention
        if self.gated:
            atte = []
            for g in range(self.h):
                attention = torch.sigmoid(self.gated_atte(query[:,g], key[:,g]))
                atte.append(attention[:,1] * attention[:,0])
            attention = torch.stack(atte, dim=1)
#             print("Attention ",attention)
            scores = torch.matmul(query, key.transpose(-2,-1)) * attention / math.sqrt(d_k)
        else:
            scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
        
        if mask_key is not None:
            scores.data.masked_fill_(mask_key.eq(0), -65504.0)   # 对 score 中对应的mask为0的位置填上负无穷

    #     print("scores: ", scores.shape, scores)
        p_attn = F.softmax(scores, dim=-1)

        if mask_query is not None:
            p_attn.data.masked_fill_(mask_query.transpose(-1,-2).eq(0), 0.0)   # 对 score 中对应的mask为0的位置填上0
    #     print("coattention: ", p_attn.shape, p_attn)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

# net = DenseCoAttn(1024,1024,2,0,1,0.0)
# x = torch.randn(128, 100, 1024)
# x_m = torch.zeros(128, 100).float()
# x_m[:,80] = 1
# y = torch.randn(128, 50, 1024)
# y_m = torch.zeros(128, 50).float()
# y_m[:,40] = 1
# weighted1, weighted2 = net(x,y,x_m,y_m)