import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from trilinear import TriAttention, TCNet
from fc import FCNet
import numpy as np

class Fusion(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus relu'd sum
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # found through grad student descent ;)
        return - (x - y)**2 + F.relu(x + y)

class ReshapeBatchNorm(nn.Module):
    def __init__(self, feat_size, affine=True):
        super(ReshapeBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(feat_size, affine=affine)

    def forward(self, x):
        assert(len(x.shape) == 3)
        batch_size, num, _ = x.shape
        x = x.view(batch_size * num, -1)
        x = self.bn(x)
        return x.view(batch_size, num, -1)

class Trilinear_Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Trilinear_Classifier, self).__init__()
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(in_features, mid_features)
        self.lin2 = nn.Linear(mid_features, out_features)
        self.bn = nn.BatchNorm1d(mid_features)
        #_____________________________________________________________________________________
        v_dim = 512               
        q_dim = 512
        a_dim = 512               
        h_mm = 512               
        rank = 32               
        gamma = 1               
        k = 1               
        h_out = 1               
        self.t_att = TriAttention(v_dim, q_dim, a_dim, h_mm, 1, rank, gamma, k, dropout=[.2, .5])               
        self.t_net = TCNet(v_dim, q_dim, a_dim, h_mm, h_out, rank, 1, dropout=[.2, .5], k=1)               
        #______________________________________________________________________________________

    def forward(self, v, q, c, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, 512]
        q: question            [batch, max_len, 512]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        att, logits = self.t_att(v, q, c)  # b x v x q x a x g
        fusion_f = self.t_net.forward_with_weights(v, q, c, att[:, :, :, :, 0])
        out = self.lin1(self.drop(fusion_f))
        out = self.lin2(self.drop(self.relu(self.bn(out))))
        return out
    
class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(in_features, mid_features)
        self.lin2 = nn.Linear(mid_features, out_features)
        self.bn = nn.BatchNorm1d(mid_features)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, 512]
        q: question            [batch, max_len, 512]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        v_mean = (v * v_mask.unsqueeze(2)).sum(1) / v_mask.sum(1).unsqueeze(1)
        q_mean = (q * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1).unsqueeze(1)
        out = self.lin1(self.drop(v_mean * q_mean))
        out = self.lin2(self.drop(self.relu(self.bn(out))))
        return out

class Ocr_classifier(nn.Module):
    def __init__(self, in_features, mid_features, out_features=1, drop=0.0):
        super(Ocr_classifier, self).__init__()
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(in_features, mid_features)
        self.lin2 = nn.Linear(mid_features, out_features)
        
    def forward(self, c, q, q_mask):
        
        # c : b n 512
        # q : b m 512
        # question average pooling
        q_mean = (q * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1).unsqueeze(1) #  b, 512
        out = self.lin1(self.drop(c * q_mean.unsqueeze(dim=1)))
        out = self.lin2(self.drop(self.relu(out)))
        
        return out

class SingleBlock(nn.Module):
    """
    Single Block Inter-/Intra-modality stack multiple times
    """
    def __init__(self, num_block, v_size, q_size, c_size, output_size, num_inter_head, num_intra_head, drop=0.0):
        super(SingleBlock, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.c_size = c_size
        self.output_size = output_size
        self.num_inter_head = num_inter_head
        self.num_intra_head = num_intra_head
        self.num_block = num_block

        self.v_lin = nn.Linear(v_size, output_size)
        self.q_lin = nn.Linear(q_size, output_size)
        self.c_lin = nn.Linear(c_size, output_size)
        
        self.interBlock = InterModalityUpdate(output_size, output_size, output_size, output_size, num_inter_head, drop)
        self.intraBlock = DyIntraModalityUpdate(output_size, output_size, output_size ,output_size, num_intra_head, drop)
        
        self.layer_norm1v = nn.LayerNorm(output_size)
        self.layer_norm1q = nn.LayerNorm(output_size)
        self.layer_norm1c = nn.LayerNorm(output_size)

        self.layer_norm2v = nn.LayerNorm(output_size)
        self.layer_norm2q = nn.LayerNorm(output_size)
        self.layer_norm2c = nn.LayerNorm(output_size)
        
        self.drop = nn.Dropout(drop)

    def forward(self, v, q, c, v_mask, q_mask, c_mask, vbbox, cbbox):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        c: context            [batch, max_len_c, feat_size]
        
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        c_mask                 [batch, max_len_c]
        
        """
        # transfor features
        v = self.v_lin(self.drop(v))
        q = self.q_lin(self.drop(q))
        c = self.c_lin(self.drop(c))
        
        for i in range(self.num_block):
#             v, q, c = self.interBlock(v, q, c, v_mask, q_mask, c_mask, vbbox, cbbox)
#             v, q, c = self.intraBlock(v, q, c, v_mask, q_mask, c_mask)
            v1, q1, c1 = self.interBlock(v, q, c, v_mask, q_mask, c_mask, vbbox, cbbox)
            v = self.layer_norm1v(v + v1)
            q = self.layer_norm1q(q + q1)
            c = self.layer_norm1c(c + c1)
            
            v2, q2, c2 = self.intraBlock(v, q, c, v_mask, q_mask, c_mask)
            v = self.layer_norm2v(v + v2)
            q = self.layer_norm2q(q + q2)
            c = self.layer_norm2c(c + c2)

        return v,q,c 

class MultiBlock(nn.Module):
    """
    Multi Block Inter-/Intra-modality
    """
    def __init__(self, num_block, v_size, q_size, c_size, output_size, num_inter_head, num_intra_head, drop=0.0):
        super(MultiBlock, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.c_size = c_size
        self.output_size = output_size
        self.num_inter_head = num_inter_head
        self.num_intra_head = num_intra_head
        self.num_block = num_block

        self.v_lin = nn.Linear(v_size, output_size)
        self.q_lin = nn.Linear(q_size, output_size)
        self.c_lin = nn.Linear(c_size, output_size)
        
        self.layer_norm1v = nn.LayerNorm(output_size)
        self.layer_norm1q = nn.LayerNorm(output_size)
        self.layer_norm1c = nn.LayerNorm(output_size)

        self.layer_norm2v = nn.LayerNorm(output_size)
        self.layer_norm2q = nn.LayerNorm(output_size)
        self.layer_norm2c = nn.LayerNorm(output_size)
        
        self.drop = nn.Dropout(drop)

        blocks = []
#         blocks.append(InterModalityUpdate(v_size, q_size, c_size, output_size, num_inter_head, drop))
#         blocks.append(DyIntraModalityUpdate(output_size, output_size, output_size, output_size, num_intra_head, drop))
        
        for i in range(num_block):
            blocks.append(InterModalityUpdate(output_size, output_size, output_size, output_size, num_inter_head, drop))
            blocks.append(DyIntraModalityUpdate(output_size, output_size, output_size ,output_size, num_intra_head, drop))
        
        self.multi_blocks = nn.ModuleList(blocks)
        
    def forward(self, v, q, c, v_mask, q_mask, c_mask, vbbox, cbbox):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        
        v = self.v_lin(self.drop(v))
        q = self.q_lin(self.drop(q))
        c = self.c_lin(self.drop(c))
        mlevelr = []
        for i in range(self.num_block):
            v1, q1, c1 = self.multi_blocks[i*2+0](v, q, c, v_mask, q_mask, c_mask, vbbox, cbbox)
            v = self.layer_norm1v(v + v1)
            q = self.layer_norm1q(q + q1)
            c = self.layer_norm1c(c + c1)
            v2, q2, c2 = self.multi_blocks[i*2+1](v, q, c, v_mask, q_mask, c_mask)
            v = self.layer_norm2v(v + v2)
            q = self.layer_norm2q(q + q2)
            c = self.layer_norm2c(c + c2)
            mlevelr.append((v, q, c))
        
        return mlevelr

class InterModalityUpdate(nn.Module):
    """
    Inter-modality Attention Flow
    """
    def __init__(self, v_size, q_size, c_size, output_size, num_head, drop=0.0):
        super(InterModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.c_size = c_size
        
        self.output_size = output_size
        self.num_head = num_head

        self.v_lin = nn.Linear(v_size, output_size * 3)
        self.q_lin = nn.Linear(q_size, output_size * 3)
        self.c_lin = nn.Linear(c_size, output_size * 3)
        
        # memory
        mv = 1
        self.m_v_k = nn.Parameter(torch.FloatTensor(1, mv, output_size))  #bs, m, 
        self.m_v_v = nn.Parameter(torch.FloatTensor(1, mv, output_size))
        self.m_v_b = nn.Parameter(torch.FloatTensor(1, mv, 4))
        mc = 1
        self.m_c_k = nn.Parameter(torch.FloatTensor(1, mc, output_size))  #bs, m, 
        self.m_c_v = nn.Parameter(torch.FloatTensor(1, mc, output_size))
        self.m_c_b = nn.Parameter(torch.FloatTensor(1, mc, 4))
        # memory end
        
        self.v_output = nn.Linear(output_size*2 + v_size, output_size)
        self.q_output = nn.Linear(output_size*2 + q_size, output_size)
        self.c_output = nn.Linear(output_size*2 + c_size, output_size)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)

        self.sinteraction = Spatial_attention()

    def forward(self, v, q, c, v_mask, q_mask, c_mask, vbbox, cbbox):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        c: context             [batch, max_len_c, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        c_mask                 [batch, max_len_c]
        vbbox                  [batch, num_obj, 4]
        cbbox                  [batch, max_len_c, 4]
        """
        
        batch_size, num_obj = v_mask.shape
        _         , max_len = q_mask.shape
        _         , max_len_c = c_mask.shape
        
        # transfor features
        v_trans = self.v_lin(self.drop(self.relu(v)))
        q_trans = self.q_lin(self.drop(self.relu(q)))
        c_trans = self.c_lin(self.drop(self.relu(c)))
        
        # mask all padding object/word features
        v_trans = v_trans * v_mask.unsqueeze(2)
        q_trans = q_trans * q_mask.unsqueeze(2)
        c_trans = c_trans * c_mask.unsqueeze(2)
        
        # split for different use of purpose
        v_k, v_q, v_v = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        q_k, q_q, q_v = torch.split(q_trans, q_trans.size(2) // 3, dim=2)
        c_k, c_q, c_v = torch.split(c_trans, c_trans.size(2) // 3, dim=2)
        
        # memory concat *
        mv = self.m_v_k.size(1)
        m_v_k = np.sqrt(self.m_v_k.size(2)) * self.m_v_k.expand(batch_size, self.m_v_k.size(1), self.m_v_k.size(2))
        m_v_v = np.sqrt(self.m_v_v.size(1)) * self.m_v_v.expand(batch_size, self.m_v_v.size(1), self.m_v_v.size(2))
        m_v_b = self.m_v_b.expand(batch_size, self.m_v_b.size(1), self.m_v_b.size(2))
        v_k = torch.cat([m_v_k, v_k],dim=1)
        v_v = torch.cat([m_v_v, v_v],dim=1)
        vbbox = torch.cat([m_v_b, vbbox],dim=1)
        # compute mask
        v_mask = torch.cat([torch.ones(batch_size, mv).cuda(v_mask.get_device()), v_mask], dim=1)
        
        mc = self.m_c_k.size(1)
        m_c_k = np.sqrt(self.m_c_k.size(2)) * self.m_c_k.expand(batch_size, self.m_c_k.size(1), self.m_c_k.size(2))
        m_c_v = np.sqrt(self.m_c_v.size(1)) * self.m_c_v.expand(batch_size, self.m_c_v.size(1), self.m_c_v.size(2))
        m_c_b = self.m_c_b.expand(batch_size, self.m_c_b.size(1), self.m_c_b.size(2))
        c_k = torch.cat([m_c_k, c_k], dim=1)
        c_v = torch.cat([m_c_v, c_v], dim=1)
        cbbox = torch.cat([m_c_b, cbbox], dim=1)
        # compute mask
        c_mask = torch.cat([torch.ones(batch_size, mc).cuda(c_mask.get_device()),c_mask], dim=1)
        # memory end *
        
        # spatial attention
        sa = self.sinteraction(vbbox, cbbox) # b, num_obj, max_len_c
#         sa = F.pad(sa,pad=(1,0,1,0),value = 1)
#         print("sa", sa.shape, sa)
        
        # apply multi-head
        vk_set = torch.split(v_k, v_k.size(2) // self.num_head, dim=2)
        vq_set = torch.split(v_q, v_q.size(2) // self.num_head, dim=2)
        vv_set = torch.split(v_v, v_v.size(2) // self.num_head, dim=2)
        
        qk_set = torch.split(q_k, q_k.size(2) // self.num_head, dim=2)
        qq_set = torch.split(q_q, q_q.size(2) // self.num_head, dim=2)
        qv_set = torch.split(q_v, q_v.size(2) // self.num_head, dim=2)
        
        ck_set = torch.split(c_k, c_k.size(2) // self.num_head, dim=2)
        cq_set = torch.split(c_q, c_q.size(2) // self.num_head, dim=2)
        cv_set = torch.split(c_v, c_v.size(2) // self.num_head, dim=2)
        
        # multi-head
        for i in range(self.num_head):
            vk_slice, vq_slice, vv_slice = vk_set[i], vq_set[i], vv_set[i]  #[batch, num_obj, feat_size]
            qk_slice, qq_slice, qv_slice = qk_set[i], qq_set[i], qv_set[i]  #[batch, max_len, feat_size]
            ck_slice, cq_slice, cv_slice = ck_set[i], cq_set[i], cv_set[i]  #[batch, max_len, feat_size]
            
            # inner product & set padding object/word attention to negative infinity & normalized by square root of hidden dimension
            q2v = (vq_slice @ qk_slice.transpose(1,2)).masked_fill(q_mask.unsqueeze(1).expand([batch_size, num_obj, max_len]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
            
            v2q = (qq_slice @ vk_slice.transpose(1,2)).masked_fill(v_mask.unsqueeze(1).expand([batch_size, max_len, num_obj+mv]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
            
            c2q = (qq_slice @ ck_slice.transpose(1,2)).masked_fill(c_mask.unsqueeze(1).expand([batch_size, max_len, max_len_c+mc]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
            
            q2c = (cq_slice @ qk_slice.transpose(1,2)).masked_fill(q_mask.unsqueeze(1).expand([batch_size, max_len_c, max_len]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
            
            c2v = (vq_slice @ ck_slice.transpose(1,2) * sa[:,mv:]).masked_fill(c_mask.unsqueeze(1).expand([batch_size, num_obj, max_len_c+mc]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
            
            v2c = (cq_slice @ vk_slice.transpose(1,2) * sa.transpose(-1, -2)[:, mc:]).masked_fill(v_mask.unsqueeze(1).expand([batch_size, max_len_c, num_obj+mv]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
            
            # softmax attention
            interMAF_q2v = F.softmax(q2v, dim=2) #[batch, num_obj, max_len]
            interMAF_v2q = F.softmax(v2q, dim=2) #[batch, max_len, num_obj]
            
            interMAF_q2c = F.softmax(q2c, dim=2) #[batch, max_len_c, max_len]
            interMAF_c2q = F.softmax(c2q, dim=2) #[batch, max_len, max_len_c]
#             p = 0.1
#             sa >
            interMAF_c2v = F.softmax(c2v, dim=2) #[batch, num_obj, max_len_c]
            interMAF_v2c = F.softmax(v2c, dim=2) #[batch, max_len_c, num_obj]
            
            # calculate update input (each head of multi-head is calculated independently and concatenate together)
            q2v_update = interMAF_q2v @ qv_slice if (i==0) else torch.cat((q2v_update, interMAF_q2v @ qv_slice), dim=2)
            v2q_update = interMAF_v2q @ vv_slice if (i==0) else torch.cat((v2q_update, interMAF_v2q @ vv_slice), dim=2)
            c2v_update = interMAF_c2v @ cv_slice if (i==0) else torch.cat((c2v_update, interMAF_c2v @ cv_slice), dim=2)
            v2c_update = interMAF_v2c @ vv_slice if (i==0) else torch.cat((v2c_update, interMAF_v2c @ vv_slice), dim=2)
            c2q_update = interMAF_c2q @ cv_slice if (i==0) else torch.cat((c2q_update, interMAF_c2q @ cv_slice), dim=2)
            q2c_update = interMAF_q2c @ qv_slice if (i==0) else torch.cat((q2c_update, interMAF_q2c @ qv_slice), dim=2)
#         print("c2v_update", c2v_update)    
        # update new feature
        cat_v = torch.cat((v, q2v_update, c2v_update), dim=2)
        cat_q = torch.cat((q, v2q_update, c2q_update), dim=2)
        cat_c = torch.cat((c, v2c_update, q2c_update), dim=2)
        
        updated_v = self.v_output(self.drop(cat_v))
        updated_q = self.q_output(self.drop(cat_q))
        updated_c = self.c_output(self.drop(cat_c))
#         print("\n updated_v",updated_v, "\n updated_q", updated_q, "\n updated_c", updated_c)
        return updated_v, updated_q, updated_c

class DyIntraModalityUpdate(nn.Module):
    """
    Dynamic Intra-modality Attention Flow
    """
    def __init__(self, v_size, q_size, c_size , output_size, num_head, drop=0.0):
        super(DyIntraModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.c_size = c_size
        
        self.output_size = output_size
        self.num_head = num_head

        self.v4q_gate_lin = nn.Linear(v_size, output_size)
        self.q4v_gate_lin = nn.Linear(q_size, output_size)

        self.v_lin = nn.Linear(v_size, output_size * 3)
        self.q_lin = nn.Linear(q_size, output_size * 3)
        self.c_lin = nn.Linear(c_size, output_size * 3)
        
        # memory **
        mv = 1
        self.m_v_k = nn.Parameter(torch.FloatTensor(1, mv, output_size))  #bs, m, 
        self.m_v_v = nn.Parameter(torch.FloatTensor(1, mv, output_size))
#         self.m_v_b = nn.Parameter(torch.FloatTensor(1, mv, 4))
        mc = 1
        self.m_c_k = nn.Parameter(torch.FloatTensor(1, mc, output_size))  #bs, m, 
        self.m_c_v = nn.Parameter(torch.FloatTensor(1, mc, output_size))
#       self.m_c_b = nn.Parameter(torch.FloatTensor(1, mc, 4))
#       memory end **
        
        self.v_output = nn.Linear(output_size, output_size)
        self.q_output = nn.Linear(output_size, output_size)
        self.c_output = nn.Linear(output_size, output_size)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(drop)
        
    def forward(self, v, q, c, v_mask, q_mask, c_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        batch_size, num_obj = v_mask.shape
        _         , max_len = q_mask.shape
        _         , max_len_c = c_mask.shape
        
        # conditioned gating vector
        v_mean = (v * v_mask.unsqueeze(2)).sum(1) / v_mask.sum(1).unsqueeze(1)
        q_mean = (q * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1).unsqueeze(1)
#         q_mean = (q * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1).unsqueeze(1)

        v4q_gate = self.sigmoid(self.v4q_gate_lin(self.drop(self.relu(v_mean)))).unsqueeze(1) #[batch, 1, feat_size]
        q4v_gate = self.sigmoid(self.q4v_gate_lin(self.drop(self.relu(q_mean)))).unsqueeze(1) #[batch, 1, feat_size]

        # key, query, value
        v_trans = self.v_lin(self.drop(self.relu(v)))
        q_trans = self.q_lin(self.drop(self.relu(q)))
        c_trans = self.c_lin(self.drop(self.relu(c)))

        # mask all padding object/word features
        v_trans = v_trans * v_mask.unsqueeze(2)
        q_trans = q_trans * q_mask.unsqueeze(2)
        c_trans = c_trans * c_mask.unsqueeze(2)
        
        # split for different use of purpose
        v_k, v_q, v_v = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        q_k, q_q, q_v = torch.split(q_trans, q_trans.size(2) // 3, dim=2)
        c_k, c_q, c_v = torch.split(c_trans, c_trans.size(2) // 3, dim=2)
        
        # memory concat *
        mv = self.m_v_k.size(1)
        m_v_k = np.sqrt(self.m_v_k.size(2)) * self.m_v_k.expand(batch_size, self.m_v_k.size(1), self.m_v_k.size(2))
        m_v_v = np.sqrt(self.m_v_v.size(1)) * self.m_v_v.expand(batch_size, self.m_v_v.size(1), self.m_v_v.size(2))
#         m_v_b = self.m_v_b.expand(batch_size, self.m_v_b.size(1), self.m_v_b.size(2))
        v_k = torch.cat([m_v_k, v_k],dim=1)
        v_v = torch.cat([m_v_v, v_v],dim=1)
#         vbbox = torch.cat([m_v_b, vbbox],dim=1)
        # compute mask
        v_mask = torch.cat([torch.ones(batch_size, mv).cuda(v_mask.get_device()), v_mask], dim=1)
        
        mc = self.m_c_k.size(1)
        m_c_k = np.sqrt(self.m_c_k.size(2)) * self.m_c_k.expand(batch_size, self.m_c_k.size(1), self.m_c_k.size(2))
        m_c_v = np.sqrt(self.m_c_v.size(1)) * self.m_c_v.expand(batch_size, self.m_c_v.size(1), self.m_c_v.size(2))
#         m_c_b = self.m_c_b.expand(batch_size, self.m_c_b.size(1), self.m_c_b.size(2))
        c_k = torch.cat([m_c_k, c_k], dim=1)
        c_v = torch.cat([m_c_v, c_v], dim=1)
#         cbbox = torch.cat([m_c_b, cbbox], dim=1)
        # compute mask
        c_mask = torch.cat([torch.ones(batch_size, mc).cuda(c_mask.get_device()),c_mask], dim=1)
        # memory end *
        
        # apply conditioned gate
        new_vq = (1 + q4v_gate) * v_q
        new_vk = (1 + q4v_gate) * v_k
        
        new_qq = (1 + v4q_gate) * q_q
        new_qk = (1 + v4q_gate) * q_k
        
        # apply multi-head
        vk_set = torch.split(new_vk, new_vk.size(2) // self.num_head, dim=2)
        vq_set = torch.split(new_vq, new_vq.size(2) // self.num_head, dim=2)
        vv_set = torch.split(v_v, v_v.size(2) // self.num_head, dim=2)
        
        qk_set = torch.split(new_qk, new_qk.size(2) // self.num_head, dim=2)
        qq_set = torch.split(new_qq, new_qq.size(2) // self.num_head, dim=2)
        qv_set = torch.split(q_v, q_v.size(2) // self.num_head, dim=2)
        
        ck_set = torch.split(c_k, c_k.size(2) // self.num_head, dim=2)
        cq_set = torch.split(c_q, c_q.size(2) // self.num_head, dim=2)
        cv_set = torch.split(c_v, c_v.size(2) // self.num_head, dim=2)
        
        # multi-head
        for i in range(self.num_head):
            vk_slice, vq_slice, vv_slice = vk_set[i], vq_set[i], vv_set[i]  #[batch, num_obj, feat_size]
            qk_slice, qq_slice, qv_slice = qk_set[i], qq_set[i], qv_set[i]  #[batch, max_len, feat_size]
            ck_slice, cq_slice, cv_slice = ck_set[i], cq_set[i], cv_set[i]  #[batch, max_len, feat_size]

            # calculate attention
            v2v = (vq_slice @ vk_slice.transpose(1,2)).masked_fill(v_mask.unsqueeze(1).expand([batch_size, num_obj, num_obj+mv]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
            
            q2q = (qq_slice @ qk_slice.transpose(1,2)).masked_fill(q_mask.unsqueeze(1).expand([batch_size, max_len, max_len]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
            
            c2c = (cq_slice @ ck_slice.transpose(1,2)).masked_fill(c_mask.unsqueeze(1).expand([batch_size, max_len_c, max_len_c+mc]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
            
            dyIntraMAF_v2v = F.softmax(v2v, dim=2)
            dyIntraMAF_q2q = F.softmax(q2q, dim=2)
            dyIntraMAF_c2c = F.softmax(c2c, dim=2)
            
            # calculate update input
            v_update = dyIntraMAF_v2v @ vv_slice if (i==0) else torch.cat((v_update, dyIntraMAF_v2v @ vv_slice), dim=2)
            q_update = dyIntraMAF_q2q @ qv_slice if (i==0) else torch.cat((q_update, dyIntraMAF_q2q @ qv_slice), dim=2)
            c_update = dyIntraMAF_c2c @ cv_slice if (i==0) else torch.cat((c_update, dyIntraMAF_c2c @ cv_slice), dim=2)
        
        # update
        updated_v = self.v_output(self.drop(v + v_update))
        updated_q = self.q_output(self.drop(q + q_update))
        updated_c = self.c_output(self.drop(c + c_update))
        
        return updated_v, updated_q, updated_c
        
class Spatial_attention(nn.Module):
    def __init__(self, hidden_dim = 512, drop = 0.):
        super(Spatial_attention, self).__init__()
        
        layes = [
            nn.Linear(17 , hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim , 1),
            nn.Sigmoid()
        ]
        
        self.anet = nn.Sequential(*layes)
    
    def forward(self, b1, b2): 
        '''
           b1, n, 4
           b2, m, 4
        '''
        
        bsize, b1n, d = b1.size()
        _ , b2n, _ = b2.size()
        iou = self.iou(b1.transpose(-1, -2) ,b2.transpose(-1, -2))
        b1 = torch.cat((b1, self.get_wh(b1), self.get_center(b1)),dim=2)
        b2 = torch.cat((b2, self.get_wh(b2), self.get_center(b2)),dim=2)
        d += 4
        b1 = b1.unsqueeze(2).expand(bsize, b1n, b2n, d)
        b1 = b1.contiguous()
        b2 = b2.unsqueeze(1).expand(bsize, b1n, b2n, d)
        b2 = b2.contiguous()
        s_i = torch.cat((b1, b2, iou.unsqueeze(dim=3)), dim=3)
        at = self.anet(s_i)
        return at.squeeze(-1) + 1
        
    def iou(self, a, b):
        # this is just the usual way to IoU from bounding boxes
        inter = self.intersection(a, b)
        area_a = self.area(a).unsqueeze(2).expand_as(inter)
        area_b = self.area(b).unsqueeze(1).expand_as(inter)
        return inter / (area_a + area_b - inter + 1e-12)
        
    def area(self, box):
        x = (box[:, 2, :] - box[:, 0, :]).clamp(min=0)
        y = (box[:, 3, :] - box[:, 1, :]).clamp(min=0)
        return x * y
        
    def intersection(self, a, b):
        size = (a.size(0), 2, a.size(2), b.size(2))

        min_point = torch.max(
            a[:, :2, :].unsqueeze(dim=3).expand(*size),
            b[:, :2, :].unsqueeze(dim=2).expand(*size),
        )

        max_point = torch.min(
            a[:, 2:, :].unsqueeze(dim=3).expand(*size),
            b[:, 2:, :].unsqueeze(dim=2).expand(*size),
        )
        
        inter = (max_point - min_point).clamp(min=0)
        
        area = inter[:, 0, :, :] * inter[:, 1, :, :]
        return area
    
    def get_center(self, b1):
        
        cx = (b1[:,:,0]+b1[:,:,2])/2
        cy = (b1[:,:,1]+b1[:,:,3])/2
        
        return torch.cat((cx.unsqueeze(dim=2), cy.unsqueeze(dim=2)), dim=2)
    
    def get_wh(self, b1):
        
        w = b1[:,:,2] - b1[:,:,0]
        h = b1[:,:,3] - b1[:,:,1]
        
        return torch.cat((w.unsqueeze(dim=2), h.unsqueeze(dim=2)),dim=2)