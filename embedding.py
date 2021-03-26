import torch
import torch.nn as nn
from layers import NonLinearElementMultiply, TransformLayer
from fusion import BlFusion2d
from fc import FCNet
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
from fusion import BlFusion, BlFusion2d

# Image attention guided by question 
class IAGQ(nn.Module):
    """
    parameters:

    input:
    image_feat_variable: [batch_size, num_location, image_feat_dim]
    or a list of [num_location, image_feat_dim]
    when using adaptive number of objects
    question_embedding:[batch_size, txt_embeding_dim]

    output:
    image_embedding:[batch_size, image_feat_dim]
    
    """
    
    def __init__(self, img_dim, question_dim, **kwargs):
        super(IAGQ, self).__init__()
        
        self.glimpse = kwargs["glimpse"]
        self.h_dim = kwargs["hidden_dim"]
        self.f_dropout = kwargs["f_dropout"]
        self.dropout = kwargs["dropout"]
        self.k = kwargs["k"]
        
        self.fusion = BlFusion2d(
            img_dim, 
            question_dim, 
            self.h_dim, 
            dropout = self.f_dropout, 
            k=self.k
        )

        layers = [
            nn.Dropout(self.dropout, inplace=False),
            weight_norm(nn.Linear(self.h_dim, self.glimpse), dim=None)
        ]
        
#         self.proj = nn.Linear(self.glimpse * img_dim, img_dim)
        
        self.attention = nn.Sequential(*layers)

    def forward(self, image_feat_variable, question_embedding, image_mask):
        question_repeat = question_embedding.unsqueeze(dim=1).repeat(1,image_feat_variable.size(1),1)
        qv_fusion_feature = self.fusion(image_feat_variable, question_repeat)
        logist = self.attention(qv_fusion_feature) # b, n, g
        
        logist.data.masked_fill_(image_mask.unsqueeze(-1).expand_as(logist).data,-float("inf"))
        
        attention = torch.softmax(logist, dim = 1) 
        
        '''
        attention :  b, n, g
        image_feat_variable : b, n, dim
        '''
        image_embedding = torch.bmm( attention.transpose(-1,-2), image_feat_variable)
        image_embedding = image_embedding.view(image_embedding.size(0),-1)
#         image_embedding = self.proj(image_embedding)
        
#         print(image_embedding.shape)
        return image_embedding, attention

# Context attention guided by question
class CAGQ(nn.Module):
    
    def __init__(self, context_dim, question_dim, h_dim, glimpse, f_dropout, dropout, k, **kwargs):
        super(CAGQ, self).__init__()
        self.glimpse = glimpse
        self.fusion = BlFusion2d(context_dim, question_dim, h_dim, dropout = f_dropout, k=k)
        
        layers = [
            nn.Dropout(dropout, inplace=False),
            weight_norm(nn.Linear(h_dim, self.glimpse), dim=None)
        ]
        
        self.attention = nn.Sequential(*layers)

    def forward(self, context_feat, question_embedding, mask, order_vectors=None):
        # N x K x n_att
        q = question_embedding.unsqueeze(dim=1).repeat(1,context_feat.size(1),1)
        f_fusion = self.fusion(context_feat, q)
        attention = self.attention(f_fusion)
        atten = torch.sigmoid(attention)
        
        atten.data.masked_fill_(mask.unsqueeze(-1).expand_as(attention).data,0)
        context_embedding = atten.unsqueeze(dim=-1)*context_feat.unsqueeze(dim=-2)
        context_embedding = context_embedding.view(context_embedding.size(0), context_embedding.size(1), -1)
        return context_embedding, atten