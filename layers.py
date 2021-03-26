import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.init as init
from utils import BertLayerNorm

import os
import _pickle as pickle
import numpy as np

from fusion import BlFusion, BlFusion2d
from fc import FCNet

class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, emb_dim, dropout= 0.0, op=''):

        super(WordEmbedding, self).__init__()
        self.op = op
        self.emb = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
        if 'c' in op:
            self.emb_ = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
            self.emb_.weight.requires_grad = False # fixed
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file,tfidf=None, tfidf_weights=None):

        weight_init = torch.from_numpy(np_file)
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init
        
        if tfidf is not None:
            
            if 0 < tfidf_weights.size:
                weight_init = torch.cat([weight_init, torch.from_numpy(tfidf_weights)], 0)
            weight_init = tfidf.matmul(weight_init) # (N x N') x (N', F)
            self.emb_.weight.requires_grad = True

        if 'c' in self.op:
            self.emb_.weight.data[:self.ntoken] = weight_init.clone()

    def forward(self, x):
        emb = self.emb(x)
        if 'c' in self.op:   # 600
            emb = torch.cat((emb, self.emb_(x)), 2)
        emb = self.dropout(emb)
        return emb

class QuestionEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, nlayers, bidirect, dropout, rnn_type='GRU', residual=False, **arg):
        """  Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        
        self.k = 1
        if bidirect:
            self.k = 2
        self.rnn_dim = (hidden_dim - (0 if not residual else embedding_dim))//self.k
        self.residual = residual
        
        self.rnn = rnn_cls(
            embedding_dim, self.rnn_dim, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = embedding_dim
        self.num_hid = hidden_dim
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)
        
#         self._init_lstm()
        
    def _init_lstm(self):
        print("----kaiming_init----")
        for w in self.rnn.weight_ih_l0.chunk(3,0):
            torch.nn.init.kaiming_normal(w)
        for w in self.rnn.weight_hh_l0.chunk(3,0):
            torch.nn.init.kaiming_normal(w)
        
        self.rnn.bias_hh_l0.data.zero_()
        self.rnn.bias_hh_l0.data.zero_()

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.rnn_dim)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        lengths = 14-(0==x.abs().sum(dim=2)).long().sum(1)
        
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        
        lens, indices = torch.sort(lengths, 0, True)
        lens = lens.data.tolist()
        _, _indices = torch.sort(indices,0)
        
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(pack(x[indices], lens, batch_first=True), hidden)
        output = unpack(output, batch_first=True)[0][_indices]
        hidden = (hidden[0].transpose(0, 1).contiguous().view(batch, -1))[_indices]
        
        forward_ = []
        if self.ndirections == 1:
            for i in range(output.size(0)):
                forward_.append(output[i][lengths[i].data-1])
            q_embeding = torch.stack(forward_,dim=1).squeeze() # b, n, dim 
        else:
            for i in range(output.size(0)):
                forward_.append(output[i][lengths[i].data-1][:,:self.rnn_dim])
            forward_ = torch.stack(forward_,dim=1).squeeze()
            backward = output[:, 0, self.rnn_dim:]
            q_embeding = torch.cat((forward_, backward), dim=1)

        return q_embeding, hidden
    
    def forward_all(self, x):
        
        # x: [batch, sequence, in_dim]
        lengths = 14 - (0==x.abs().sum(dim=2)).long().sum(1)
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        
        lens, indices = torch.sort(lengths, 0, True)
        lens = lens.data.tolist()
        _, _indices = torch.sort(indices,0)

        self.rnn.flatten_parameters()
        output, hidden = self.rnn(pack(x[indices], lens, batch_first=True), hidden)
        output = unpack(output, batch_first=True)[0][_indices]
        hidden = (hidden[0].transpose(0, 1).contiguous().view(batch, -1))[_indices]
        
        if self.residual:
#             pad = Variable(torch.zeros(x.size(1) - output.size(1), output.size(2))).cuda().unsqueeze(0).repeat(output.size(0),1,1)
            output = torch.cat([F.pad(output,pad=(0,0,0,x.size(1) - output.size(1))), x], dim=2)

        return output, hidden
        
class TransformLayer(nn.Module):
    def __init__(self, transform_type, in_dim, out_dim, hidden_dim=None):
        super(TransformLayer, self).__init__()

        if transform_type == "linear":
            self.module = LinearTransform(in_dim, out_dim)
        elif transform_type == "conv":
            self.module = ConvTransform(in_dim, out_dim, hidden_dim)
        else:
            raise NotImplementedError(
                "Unknown post combine transform type: %s" % transform_type
            )
        self.out_dim = self.module.out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

class LinearTransform(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearTransform, self).__init__()
        self.lc = weight_norm(
            nn.Linear(in_features=in_dim, out_features=out_dim), dim=None
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.lc(x)

class ConvTransform(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(ConvTransform, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=hidden_dim, kernel_size=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=out_dim, kernel_size=1
        )
        self.out_dim = out_dim

    def forward(self, x):
        if len(x.size()) == 3:  # N x k xdim
            # N x dim x k x 1
            x_reshape = torch.unsqueeze(x.permute(0, 2, 1), 3)
        elif len(x.size()) == 2:  # N x dim
            # N x dim x 1 x 1
            x_reshape = torch.unsqueeze(torch.unsqueeze(x, 2), 3)

        iatt_conv1 = self.conv1(x_reshape)  # N x hidden_dim x * x 1
        iatt_relu = nn.functional.relu(iatt_conv1)
        iatt_conv2 = self.conv2(iatt_relu)  # N x out_dim x * x 1

        if len(x.size()) == 3:
            iatt_conv3 = torch.squeeze(iatt_conv2, 3).permute(0, 2, 1)
        elif len(x.size()) == 2:
            iatt_conv3 = torch.squeeze(torch.squeeze(iatt_conv2, 3), 2)
        return iatt_conv3

class NonLinearElementMultiply(nn.Module):
    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super(NonLinearElementMultiply, self).__init__()
        self.fa_image = ReLUWithWeightNormFC(image_feat_dim, kwargs["hidden_dim"])
        self.fa_txt = ReLUWithWeightNormFC(ques_emb_dim, kwargs["hidden_dim"])

        context_dim = kwargs.get("context_dim", None)
        if context_dim is None:
            context_dim = ques_emb_dim

        self.fa_context = ReLUWithWeightNormFC(context_dim, kwargs["hidden_dim"])
        self.dropout = nn.Dropout(kwargs["dropout"])
        self.out_dim = kwargs["hidden_dim"]

    def forward(self, image_feat, question_embedding, context_embedding=None):
        image_fa = self.fa_image(image_feat)
        question_fa = self.fa_txt(question_embedding)

        if len(image_feat.size()) == 3:
            question_fa_expand = question_fa.unsqueeze(1)
        else:
            question_fa_expand = question_fa
        
        # 联合表征 N, K, D
        joint_feature = image_fa * question_fa_expand
        
        if context_embedding is not None:
            context_fa = self.fa_context(context_embedding)
            
            context_text_joint_feaure = context_fa * question_fa_expand
            joint_feature = torch.cat([joint_feature, context_text_joint_feaure], dim=1)

        joint_feature = self.dropout(joint_feature)
        
        return joint_feature

# TODO: Do clean implementation without Sequential
class ReLUWithWeightNormFC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ReLUWithWeightNormFC, self).__init__()

        layers = []
        layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class LogitClassifier(nn.Module): 
    def __init__(self, in_dim, out_dim, **kwargs):
        super(LogitClassifier, self).__init__()
        input_dim = in_dim
        num_ans_candidates = out_dim
        text_non_linear_dim = kwargs["text_hidden_dim"] 
        image_non_linear_dim = kwargs["img_hidden_dim"]

        self.f_o_text = ReLUWithWeightNormFC(input_dim, text_non_linear_dim)
        self.f_o_image = ReLUWithWeightNormFC(input_dim, image_non_linear_dim)
        self.linear_text = nn.Linear(text_non_linear_dim, num_ans_candidates)
        self.linear_image = nn.Linear(image_non_linear_dim, num_ans_candidates)

        if "pretrained_image" in kwargs and kwargs["pretrained_text"] is not None:
            self.linear_text.weight.data.copy_(
                torch.from_numpy(kwargs["pretrained_text"])
            )

        if "pretrained_image" in kwargs and kwargs["pretrained_image"] is not None:
            self.linear_image.weight.data.copy_(
                torch.from_numpy(kwargs["pretrained_image"])
            )

    def forward(self, joint_embedding):
        text_val = self.linear_text(self.f_o_text(joint_embedding))
        image_val = self.linear_image(self.f_o_image(joint_embedding))
        logit_value = text_val + image_val

        return logit_value

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits

class VisualFeatEncoder(nn.Module):
    def __init__(self, in_dim, weights_file, bias_file, hidden_size, model_data_dir='../data/'):
        super(VisualFeatEncoder, self).__init__()
        
        self.pos_dim = 4
        self.hidden_size = hidden_size
        if not os.path.isabs(weights_file):
            weights_file = os.path.join(model_data_dir, weights_file)
        if not os.path.isabs(bias_file):
            bias_file = os.path.join(model_data_dir, bias_file)
        with open(weights_file, "rb") as w:
            weights = pickle.load(w)
        with open(bias_file, "rb") as b:
            bias = pickle.load(b)
        out_dim = bias.shape[0]
        
        self.lc = nn.Linear(in_dim, out_dim)
        self.lc.weight.data.copy_(torch.from_numpy(weights))
        self.lc.bias.data.copy_(torch.from_numpy(bias))
        self.out_dim = out_dim
#         for m in self.lc.parameters():
#             m.requires_grad = False
        
    def forward(self, image, bb):
        i2 = self.lc(image)
        i3 = nn.functional.relu(i2)
        return i3

# Image_text_context_concat_fusion
class ITCFusion(nn.Module):
    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super(ITCFusion, self).__init__()
        context_dim = kwargs.get("context_dim")
        h_dim = kwargs["hidden_dim"]
        self.k = kwargs["k"]
        self.image_text_combine = BlFusion(
            v_dim = image_feat_dim, 
            q_dim= ques_emb_dim, 
            h_dim = h_dim, 
            k=self.k
        )
        
        self.text_context_combine = BlFusion2d(
            v_dim = ques_emb_dim, 
            q_dim= context_dim, 
            h_dim = h_dim, 
            k=self.k
        )
        
        self.dropout = nn.Dropout(kwargs["dropout"])
        self.out_dim = kwargs["hidden_dim"]

    def forward(self, image_feat, question_embedding, context_embedding, question_embedding1):
        
        itfeature = self.image_text_combine(image_feat, question_embedding)
        question_repeat = question_embedding1.unsqueeze(dim=1).repeat(1,context_embedding.size(1),1)
#         print(question_repeat.shape, context_embedding.shape)
        tcfeature = self.text_context_combine(question_repeat, context_embedding)
        return itfeature, tcfeature  #, joint_feature


# class Image_text_context_concat_fusion(nn.Module):

#     def __init__(self, **kwargs):
#         super(Image_text_context_concat_fusion, self).__init__()
#         image_feat_dim = kwargs.get("image_dim")
#         ques_emb_dim = kwargs.get("text_dim")
#         context_dim = kwargs.get("context_dim")
#         h_dim = kwargs["hidden_dim"]
#         dropout = kwargs["dropout"]
#         self.k = kwargs["k"]
#         self.image_feat_dim = image_feat_dim
#         self.ques_emb_dim = ques_emb_dim
#         self.context_dim = context_dim
#         self.v_net = FCNet([image_feat_dim,h_dim*self.k], dropout=dropout[0])
#         self.q_net = FCNet([ques_emb_dim,h_dim*self.k], dropout=dropout[0])
#         self.c_net = FCNet([context_dim,h_dim*self.k], dropout=dropout[0])
#         self.dropout = nn.Dropout(dropout[1])
#         if self.k>1:
#             self.p_net = nn.AvgPool1d(self.k,stride = self.k)
#             self.p_net1 = nn.AvgPool1d(self.k,stride = self.k)

#     def forward(self, v, q, c):
        
#         v_emb = self.dropout(self.v_net(v))
# #         v_emb = self.v_net(v)
#         c_emb = self.dropout(self.c_net(c))

#         q_emb = self.q_net(q)
        
#         logits1 = v_emb*q_emb
#         logits2 = c_emb*q_emb
#         if self.k>1:
#             logits1 = logits1.unsqueeze(dim=1)
#             logits1 = self.p_net(logits1).squeeze(dim=1)*self.k
#             logits2 = logits2.unsqueeze(dim=1)
#             logits2 = self.p_net1(logits2).squeeze(dim=1)*self.k
#         logits = torch.cat([logits1, logits2], dim=1)
#         return logits