import torch
import torch.nn as nn
import yaml
from layers import SimpleClassifier, WordEmbedding, VisualFeatEncoder, QuestionEmbedding
import torch.nn.functional as F
from dfaf import SingleBlock, Classifier, Ocr_classifier, MultiBlock
from torch.nn.utils.weight_norm import weight_norm
from transformers import BertModel

class LoRRA(nn.Module):
    def __init__(self, 
                 image_encoder,
                 interaction_net, 
                 classifier, 
                 **arg):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        del self.bert.encoder.layer[3:]
        self.image_encoder = image_encoder
        
        self.interaction_net = interaction_net
        self.classifier = classifier
        
        for m in self.bert.parameters():
            m.requires_grad = False
        
#         for m in self.image_encoder.parameters():
#             m.requires_grad = False
        
    def forward(self, image, bbox, input_ids, token_type_ids, attention_mask, context, cbbox):
        
        '''
        image:     (b, N, 2048)
        bbox:      (b, N, 4)
        text :     (b, T)
        contex :   (b, M, 300)
        '''
        
        text_encode, _ = self.bert(input_ids=input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)
        image_encode = self.image_encoder(image, bbox)
        
        '''
        obtain the mask of image , question and context feature
        '''
        
        text_mask = attention_mask.float()
        context_mask = (self.get_mask(context) == False).float()
        image_mask = (self.get_mask(image_encode) == False).float()
        
        mlevelr = self.interaction_net(image_encode, text_encode, context, image_mask, text_mask, context_mask, bbox, cbbox)
        score = []
        l = len(mlevelr)
        for i in range(l):
            v = mlevelr[i][0]
            q = mlevelr[i][1]
            c = mlevelr[i][2]
            score1 = self.classifier[0](v, q, image_mask, text_mask)
            score2 = self.classifier[1](c, q, text_mask).squeeze(dim=2)
            total_score = torch.cat((score1,score2), dim = 1)
            score.append(total_score.unsqueeze(dim=1))
        
        score = torch.cat(score, dim=1)
        return score.sum(dim=1) / l
        
    def get_mask(self, x):
        return (x.abs().sum(dim=2) == 0)
    
def build_model(dataset, config):
    
    image_encoder = VisualFeatEncoder(
            in_dim=config['image_feature_dim'],
            **config['image_feature_encodings']
    )
    
    interaction_net = MultiBlock(**config['interIntrablocks'])
    config["classifier"]["out_features"] = 1
    classifier2 = Ocr_classifier(**config["classifier"])
    config["classifier"]["out_features"] = dataset.answer_process.length
    classifier1 = Classifier(**config["classifier"])
    classifier = nn.ModuleList([classifier1, classifier2])
    
    modules = {
        'image_encoder': image_encoder,
        "interaction_net": interaction_net,
        'classifier' : classifier
    }
    
    return LoRRA(**modules)