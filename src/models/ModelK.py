# modified from https://github.com/tensorflow/models/blob/master/research/slim/nets/s3dg.py
from sklearn.metrics import log_loss
import torch.nn as nn
import torch
import math
import numpy as np  
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .transformer import *
import torchvision.models as models
from einops import rearrange
from transformers import AutoModel,BertConfig,AutoTokenizer
from typing import Tuple, Union, Callable, Optional
import random
'''
args.N
args.d_model
args.res_base_model
args.H 
args.num_queries
args.dropout
args.attribute_set_size
'''

class CLP_clinical(nn.Module):
    def __init__(self,
                bert_model_name: str,
                embed_dim: int = 768,
                freeze_layers:Union[Tuple[int, int], int] = None):
        super().__init__()
        self.bert_model = self._get_bert_basemodel(bert_model_name=bert_model_name, freeze_layers=freeze_layers)
        self.mlp_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.embed_dim = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        for m in self.mlp_embed:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.embed_dim ** -0.5)

    def _get_bert_basemodel(self, bert_model_name, freeze_layers=None):#12
        try:
            print(bert_model_name)
            config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)#bert-base-uncased
            model = AutoModel.from_pretrained(bert_model_name, config=config)#, return_dict=True)
            print("Text feature extractor:", bert_model_name)
            print("bert encoder layers:",len(model.encoder.layer))
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model
    
    def encode_text(self, text):
        output = self.bert_model(input_ids = text['input_ids'],attention_mask = text['attention_mask'] )
        last_hidden_state, pooler_output, hidden_states = output[0],output[1],output[2]
        encode_out = self.mlp_embed(pooler_output)
        return encode_out
    
    def forward(self,text1,text2):
        text1_features = self.encode_text(text1)
        text2_features = self.encode_text(text2)
        text1_features = F.normalize(text1_features, dim=-1)
        text2_features = F.normalize(text2_features, dim=-1)
        return text1_features, text2_features, self.logit_scale.exp()


class DQN(nn.Module):

    def __init__(self, config, disease_book, mode='train'):
        super(DQN, self).__init__()
        self.device = disease_book['input_ids'].device
        self.mode = mode
        self.d_model = config['d_model']
        # ''' book embedding'''
        with torch.no_grad():
            bert_model = CLP_clinical(bert_model_name="emilyalsentzer/Bio_ClinicalBERT")
            checkpoint_path = '/mnt/petrelfs/wuchaoyi/class-to-text/TQN/models/xmbert/epoch_100.pt'
            checkpoint = torch.load(checkpoint_path,map_location='cpu')
            state_dict = checkpoint["state_dict"]
            bert_model.load_state_dict(state_dict)
            bert_model.to(disease_book['input_ids'].device)
            #print(disease_book['input_ids'].device)
            self.disease_book = bert_model.encode_text(disease_book)

        ## prompt 'PROMPT GENERATION NETWORKS FOR EFFICIENT ADAPTATION OF FROZEN VISION TRANSFORMERS' ##
        tl_vectors = torch.empty(
            (config['Tokens'],256),
            device=self.device,
        )
        torch.nn.init.normal_(tl_vectors, std=0.02)
        self.tl_vectors = torch.nn.Parameter(tl_vectors)
        self.disease_prompt = nn.Linear(768,config['Tokens'])
        
        #self.disease_embedding_layer = nn.Linear(768,256)
        self.cl_fc = nn.Linear(256,768)

        ''' visual backbone'''
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}
        resnet = self._get_res_basemodel(config['res_base_model'])
        num_ftrs = int(resnet.fc.in_features/2)
        self.res_features = nn.Sequential(*list(resnet.children())[:-3])
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, self.d_model)


        ###################################
        ''' Query Decoder'''
        ###################################

        self.H = config['H'] 
        decoder_layer = TransformerDecoderLayer(self.d_model, config['H'] , 1024,
                                        0.1, 'relu',normalize_before=True)
        decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, config['N'] , decoder_norm,
                                  return_intermediate=False)

        self.dropout_feas = nn.Dropout(config['dropout'] )

        # Attribute classifier
        self.classifier = nn.Linear(self.d_model,config['attribute_set_size'])

        self.apply(self._init_weights)
        

    def _get_res_basemodel(self, res_model_name):
        try:
            res_model = self.resnet_dict[res_model_name]
            print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        try:
            model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
            print("text feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model
    
    def image_encoder(self, xis):
        #patch features
        """
        16 torch.Size([16, 1024, 14, 14])
        torch.Size([16, 196, 1024])
        torch.Size([3136, 1024])
        torch.Size([16, 196, 256])
        """
        batch_size = xis.shape[0]
        res_fea = self.res_features(xis) #batch_size,feature_size,patch_num,patch_num
        res_fea = rearrange(res_fea,'b d n1 n2 -> b (n1 n2) d')
        h = rearrange(res_fea,'b n d -> (b n) d')
        #batch_size,num,feature_size
        # h = h.squeeze()
        x = self.res_l1(h)
        x = F.relu(x)
        
        
        x = self.res_l2(x)
        out_emb = rearrange(x,'(b n) d -> b n d',b=batch_size)
        return out_emb

    def forward(self, images, labels, is_train = True):
        
        B = images.shape[0]
        ''' Visual Backbone '''
        x = self.image_encoder(images) #batch_size,patch_num,dim

        features = x.transpose(0,1) #patch_num b dim
        #print(self.disease_book.device)
        Token_chosen_embedding = F.softmax(self.disease_prompt(self.disease_book),dim = -1) 
        query_embed = torch.einsum(
            'qm,mv->qv',
            [Token_chosen_embedding, self.tl_vectors]
        )
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)
        features,ws = self.decoder(query_embed, features, 
            memory_key_padding_mask=None, pos=None, query_pos=None)
        
        
        out = self.dropout_feas(features)
        x= self.classifier(out).transpose(0,1) #B query Atributes
         
        labels = labels.reshape(-1,1)
        logits = x.reshape(-1, x.shape[-1])
        
        Mask = ((labels != -1) & (labels != 2)).squeeze()
        if is_train == True:
            labels = labels[Mask].long()
            logits = logits[Mask]
            loss_ce = F.cross_entropy(logits,labels[:,0])
        else:
            loss_ce = torch.tensor(0)
        if is_train==True:
            return loss_ce
        else:
            return x,ws
        
    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()