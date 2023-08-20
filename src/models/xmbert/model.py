from collections import OrderedDict
from dataclasses import dataclass
import logging
import math
from typing import Tuple, Union, Callable, Optional

from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from transformers import AutoModel,BertConfig,AutoTokenizer


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
    

if __name__ == "__main__":
    model = CLP_clinical(bert_model_name="emilyalsentzer/Bio_ClinicalBERT")
    checkpoint_path = './epoch_50.pt'
    checkpoint = torch.load(checkpoint_path,map_location='cpu')
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    print('Loading success')
    # model = CLP_clinical(embed_dim=768)
    img_caption = "A pleural disease that is characterized as an abnormal collection of air in the pleural space between the lung and the chest wall" #pneumothorax
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",local_files_only=True)
    text = tokenizer(img_caption,add_special_tokens=True,max_length=128,pad_to_max_length=True,return_tensors='pt')
    text_embed = model.encode_text(text)
    print(text_embed.shape)
    #/mnt/petrelfs/zhangxiaoman/MODEL/2022_CLIP/A3_CLP/logs_128_atten/clp_half_freeze_clinical_bert