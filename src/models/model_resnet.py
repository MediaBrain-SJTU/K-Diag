# modified from https://github.com/tensorflow/models/blob/master/research/slim/nets/s3dg.py
import torch.nn as nn
import torch
import torch.nn.functional as F
from .transformer import *
import torchvision.models as models



class RESNET_base(nn.Module):

    def __init__(self, config, disease_book, mode='train'):
        super(RESNET_base, self).__init__()

        self.mode = mode
        self.d_model = config['d_model']

        ''' visual backbone'''
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}
        resnet = self._get_res_basemodel(config['res_base_model'])
        #num_ftrs = int(resnet.fc.in_features/2)
        num_ftrs = int(resnet.fc.in_features)
        self.res_features = nn.Sequential(*list(resnet.children())[:-1])
        self.res_out = nn.Linear(num_ftrs, len(disease_book))

    def _get_res_basemodel(self, res_model_name):
        try:
            res_model = self.resnet_dict[res_model_name]
            print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, images, labels, is_train = True):
        
        ''' Visual Backbone '''
        x = self.res_features(images)
        x = x.squeeze()
        x = self.res_out(x)
         
        labels = labels.reshape(-1,1)
        logits = x.reshape(-1, 1)
        
        Mask = ((labels != -1) & (labels != 2)).squeeze()
        if is_train == True:
            labels = labels[Mask].float()
            logits = logits[Mask].float()
            loss_ce = F.binary_cross_entropy_with_logits(logits[:,0],labels[:,0])
        else:
            loss_ce = torch.tensor(0)
        if is_train==True:
            return loss_ce
        else:
            return x
        