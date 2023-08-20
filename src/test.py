
import argparse
import os
import cv2
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from einops import rearrange
from sklearn.feature_extraction import image
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torchvision.utils import save_image
from einops import rearrange
from tensorboardX import SummaryWriter
from torchsummary import summary
import math
import utils
from scheduler import create_scheduler
from optim import create_optimizer
from dataset.dataset_whole import Merge_datasets
from models.ModelK import DQN
from models.tokenization_bert import BertTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score
from skimage import measure
from tqdm import tqdm
whole_pathologies = ['Atelectasis','Fibrosis' ,'Pneumonia' ,'Effusion' ,'Lung Lesion',
 'Cardiomegaly' ,'Calcified Granuloma', 'Fracture' ,'Edema' ,'Granuloma',
 'Emphysema', 'Hernia', 'Mass', 'Nodule' ,'Lung Opacity', 'Infiltration',
 'Pleural Thickening', 'Pneumothorax', 'Consolidation', 'Aortic Enlargement',
 'Calcification', 'Clavicle Fracture' ,'Enlarged PA', 'ILD' ,'Lung Cavity',
 'Lung Cyst' ,'Mediastinal Shift' ,'Nodule/Mass', 'Rib Fracture', 'COPD',
 'Lung Tumor', 'Tuberculosis', 'Other Diseases', 'No Finding', 'Covid-19',
 'Enlarged Cardiomediastinum', 'Support Devices','Foreign Object']
def get_tokenizer(tokenizer,target_text):
    
    target_tokenizer = tokenizer(list(target_text), padding='max_length', truncation=True, max_length= 256, return_tensors="pt")
    
    return target_tokenizer

import numpy as np

def compute_AUCs(gt, pred, n_class):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(n_class):
        gt_temp = gt_np[:,i]
        pred_temp = pred_np[:, i]
        # valid_index = np.where(np.logical_or(gt_temp == 1,gt_temp == 0))
        # gt_temp = gt_temp[valid_index]
        # pred_temp = pred_temp[valid_index]
        try:
            AUROCs.append(roc_auc_score(gt_temp, pred_temp))
        except:
            AUROCs.append(-1)
    return AUROCs

def main(args, config,Model_mode):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    test_dataset =  Merge_datasets(config['test_file'], config['mode'],mode = 'test') 
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            num_workers=16,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
        )  
    
    disease_book = np.array(whole_pathologies)
    if 'prompt' in Model_mode:
        disease_book = whole_pathologies
    test_index = []
    for disease in test_dataset.pathologies :
        test_index.append(disease_book.tolist().index(disease))
    print(disease_book)
    print(test_index)
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    disease_book_tokenizer = get_tokenizer(tokenizer,disease_book)
    disease_book_tokenizer['input_ids'] = disease_book_tokenizer['input_ids'].to(device)
    disease_book_tokenizer['attention_mask'] = disease_book_tokenizer['attention_mask'].to(device)
    
    print("Creating model")
    model = TQN(config, disease_book_tokenizer, mode = 'test')
    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model = model.to(device)   
    

    checkpoint = torch.load(args.checkpoint, map_location='cpu') 
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)    
    print('load checkpoint from %s'%args.checkpoint)
    print("Start testing")
    model.eval()
    gt = torch.FloatTensor()
    gt = gt.to(device)
    pred = torch.FloatTensor()
    pred = pred.to(device)
    
    iter = 0
    for i, sample in enumerate(test_dataloader):
        # print(iter)
        images = sample['image'].to(device)
        labels = sample['label'].to(device)

        gt = torch.cat((gt, labels), 0)
        with torch.no_grad():
            x,ws= model(images,labels,is_train= False) #batch_size,batch_size,image_patch,text_patch
            pred_class = x
            pred_class = F.softmax(pred_class.reshape(-1,2),dim = -1).reshape(-1,len(disease_book),2)
            pred = torch.cat((pred, pred_class[:,test_index,1]), 0)
        # iter = iter +1 
    
    AUROCs = compute_AUCs(gt, pred, len(test_dataset.pathologies))
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=AUROC_avg))       
    for i in range(len(test_dataset.pathologies)):
        print('The AUROC of ' + test_dataset.pathologies[i] + ' is {}'.format(AUROCs[i]))
    return AUROC_avg        

if __name__ == '__main__':
    Dataset_mode = 'PadChest'
    Model_mode = 'whole'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./congfig/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='./Results/checkpoint.pth') 
    #parser.add_argument('--output_dir', default='/remote-home/chaoyiwu/classification_to_text/Results/xmbert_' + Dataset_mode)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', type=str,default='0', help='gpu')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    #Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    #yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu !='-1':
        torch.cuda.current_device()
        torch.cuda._initialized = True

    main(args, config,Model_mode)