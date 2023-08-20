
import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from einops import rearrange
from sklearn.feature_extraction import image


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn


from tensorboardX import SummaryWriter

import utils
from scheduler import create_scheduler
from optim import create_optimizer
from dataset.dataset_whole import Merge_datasets
from models.ModelK import DQN
from models.tokenization_bert import BertTokenizer


def get_tokenizer(tokenizer,target_text):
    
    target_tokenizer = tokenizer(list(target_text), padding='max_length', truncation=True, max_length=256, return_tensors="pt")
    
    return target_tokenizer

def train(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, args, config, writer):
    model.train()  
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.update(loss_ce=1.0)
    metric_logger.update(lr = scheduler._get_lr(epoch)[0])

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 1   
    step_size = 1
    warmup_iterations = warmup_steps*step_size 
    scalar_step = epoch*len(data_loader) 

    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        images = sample['image'].to(device)
        labels = sample['label'].to(device)

        optimizer.zero_grad()

        loss_ce= model(images,labels,is_train= True)
        loss_ce.backward()
        optimizer.step()    
        writer.add_scalar('loss/loss_ce', loss_ce, scalar_step)
        scalar_step += 1
        metric_logger.update(loss_ce=loss_ce.item())
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        metric_logger.update(lr = scheduler._get_lr(epoch)[0])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} #,loss_epoch.mean()


def valid(model, data_loader, epoch, device,config,writer):
    
    model.eval()
    val_scalar_step = epoch*len(data_loader)
    #print(len(data_loader))
    val_loss = []
    for i, sample in enumerate(data_loader):
        
        images = sample['image'].to(device)
        labels = sample['label'].to(device)
        
        with torch.no_grad():
            loss_ce = model(images,labels, is_train= True)
            val_loss.append(loss_ce.item())
            writer.add_scalar('val_loss/loss_ce', loss_ce, val_scalar_step)
            val_scalar_step += 1
            #print(i)
    avg_val_loss = np.array(val_loss).mean()
    return avg_val_loss

def main(args, config):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    #### Dataset #### 
    print("Creating dataset")
    train_datasets = Merge_datasets(config['train_file'], config['mode'],mode = 'train')
    train_dataloader = DataLoader(
            train_datasets,
            batch_size=config['batch_size'],
            num_workers=16,
            pin_memory=True,
            sampler=None,
            shuffle=True,
            collate_fn=None,
            drop_last=False,
        )     
    
    val_datasets = Merge_datasets(config['valid_file'], config['mode'], mode = 'train')
    val_dataloader = DataLoader(
            val_datasets,
            batch_size=config['batch_size'],
            num_workers=16,
            pin_memory=True,
            sampler=None,
            shuffle=True,
            collate_fn=None,
            drop_last=False,
        )   

    print("Creating book")
    disease_book = train_datasets.pathologies 
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    disease_book_tokenizer = get_tokenizer(tokenizer,disease_book)
    disease_book_tokenizer['input_ids'] = disease_book_tokenizer['input_ids'].to(device)
    disease_book_tokenizer['attention_mask'] = disease_book_tokenizer['attention_mask'].to(device)
    print("Creating model")
    model = TQN(config, disease_book_tokenizer, mode = 'train')
    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model = model.to(device)  


    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer) 

    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']                      
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']+1    
        model.load_state_dict(state_dict)    
        print('load checkpoint from %s'%args.checkpoint)
        
    
    
    print("Start training")
    start_time = time.time()

    writer = SummaryWriter(os.path.join(args.output_dir,  'log'))
    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)
        #val_loss = valid(model, val_dataloader, epoch,device,config,writer)    
        train_stats = train(model, train_dataloader, optimizer, epoch, warmup_steps, device, lr_scheduler, args,config,writer) 

        for k, v in train_stats.items():
            train_loss_epoch = v
        
        writer.add_scalar('loss/train_loss_epoch', float(train_loss_epoch), epoch)
        writer.add_scalar('loss/leaning_rate',  lr_scheduler._get_lr(epoch)[0] , epoch)

        val_loss = valid(model, val_dataloader, epoch,device,config,writer)
        writer.add_scalar('loss/val_loss_epoch', val_loss, epoch)

        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'val_loss': val_loss.item()
                        }                     
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_state.pth'))  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if epoch % 20 == 1 and epoch>1:
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_'+str(epoch)+'.pth'))  

                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    Dataset_mode = 'MIMIC_CXR'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/mnt/petrelfs/wuchaoyi/class-to-text/TQN/congfig/Pretrain_'+ Dataset_mode+'_xmbert.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--output_dir', default='/mnt/petrelfs/wuchaoyi/class-to-text/ResultSSSSSs/xmbert_prompt_' + Dataset_mode +'/seed_44')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=44, type=int)
    parser.add_argument('--gpu', type=str,default='0', help='gpu')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu !='-1':
        torch.cuda.current_device()
        torch.cuda._initialized = True

    main(args, config)