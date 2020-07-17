from dataset import *

import os
import gc
import math
import time
import pytz
import torch
import yaml
import random
import argparse
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

import warnings
warnings.filterwarnings("ignore")

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

## Helper
class DotConfig:
  def __init__(self, cfg):
    self._cfg = cfg
  def __getattr__(self, k):
    v = self._cfg[k]
    if isinstance(v, dict):
      return DotConfig(v)
    return v


## Network
def get_net(img_sz):
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=True)

    config.num_classes = 1
    config.image_size = img_sz
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    
    return DetBenchTrain(net, config)

def load_net(checkpoint_path, img_sz):
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)

    config.num_classes = 1
    config.image_size=img_sz
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    checkpoint = torch.load(checkpoint_path)
    if 'model_state_dict' in checkpoint.keys():
        net.load_state_dict(checkpoint['model_state_dict']) # model 2 & 3
    else:
        net.load_state_dict(checkpoint)  # model 0 & 1

    del checkpoint
    gc.collect()

    return DetBenchTrain(net, config) 


## Engine
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Fitter:
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = config.folder
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.model = model.to(device)
        self.device = device

         # Optimizer
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                if '.bias' in k:                        pg2.append(v)  # biases
                elif '.weight' in k and '.bn' not in k: pg1.append(v)  # apply weight decay
                else:                                   pg0.append(v)  # all else

        self.optimizer = torch.optim.SGD(pg0, lr=config.lr, momentum=0.937, nesterov=True)
        self.optimizer.add_param_group({'params': pg1, 'weight_decay': 5e-4})  # add pg1 with weight_decay
        self.optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        self.lf = lambda x: (((1 + math.cos(x * math.pi / config.n_epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        
        if mixed_precision:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1', verbosity=0)

        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            output = f'[EPOCH {str(self.epoch).zfill(2)}] '
            t = time.time()
            if self.epoch == 0:
                summary_loss = self.train_one_epoch(train_loader, warmup=True)
            else:
                summary_loss = self.train_one_epoch(train_loader)
            output += f'Train: {summary_loss.avg:.5f} '

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint-{self.config.fold}.bin')

            t = time.time()
            summary_loss = self.validation(validation_loader)
            output += f'Valid: {summary_loss.avg:.5f} '

            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')

            print(output + f'\n')
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(2)}epoch-{self.config.fold}.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch-{self.config.fold}.bin'))[:-2]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step()

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc='Val')
        for step, (images, targets, image_ids) in pbar:
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                boxes = [target['boxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]

                loss, _, _ = self.model(images, boxes, labels)
                summary_loss.update(loss.detach().item(), batch_size)

            pbar.set_description(f'Valid: {summary_loss.avg:.5f}')

        return summary_loss

    def train_one_epoch(self, train_loader, warmup=False):
        self.model.train()
        accumulation_steps = self.config.accumulation_steps
        summary_loss = AverageMeter()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train')
        for i, (images, targets, image_ids) in pbar: 
            images = torch.stack(images)
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]

            # Warmup
            nw = self.config.warmup
            if i <= nw:
                xi = [0, nw]  # x interp
                for j, x in enumerate(self.optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(i, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * self.lf(self.epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(i, xi, [0.9, 0.937]) # momentum, yolov5
            
            loss, _, _ = self.model(images, boxes, labels)
            loss  = loss / accumulation_steps

            # Backward
            if mixed_precision:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else: loss.backward()
            
            # update
            if (i+1) % accumulation_steps == 0:
              summary_loss.update(loss.detach().item(), accumulation_steps)
              self.optimizer.step()
              self.optimizer.zero_grad()

            pbar.set_description(f'Train: {summary_loss.avg:.5f}')

        return summary_loss
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config.yaml', help='config.yaml path')
    parser.add_argument('--path', type=str, default='.', help='base_directory path where you have all data downloaded from kaggle')
    parser.add_argument('--train' , type=str, default='data/train.csv', help='train.csv path')
    parser.add_argument('--folds' , type=str, default='train_folds.csv', help='folds.csv path')
    parser.add_argument('--weights', type=str, help='checkpoint.pt path')
    opt = parser.parse_args()

    ## CONFIG
    with open(opt.cfg, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    config = DotConfig(config)
    
    ## DATA
    df_folds = pd.read_csv(opt.folds)
    markings = pd.read_csv(opt.train)
    bboxs = np.stack(markings['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        markings[column] = bboxs[:,i]
    markings.drop(columns=['bbox'], inplace=True)
    
    train_loader, val_loader = get_dataloaders(df_folds, markings, config, Path(opt.path))
    
    ## MODEL
    if opt.weights:
        net = load_net(opt.weights, config.img_sz)
    else:
        net = get_net(config.img_sz)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)

    mixed_precision = config.mixed_precision
    
    ## TRAINING
    fitter = Fitter(model=net, device=device, config=config)
    fitter.fit(train_loader, val_loader)