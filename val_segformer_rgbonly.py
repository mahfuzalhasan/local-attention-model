import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
from  torch.utils.data import DataLoader
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel, DataParallel

## Dataset-specific imports
# from config import config
from config_cityscapes import config
from eval import SegEvaluator
# from dataloader.dataloader import get_train_loader

from models.builder import EncoderDecoder as segmodel
# from dataloader.RGBXDataset import RGBXDataset


from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor

from tensorboardX import SummaryWriter


def val_cityscape(epoch, val_dataset, val_loader, model):
    model.eval()
    sum_loss = 0
    with torch.no_grad():
        for idx, sample in enumerate(val_loader):

            # engine.update_iteration(epoch, idx)
            # minibatch = next(dataloader) #minibatch = dataloader.next()
            imgs = sample['image']
            gts = sample['label']
            imgs = imgs.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
            gts = gts.to(f'cuda:{model.device_ids[0]}', non_blocking=True)  

            aux_rate = 0.2
            loss = model(imgs, gts)

            # mean over multi-gpu result
            loss = torch.mean(loss) 

            
            sum_loss += loss

            print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                    + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                    + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))+'\n'

            del loss
            if idx % config.print_stats == 0:
                #pbar.set_description(print_str, refresh=True)
                print(f'{print_str}')

        val_loss = sum_loss/len(val_loader)
        print(f"########## epoch:{epoch} val_loss:{val_loss}############")
        
        print(f"\n $$$$$$$ evaluating in epoch:{epoch} $$$$$$$ \n")

        
        all_devices = config.device_ids
        save_path = None
        verbose = False
        show_image = False
        segmentor = SegEvaluator(val_dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, model,
                                 config.eval_scale_array, config.eval_flip,
                                 all_devices, verbose, save_path,
                                 show_image)
                                 
        output_dict = segmentor.run_eval(model)

        return val_loss, output_dict
