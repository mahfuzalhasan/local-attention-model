import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm
from datetime import datetime
import numpy as np

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
from utils.metric import cal_mean_iou

from tensorboardX import SummaryWriter


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')



def val_cityscape(epoch, val_loader, model):
    model.eval()
    sum_loss = 0
    m_iou_batches = []
    with torch.no_grad():
        for idx, sample in enumerate(val_loader):

            # engine.update_iteration(epoch, idx)
            # minibatch = next(dataloader) #minibatch = dataloader.next()
            imgs = sample['image']
            gts = sample['label']
            imgs = imgs.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
            gts = gts.to(f'cuda:{model.device_ids[0]}', non_blocking=True)  

            aux_rate = 0.2
            loss, out = model(imgs, gts)
            # mean over multi-gpu result
            loss = torch.mean(loss)
            m_iou = cal_mean_iou(out, gts)
            
            m_iou_batches.append(m_iou)

            sum_loss += loss

            print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                    + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                    + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))+'\n'

            del loss
            if idx % config.val_print_stats == 0:
                #pbar.set_description(print_str, refresh=True)
                print(f'{print_str}')

        val_loss = sum_loss/len(val_loader)
        val_mean_iou = np.mean(np.asarray(m_iou_batches))
        print(f"\n $$$$$$$ evaluating in epoch:{epoch} $$$$$$$ \n")
        print(f"########## epoch:{epoch} val_loss:{val_loss} mean val iou: {val_mean_iou}############")

        return val_loss, val_mean_iou
