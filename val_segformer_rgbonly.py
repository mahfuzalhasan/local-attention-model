import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm
from datetime import datetime
import numpy as np
import cv2
import copy

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
from utils.metric import cal_mean_iou, hist_info, compute_score
from utils.visualize import unnormalize_img_numpy

from tensorboardX import SummaryWriter


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def compute_metric(results):
    hist = np.zeros((config.num_classes, config.num_classes))
    correct = 0
    labeled = 0
    count = 0
    for d in results:
        hist += d['hist']
        correct += d['correct']
        labeled += d['labeled']
        count += 1
    iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
    print(f'iou:{iou} miou:{mean_IoU}')
    result_dict = dict(mean_iou=mean_IoU, freq_iou=freq_IoU, mean_pixel_acc=mean_pixel_acc)
    return result_dict

def plot_attention(atn_h, output_folder, head_no):
    # avg = atn_h.mean(dim=2)
    # avg = avg.detach().cpu().numpy()
    # avg = avg * 255.0
    # avg = np.array(avg, dtype=np.uint8)
    # cv2.imwrite(os.path.join(output_folder, f'avg_h_avg_c.jpg'), avg)
    head_folder = os.path.join(output_folder, str(head_no))
    if not os.path.exists(head_folder):
        os.makedirs(head_folder)
    for c in range(atn_h.shape[2]):
        single_head = atn_h[:, :, c]
        single_head = single_head.detach().cpu().numpy()
        single_head = single_head * 255.0
        single_head = np.array(single_head, dtype=np.uint8)
        
        cv2.imwrite(os.path.join(head_folder, f'c_{c}.jpg'), single_head)

def val_cityscape(epoch, val_loader, model):
    model.eval()
    sum_loss = 0
    m_iou_batches = []
    all_results = []
    unique_values = []
    attn_heads = [2, 4, 5, 8]
    base_output_folder = './check_output'
    with torch.no_grad():
        for idx, sample in enumerate(val_loader):
            imgs = sample['image']      #B, 3, 1024, 2048
            gts = sample['label']       #B, 1024, 2048
            imgs = imgs.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
            gts = gts.to(f'cuda:{model.device_ids[0]}', non_blocking=True)

            imgs_1, imgs_2 = imgs[:, :, :, :1024], imgs[:, :, :, 1024:]
            gts_1, gts_2 = gts[:, :, :1024], gts[:, :, 1024:]
            n_img = unnormalize_img_numpy(imgs_1)
            n_img = cv2.resize(n_img, (128, 128))
            cv2.imwrite(os.path.join(base_output_folder, f'batch_{idx}_1.jpg'), n_img)
            # exit()

            
            
            loss_1, out_1, attention_matrices = model(imgs_1, gts_1, visualize=True, attention=True)
            
            for i, attn_matrix in enumerate(attention_matrices):
                print(i)
                if i != len(attention_matrices)-1:
                    continue
                num_heads = attn_heads[i]
                print('nh: ',num_heads)
                B, N, C = attn_matrix.shape
                # print('attn matrix: ',attn_matrix.shape)
                attn_matrix = attn_matrix.reshape(B, N, num_heads, C // num_heads).permute(0, 2, 1, 3)
                # # nh = attn_matrix.shape[1]  # number of head
                # # attn_matrix = attn_matrix.mean(dim=1)
                # attn_matrix = attn_matrix[0]
                # atn_h = attn_matrix.reshape(1024//(2**(i+2)), 1024//(2**(i+2)), C//num_heads).permute(2,0,1)
                # # print('atn_h before: ',atn_h.shape)
                # factor = (4, 4)
                # atn_h = nn.functional.interpolate(atn_h.unsqueeze(
                #             0), scale_factor=factor, mode="nearest")[0]
                # atn_h = atn_h.permute(1, 2, 0)
                
                # # print('atn_h after: ',atn_h.shape)
                # stage = os.path.join(base_output_folder, f'attention_stage_{i}')
                # if not os.path.exists(stage):
                #     os.makedirs(stage)
                # plot_attention(atn_h, stage, 0)
                # print('attn matrix: ',attn_matrix.shape)
                for k in range(num_heads):
                    atn_h = attn_matrix[0, k, :, :]
                    # print('attn matrix head: ',atn_h.shape)
                    atn_h = atn_h.reshape(1024//(2**(i+2)), 1024//(2**(i+2)), C//num_heads).permute(2,0,1)
                    # print('atn_h before: ',atn_h.shape)
                    factor = (4, 4)
                    atn_h = nn.functional.interpolate(atn_h.unsqueeze(
                                0), scale_factor=factor, mode="nearest")[0]
                    atn_h = atn_h.permute(1, 2, 0)
                    
                    # print('atn_h after: ',atn_h.shape)
                    stage = os.path.join(base_output_folder, f'attention_stage_{i}')
                    if not os.path.exists(stage):
                        os.makedirs(stage)
                    plot_attention(atn_h, stage, k)
                    
                exit()
                
                # # keep only the output patch attention
                # attn_matrix = attn_matrix[0, :, 0, 1:].reshape(nh, -1)
                # print('attn matrix: ',attn_matrix.shape)

                # attn_matrix = attn_matrix.reshape(nh, w_featmap, h_featmap)
                # attentions = nn.functional.interpolate(attentions.unsqueeze(
                #     0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

                # print(attn_matrix.size())
            # print(f'loss: {loss_1}, out_1: {out_1.shape}')
            # print(f'imgs_1: {imgs_1.shape} attn matrices:{len(attention_matrices)}')

            
            exit()
            # print('loss_2: ',loss_2)
            loss_2, out_2 = model(imgs_2, gts_2)

            out = torch.cat((out_1, out_2), dim = 3)

            # mean over multi-gpu result
            loss = torch.mean(loss_1) + torch.mean(loss_2)
            #miou using torchmetric library
            m_iou = cal_mean_iou(out, gts)

            score = out[0]      #1, C, H, W --> C, H, W = 19, H, W
            score = torch.exp(score)    
            score = score.permute(1, 2, 0)  #H,W,C
            pred = score.argmax(2)  #H,W
            
            pred = pred.detach().cpu().numpy()
            gts = gts[0].detach().cpu().numpy() #1, H, W --> H, W
            confusionMatrix, labeled, correct = hist_info(config.num_classes, pred, gts)
            results_dict = {'hist': confusionMatrix, 'labeled': labeled, 'correct': correct}
            all_results.append(results_dict)

            m_iou_batches.append(m_iou)

            sum_loss += loss

            # print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
            #         + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
            #         + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))+'\n'

            del loss
            if idx % config.val_print_stats == 0:
                #pbar.set_description(print_str, refresh=True)
                print(f'sample {idx}')

        val_loss = sum_loss/len(val_loader)
        result_dict = compute_metric(all_results)
        # print('all unique class values: ', list(set(unique_values)))

        print(f"\n $$$$$$$ evaluating in epoch:{epoch} $$$$$$$ \n")
        print('result: ',result_dict)
        val_mean_iou = np.mean(np.asarray(m_iou_batches))
        print(f"########## epoch:{epoch} mean_iou:{result_dict['mean_iou']} ############")
        print(f"########## mean_iou using torchmetric library:{val_mean_iou} ############")
        
        return val_loss, result_dict['mean_iou']
