import os
import cv2
import argparse
import numpy as np
import scipy.io as sio

import os.path as osp
import sys
import time
import argparse
from tqdm import tqdm
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from  torch.utils.data import DataLoader
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel, DataParallel

from config_cityscapes import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score

from models.builder import EncoderDecoder as segmodel

from dataloader.cfg_defaults import get_cfg_defaults
from dataloader.cityscapes_dataloader import CityscapesDataset

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

logger = get_logger()

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['image']      # H, W, C
        label = data['label']  
        # modal_x = data['modal_x']
        name = data['id']
        # print(f'Image Shape: ####  {img.shape} {type(img)}')
        # print(f'Label Shape: ####  {label.shape} {type(img)}')

        img = torch.permute(img, (1, 2, 0))
        img = img.detach().cpu().numpy()
        # img = img.numpy(force = True)
        # label = torch.permute(label, (1, 2, 0))
        label = label.detach().cpu().numpy()
        # label = label.numpy(force = True)

        pred = self.sliding_eval_rgbX(img, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}


        ###### output is saved here.
        if self.save_path is not None:
            ensure_dir(self.save_path)
            ensure_dir(self.save_path+'_color')

            fn = name + '.png'

            # save colored result
            result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
            class_colors = get_class_colors()
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(self.save_path+'_color', fn))

            # save raw result
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info('Save the image ' + fn)

        if self.show_image:
            colors = self.dataset.get_class_colors
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
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
        result_dict = dict(mean_iou=mean_IoU, freq_iou=freq_IoU, mean_pixel_acc=mean_pixel_acc)
        print('result dict: ',result_dict)
        result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                dataset.class_names, show_no_back=False)
        return result_line, result_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test cityscapes Loader")
    parser.add_argument('config_file', help='config file path')
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    print(args.opts)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file) # dataloader/cityscapes_rgbd_config.yaml
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)

    logger = get_logger()

    os.environ['MASTER_PORT'] = '169710'
    
    run_id = datetime.today().strftime('%m-%d-%y_%H%M')
    print(f'$$$$$$$$$$$$$ run_id:{run_id} $$$$$$$$$$$$$')
    args = parser.parse_args()

    cudnn.benchmark = True
    seed = config.seed
    # if engine.distributed:
    #     seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    dataset = CityscapesDataset(cfg, split='val')
    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    # print("multigpu training")
    # network = nn.DataParallel(network, device_ids = [0])
    # network.to(f'cuda:{network.device_ids[0]}', non_blocking=True) #wrap weights inside module

    all_devices = config.device_ids

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_devices, verbose=False, save_path=None,
                                 show_image = False)
        # saved_model_path = os.path.join(config.checkpoint_dir, "07-14-23_1803")
        saved_model_path = config.checkpoint_dir
        saved_model_names = ["model_435_11-06-23_0957.pth", "model_495_11-06-23_0957.pth"]
        
        for i in range(len(saved_model_names)):
            name = saved_model_names[i][:saved_model_names[i].rindex('.')]+'.log'
            log_file = os.path.join(config.log_dir, name)
            print(f" ####### \n Testing with model {saved_model_names[i]} \n #######")
            segmentor.run(saved_model_path, saved_model_names[i], log_file,
                        config.link_val_log_file)
