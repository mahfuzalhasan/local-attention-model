import os
import cv2
import argparse
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn

from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.RGBXDataset import RGBXDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre

logger = get_logger()

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        modal_x = data['modal_x']
        name = data['fn']
        pred = self.sliding_eval_rgbX(img, modal_x, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

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
        
        result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                dataset.class_names, show_no_back=False)
        return result_line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)

    args = parser.parse_args()
    # all_devices = parse_devices(config.device_ids)
    all_devices = config.device_ids


    #data_dir = r'./data/nyudv2'
    splits = sio.loadmat(os.path.join(config.dataset_path,'splits.mat'))

    train = splits['trainNdxs']
    test = splits['testNdxs']
    trainIds = []
    testIds = []

    for i in range(len(train)):
        trainIds.append(int(train[i][0]))
    for i in range(len(test)):
        testIds.append(int(test[i][0]))

    trainIds = [idx-1 for idx in trainIds]
    testIds = [idx-1 for idx in testIds]

    #print("train Ids: ",trainIds, len(trainIds))
    #print('###############################')
    #print("test Ids: ",testIds,len(testIds))

    #exit()
    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    print("multigpu training")
    # network = nn.DataParallel(network, device_ids = config.device_ids)
    # network.to(f'cuda:{network.device_ids[0]}', non_blocking=True) # force model to first GPU. Later flow is managed by DataParallel

    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': trainIds,
                    'eval_source': testIds,
                    'class_names': config.class_names}
    val_pre = ValPre()
    dataset = RGBXDataset(data_setting, 'val', val_pre)
 
    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_devices, args.verbose, args.save_path,
                                 args.show_image)
        saved_model_path = './pretrained' #os.path.join(config.checkpoint_dir, "07-14-23_1803")
        """ segmentor.run(config.checkpoint_dir, "NYUDV2_CMX+Segformer-B2.pth", config.val_log_file,
                      config.link_val_log_file) """
        segmentor.run(saved_model_path, "model_425.pth", config.val_log_file,
                      config.link_val_log_file)
