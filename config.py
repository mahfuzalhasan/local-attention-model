import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 12345

remoteip = os.popen('pwd').read()
#C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.root_dir = os.getcwd()
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'NYUDepthv2'
#C.dataset_path = r'./data/NYUDepthv2'
C.dataset_path = osp.join(C.root_dir, 'data/NYUDepthv2')
C.rgb_root_folder = osp.join(C.dataset_path, 'RGB')
C.rgb_format = '.jpg'
C.gt_root_folder = osp.join(C.dataset_path, 'Label')
C.gt_format = '.png'
C.gt_transform = True
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, Faslse for MFNet(?)
C.x_root_folder = osp.join(C.dataset_path, 'HHA')
C.x_format = '.jpg'
C.x_is_single_channel = False # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = False
C.num_train_imgs = 795
C.num_eval_imgs = 654
C.num_classes = 40
C.class_names =  ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
    'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
    'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
    'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']

"""Image Config"""
C.background = 255
C.image_height = 480
C.image_width = 640
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'mit_b2' # Remember change the path below.
C.pretrained_model = osp.join(C.root_dir, 'pretrained/mit_b2_imagenet.pth')
#C.pretrained_model = "../../Results/saved_models/NYUDV2_CMX+Segformer-B2.pth"
C.decoder = 'MLPDecoder'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8
C.nepochs = 500
C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1
C.num_workers = 4
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.print_stats = 30
# C.device_ids = [0, 1] # for mahdi
C.device_ids = [0, 1, 2, 3] # for sakin
C.resume_train = False 
C.resume_model_path = osp.join(C.root_dir, 'Results/saved_models/07-10-23_2314/model_330.pth')

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [0.75, 1, 1.25] # 
C.eval_flip = False # True # 
C.eval_crop_size = [480, 640] # [height width]

"""Store Config"""
C.checkpoint_start_epoch = 250
C.checkpoint_step = 5

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.log_dir = "./Results/logs"
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = "./Results/saved_models"

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()
