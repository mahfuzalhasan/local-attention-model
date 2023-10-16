import os, sys
currentFolder = os.path.abspath('')
try:
    sys.path.remove(str(currentFolder))
except ValueError: # Already removed
    pass

projectFolder = 'C:/Users/abjawad/Documents/GitHub/local-attention-model'
sys.path.append(str(projectFolder))
os.chdir(projectFolder)
print( f"current working dir{os.getcwd()}")






import torch
from  torch.utils.data import DataLoader
import torch.nn as nn
from models.builder import EncoderDecoder as segmodel
from dataloader.cfg_defaults import get_cfg_defaults
from utils.lr_policy import WarmUpPolyLR
from utils.init_func import init_weight, group_weight
from config_cityscapes import *
import yaml
import os
from dataloader.cityscapes_dataloader import CityscapesDataset





config_path = 'C:/Users/abjawad/Documents/GitHub/local-attention-model/dataloader/cityscapes_rgbd_config.yaml'
# with open(config_path) as info:
#     info_dict = yaml.load(info, Loader=yaml.FullLoader)
cfg = get_cfg_defaults()
cfg.merge_from_file(config_path)
cfg.freeze()

data_mean = [0.291,  0.329,  0.291]
data_std = [0.190,  0.190,  0.185]




cityscapes_train = CityscapesDataset(cfg, split='train')
train_loader = DataLoader(cityscapes_train, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
print(f'total train: {len(cityscapes_train)} t_iteration:{len(train_loader)}')

cityscapes_val = CityscapesDataset(cfg, split='val')
val_loader = DataLoader(cityscapes_val, batch_size=1, shuffle=False, num_workers=0)
print(f'total val: {len(cityscapes_val)} v_iteration:{len(val_loader)}')




import matplotlib.pyplot as plt
import numpy as np

print('hello')
# Access the first batch from the DataLoader
for batch in train_loader:
    images = batch['image']

    # Process each image in the batch
    for i in range(images.shape[0]):
        img = images[i].numpy().transpose((1, 2, 0))
        img *= data_std
        img += data_mean

        # print(img.shape)
    #     # Plot the image in the subplot
    #     plt.subplot(1, 4, i + 1)  # 1 row, 4 columns, i+1 is the current subplot
    #     plt.imshow(img)
    #     plt.axis('off')

    # plt.show()

    # Break the loop after the first batch
    break
