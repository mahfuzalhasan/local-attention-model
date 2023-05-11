import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from config import config
from dataloader.dataloader import get_train_loader
from models.student_builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor

from tensorboardX import SummaryWriter


model_file = "../../Results/saved_models/segformer/mit_b2.pth"

criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
model = segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d)

model = nn.DataParallel(model, device_ids = [0,1])
model.to(f'cuda:{model.device_ids[0]}', non_blocking=True)


#state_dict = torch.load(model_file)

#model.load_state_dict(model_file)
#print("model loaded")

B = 8
H = 480
W = 640
C = 3

x = torch.randn((B,C,H,W))

y = model(x)
print(y.size())