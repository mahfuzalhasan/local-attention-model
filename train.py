import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel, DataParallel

from config import config
from dataloader.nyudv2_dataloader import get_train_loader
from models.builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
logger = get_logger()

os.environ['MASTER_PORT'] = '169710'

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# print("parser: ",parser)

with Engine(custom_parser=parser) as engine:
    run_id = datetime.today().strftime('%m-%d-%y_%H%M')
    print(f'$$$$$$$$$$$$$ run_id:{run_id} $$$$$$$$$$$$$')
    args = parser.parse_args()

    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset)
    print('train dataloader: ',len(train_loader))

    # if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
    #     tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
    #     generate_tb_dir = config.tb_dir + '/tb'
    #     tb = SummaryWriter(log_dir=tb_dir)
    #     engine.link_tb(tb_dir, generate_tb_dir)
    save_log = os.path.join(config.log_dir, str(run_id))
    if not os.path.exists(save_log):
        os.makedirs(save_log)
    writer = SummaryWriter(save_log)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    
    model=segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    
    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr
    
    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
    
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    # 6e-5, 0.9, 500*100 = 50000, 10000
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        
        if torch.cuda.is_available():
            ####### Use DataParallel
            model.cuda()

            model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                            output_device=engine.local_rank, find_unused_parameters=False)
    else:
        print("multigpu training")
        print('device ids: ',config.device_ids)
        model = nn.DataParallel(model, device_ids = config.device_ids)
        model.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device)

    # engine.register_state(dataloader=train_loader, model=model,
    #                       optimizer=optimizer)
    # if engine.continue_state_object:
    #     engine.restore_checkpoint()

    starting_epoch = 1
    if config.resume_train:
        print('Loading model to resume train')
        state_dict = torch.load(config.resume_model_path)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        starting_epoch = state_dict['epoch']
        print('resuming training with model: ', config.resume_model_path)

    optimizer.zero_grad()
    model.train()
    logger.info('begin training:')
    
    # total epoch
    # for epoch in range(engine.state.epoch, config.nepochs+1):
    for epoch in range(starting_epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        # how manyiteration in ine epoch
        dataloader = iter(train_loader)

        sum_loss = 0

        for idx in pbar:
            engine.update_iteration(epoch, idx)
            #print('epoch: ', epoch, 'idx: ', idx)
            #print(len(dataloader))
            minibatch = next(dataloader) #minibatch = dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']


            # imgs = imgs.cuda(non_blocking=True)
            # gts = gts.cuda(non_blocking=True)
            # modal_xs = modal_xs.cuda(non_blocking=True)

            imgs = imgs.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
            gts = gts.to(f'cuda:{model.device_ids[0]}', non_blocking=True) 
            modal_xs = modal_xs.to(f'cuda:{model.device_ids[0]}', non_blocking=True) 

            aux_rate = 0.2
            loss = model(imgs, modal_xs, gts)
            #print('loss size: ',loss.size())

            # mean over multi-gpu result
            loss = torch.mean(loss) 

            # print("loss: ",loss)

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch- 1) * config.niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
            else:
                sum_loss += loss
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))+'\n'

            del loss
            if idx % config.print_stats == 0:
                #pbar.set_description(print_str, refresh=True)
                print(f'{print_str}')

        print(f"########## epoch:{epoch} train_loss:{sum_loss/len(pbar)}############")
        writer.add_scalar('train_loss', sum_loss / len(pbar), epoch)
        # if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        #     tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)

        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            # if engine.distributed and (engine.local_rank == 0):
            #     engine.save_and_link_checkpoint(config.checkpoint_dir,
            #                                     config.log_dir,
            #                                     config.log_dir_link)
            # elif not engine.distributed:
            #     engine.save_and_link_checkpoint(config.checkpoint_dir,
            #                                     config.log_dir,
            #                                     config.log_dir_link)
            save_dir = os.path.join(config.checkpoint_dir, str(run_id))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file_path = os.path.join(save_dir, 'model_{}.pth'.format(epoch))
            ##### first send model to CPU and then save weight
            states = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            torch.save(states, save_file_path)
