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
# from dataloader.dataloader import get_train_loader
from dataloader.cityscapes_dataloader import get_train_loader
from models.builder import EncoderDecoder as segmodel
# from dataloader.RGBXDataset import RGBXDataset
from dataloader.cityscapes_dataloader import CityscapesDataset
from val_segformer_rgbonly import val_cityscape

from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor

from dataloader.cfg_defaults import get_cfg_defaults
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


from tensorboardX import SummaryWriter


def Main(parser, cfg, args):
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

        # train_loader, train_sampler = get_train_loader(engine, CityscapesDataset) 
        # print('train dataloader size: ',len(train_loader))


        cityscapes_train = CityscapesDataset(cfg, split='train')
        train_loader = DataLoader(cityscapes_train, batch_size=8, shuffle=True, num_workers=4)

        cityscapes_val = CityscapesDataset(cfg, split='val')
        val_loader = DataLoader(cityscapes_val, batch_size=8, shuffle=False, num_workers=4)

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
        print('Loaded fresh model')

        # Loading model from Segformer's starting point (load model pretrained on ImageNet)   
        #load_model_from = osp.join(config.pretrained_model)
        #state_dict = torch.load(load_model_from)
        #model.load_state_dict(state_dict, strict=True)
        print('Loaded model from state_dict')
        # exit()
        
        
        # group weight and config optimizer
        base_lr = config.lr
        if engine.distributed:
            base_lr = config.lr
        
        params_list = []
        params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
        
        if config.optimizer == 'AdamW': # Segformer original uses AdamW
            optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
        elif config.optimizer == 'SGDM':
            optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
        else:
            raise NotImplementedError

        # config lr policy
        total_iteration = config.nepochs * config.niters_per_epoch
        # 6e-5, 0.9, 500*100 = 50000, 10000
        lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

        

    
        starting_epoch = 1
        if config.resume_train:
            print('Loading model to resume train')
            state_dict = torch.load(config.resume_model_path)
            model.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])
            starting_epoch = state_dict['epoch']
            print('resuming training with model: ', config.resume_model_path)

    

        logger.info('begin training:')
        
        for epoch in range(starting_epoch, config.nepochs):
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

            model.train()
            optimizer.zero_grad()
            sum_loss = 0
            for idx, sample in enumerate(train_loader):

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

                # print("loss: ",loss)

                # reduce the whole loss over multi-gpu
                # if engine.distributed:
                #     reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Need to Change Lr Policy
                current_idx = (epoch- 1) * config.niters_per_epoch + idx 
                lr = lr_policy.get_lr(current_idx)

                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = lr

            
                sum_loss += loss
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))+'\n'

                del loss
                if idx % config.print_stats == 0:
                    #pbar.set_description(print_str, refresh=True)
                    print(f'{print_str}')

            train_loss = sum_loss/len(train_loader)
            print(f"########## epoch:{epoch} train_loss:{train_loss}############")
            writer.add_scalar('train_loss', train_loss, epoch)
            
            if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
                save_dir = os.path.join(config.checkpoint_dir, str(run_id))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file_path = os.path.join(save_dir, 'model_{}.pth'.format(epoch))
                states = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(states, save_file_path)

            ## Need to compute metrices for validation set
            val_loss, val_metrics = val_cityscape(epoch, cityscapes_val, val_loader, model)

            
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_mIOU', val_metrics['mean_iou'], epoch)
            writer.add_scalar('freq_IOU', val_metrics['freq_iou'], epoch)
            writer.add_scalar('m_pixel_acc', val_metrics['mean_pixel_acc'], epoch)

            val_mean_iou = val_metrics['mean_iou']
            print(f't_loss:{train_loss:.4f} v_loss:{val_loss:.4f} val_mIOU:{val_mean_iou:.4f}')

        
        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Test cityscapes Loader")
    parser.add_argument('config_file', help='config file path')
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    #parser.add_argument('devices', default='0,1', type=str)
    args = parser.parse_args()
    print(args.opts)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file) # dataloader/cityscapes_rgbd_config.yaml
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)

    logger = get_logger()

    os.environ['MASTER_PORT'] = '169710'
    Main(parser, cfg, args)