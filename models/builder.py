
import os, sys

# from models.builder import EncoderDecoder as segmodel
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) 
sys.path.append(parent_dir)

model_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(model_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, flop_count_table
from ptflops import get_model_complexity_info
# from utils.init_func import init_weight
# from utils.load_utils import load_pretrain
from functools import partial
import numpy as np

from engine.logger import get_logger

from utils.init_func import init_weight
from config_cityscapes import config

logger = get_logger()

class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.BatchNorm2d, test=False):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        self.test = test
        # import backbone and decoder
        if cfg.backbone == 'swin_s':
            logger.info('Using backbone: Swin-Transformer-small')
            from .encoders.dual_swin import swin_s as backbone
            self.channels = [96, 192, 384, 768]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'swin_b':
            logger.info('Using backbone: Swin-Transformer-Base')
            from .encoders.dual_swin import swin_b as backbone
            self.channels = [128, 256, 512, 1024]
            self.backbone = backbone(norm_fuse=norm_layer)
        
        elif cfg.backbone == 'mit_b5':
            logger.info('Using backbone: Segformer-B5')
            from .encoders.dual_segformer import mit_b5 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        
        elif cfg.backbone == 'mit_b4':
            logger.info('Using backbone: Segformer-B4')
            from .encoders.dual_segformer import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        
        elif cfg.backbone == 'mit_b2':
            logger.info('Using backbone: Segformer-B2')
            from encoders.student_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        
        elif cfg.backbone == 'mit_b1':
            logger.info('Using backbone: Segformer-B1')
            from .encoders.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b0':
            logger.info('Using backbone: Segformer-B0')
            self.channels = [32, 64, 160, 256]
            from .encoders.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        else:
            logger.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        self.aux_head = None

        if cfg.decoder == 'MLPDecoder':
            logger.info('Using MLP Decoder')
            from decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)
        
        elif cfg.decoder == 'UPernet':
            logger.info('Using Upernet Decoder')
            from .decoders.UPernet import UPerHead
            self.decode_head = UPerHead(in_channels=self.channels ,num_classes=cfg.num_classes, norm_layer=norm_layer, channels=512)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)
        
        elif cfg.decoder == 'deeplabv3+':
            logger.info('Using Decoder: DeepLabV3+')
            from .decoders.deeplabv3plus import DeepLabV3Plus as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        else:
            logger.info('No decoder(FCN-32s)')
            from .decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=cfg.num_classes, norm_layer=norm_layer)

        self.criterion = criterion
        if self.criterion and not self.test:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)
    
    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            # pretrained = '/home/abjawad/Documents/GitHub/local-attention-model/pretrained/mit_b2.pth'
            pretrained = cfg.pretrained_model
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        x = self.backbone(rgb)
        out = self.decode_head.forward(x)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        if self.aux_head:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
            return out, aux_fm
        return out

    def forward(self, rgb, label=None):
        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb)
        else:
            out = self.encode_decode(rgb)
            # print(f'#############output: {out.size()}')
            # print(out)

        # print(f'output: {out.size()}')
        # print('############# label ################## \n ')
        # label = label.long()
        # unique_values = torch.unique(label)
        # print(unique_values)
        # print('##########################################')
        if label is not None:
            loss = self.criterion(out, label.long())
        
        # print(f'loss:{loss}')
        if self.aux_head:
            loss += self.aux_rate * self.criterion(aux_fm, label.long())
            return loss, out
        return out


def sra_flops(h, w, r, dim, num_heads):
    dim_h = dim / num_heads
    n1 = h * w
    n2 = h / r * w / r

    f1 = n1 * dim_h * n2 * num_heads
    f2 = n1 * n2 * dim_h * num_heads

    return f1 + f2

def get_tr_flops(net, input_shape):
    flops, params = get_model_complexity_info(net, input_shape, as_strings=False)
    _, H, W = input_shape
    net = net.backbone
    try:
        stage1 = sra_flops(H // 4, W // 4,
                           net.block1[0].attn.sr_ratio,
                           net.block1[0].attn.dim,
                           net.block1[0].attn.num_heads) * len(net.block1)
        stage2 = sra_flops(H // 8, W // 8,
                           net.block2[0].attn.sr_ratio,
                           net.block2[0].attn.dim,
                           net.block2[0].attn.num_heads) * len(net.block2)
        stage3 = sra_flops(H // 16, W // 16,
                           net.block3[0].attn.sr_ratio,
                           net.block3[0].attn.dim,
                           net.block3[0].attn.num_heads) * len(net.block3)
        stage4 = sra_flops(H // 32, W // 32,
                           net.block4[0].attn.sr_ratio,
                           net.block4[0].attn.dim,
                           net.block4[0].attn.num_heads) * len(net.block4)
    except:
        stage1 = sra_flops(H // 4, W // 4,
                           net.block1[0].attn.squeeze_ratio,
                           64,
                           net.block1[0].attn.num_heads) * len(net.block1)
        stage2 = sra_flops(H // 8, W // 8,
                           net.block2[0].attn.squeeze_ratio,
                           128,
                           net.block2[0].attn.num_heads) * len(net.block2)
        stage3 = sra_flops(H // 16, W // 16,
                           net.block3[0].attn.squeeze_ratio,
                           320,
                           net.block3[0].attn.num_heads) * len(net.block3)
        stage4 = sra_flops(H // 32, W // 32,
                           net.block4[0].attn.squeeze_ratio,
                           512,
                           net.block4[0].attn.num_heads) * len(net.block4)

    print(stage1 + stage2 + stage3 + stage4)
    flops += stage1 + stage2 + stage3 + stage4
    # return flops_to_string(flops), params_to_string(params)
    return flops, params

if __name__ == "__main__":


    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
    model=EncoderDecoder(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d)

    input = torch.randn(1, 3, 1024, 1024).cuda()

    model.cuda()
    # flops = FlopCountAnalysis(model, input)

    input_shape = (3, 512, 512)
    flops, params = get_tr_flops(model, input_shape)
    
    # macs, params = get_model_complexity_info(model, (3, 1024, 1024), as_strings=True,
    #                                        print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    print('flops:', flops)
    # print(flop_count_table(flops))
    print('params:', params)

