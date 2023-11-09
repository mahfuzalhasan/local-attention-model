import os
import cv2
import numpy as np
import time
from tqdm import tqdm
from timm.models.layers import to_2tuple

import torch
import multiprocessing as mp

from engine.logger import get_logger
from utils.pyt_utils import load_model, link_file, ensure_dir
from utils.transforms import pad_image_to_shape, normalize

logger = get_logger()


class Evaluator(object):
    def __init__(self, dataset, class_num, norm_mean, norm_std, network, multi_scales, 
                is_flip, devices, verbose=False, save_path=None, show_image=False):
        self.eval_time = 0
        self.dataset = dataset      # not wrapped by DataLoader
        ##### batch_size???? 
        self.ndata = self.dataset.get_length()      # total test file length
        self.class_num = class_num
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.multi_scales = multi_scales
        self.is_flip = is_flip
        self.network = network
        self.devices = devices

        self.context = mp.get_context('spawn')
        self.val_func = None
        self.results_queue = self.context.Queue(self.ndata)
        

        self.verbose = verbose
        self.save_path = save_path
        if save_path is not None:
            ensure_dir(save_path)
        self.show_image = show_image

    #### model in gpu

    def run(self, model_path, model_indice, log_file, log_file_link):
        """There are four evaluation modes:
            1.only eval a .pth model: -e *.pth
            2.only eval a certain epoch: -e epoch
            3.eval all epochs in a given section: -e start_epoch-end_epoch
            4.eval all epochs from a certain started epoch: -e start_epoch-
            """
        if '.pth' in model_indice:
            models = [model_indice, ]
        elif "-" in model_indice:
            start_epoch = int(model_indice.split("-")[0])
            end_epoch = model_indice.split("-")[1]

            models = os.listdir(model_path)
            models.remove("epoch-last.pth")
            sorted_models = [None] * len(models)
            model_idx = [0] * len(models)

            for idx, m in enumerate(models):
                num = m.split(".")[0].split("-")[1]
                model_idx[idx] = num
                sorted_models[idx] = m
            model_idx = np.array([int(i) for i in model_idx])

            down_bound = model_idx >= start_epoch
            up_bound = [True] * len(sorted_models)
            if end_epoch:
                end_epoch = int(end_epoch)
                assert start_epoch < end_epoch
                up_bound = model_idx <= end_epoch
            bound = up_bound * down_bound
            model_slice = np.array(sorted_models)[bound]
            models = [os.path.join(model_path, model) for model in
                      model_slice]
        else:
            if os.path.exists(model_path):
                models = [os.path.join(model_path, 'epoch-%s.pth' % model_indice), ]
            else:
                models = [None]

        results = open(log_file, 'a')
        #link_file(log_file, log_file_link)
        print("models using: ", models)
        for model in models:
            logger.info("Load Model: %s" % model)
            print('model path: ',model_path)
            self.val_func = load_model(self.network, os.path.join(model_path, model))
            if len(self.devices ) == 1:
                result_line = self.single_process_evalutation()
            else:
                result_line = self.multi_process_evaluation()

            results.write('Model: ' + model + '\n')
            results.write(result_line)
            results.write('\n')
            results.flush()

        results.close()


    def single_process_evalutation(self):
        start_eval_time = time.perf_counter()

        logger.info('GPU %s handle %d data.' % (self.devices[0], self.ndata))
        all_results = []
        
        for idx in tqdm(range(self.ndata)):
            dd = self.dataset[idx]      # one data/image
            ### called from children class --> SegEvaluator
            results_dict = self.func_per_iteration(dd, self.devices[0])
            all_results.append(results_dict)
        print("all results single process: ",all_results)
        result_line = self.compute_metric(all_results)
        logger.info(
            'Evaluation Elapsed Time: %.2fs' % (
                    time.perf_counter() - start_eval_time))
        return result_line


    def multi_process_evaluation(self):
        start_eval_time = time.perf_counter()
        nr_devices = len(self.devices)
        stride = int(np.ceil(self.ndata / nr_devices))

        ### data --> eqully distribute to devices
        ###3 nr_devices = 2, stride = 1000/2 = 500

        # start multi-process on multi-gpu
        procs = []
        for d in range(nr_devices):

            e_record = min((d + 1) * stride, self.ndata)    #500
            shred_list = list(range(d * stride, e_record))
            device = self.devices[d]
            logger.info('GPU %s handle %d data.' % (device, len(shred_list)))

            p = self.context.Process(target=self.worker,
                                     args=(shred_list, device)) # ([0-500], 0), ([500-1000], 1)    
            procs.append(p)

        for p in procs:

            p.start()

        all_results = []
        
        for _ in tqdm(range(self.ndata)):
            t = self.results_queue.get()
            all_results.append(t)
            if self.verbose:
                self.compute_metric(all_results)

        for p in procs:
            p.join()

        result_line, output_dict = self.compute_metric(all_results)
        #print("all results multi process: ",all_results)
        logger.info(
            'Evaluation Elapsed Time: %.2fs' % (
                    time.perf_counter() - start_eval_time))
        return result_line, output_dict
    
    # K threads will run this function in parallel using the passed arguments
    def worker(self, shred_list, device):
        start_load_time = time.time()
        logger.info('Load Model on Device %d: %.2fs' % (
            device, time.time() - start_load_time))

        # change for different dataset
        for idx in shred_list:
            dd = self.dataset[idx]
            # print(f'############### \n idx: {idx} device:{device}')
            # print(f'dd: {dd.keys()} \n ##################')
            results_dict = self.func_per_iteration(dd, device)
            self.results_queue.put(results_dict)
            # self.metrics_queue.put(output_dict)

    def func_per_iteration(self, data, device):
        raise NotImplementedError

    def compute_metric(self, results):
        raise NotImplementedError
    
    # add new funtion for rgb and modal X segmentation
    def sliding_eval_rgbX(self, img, crop_size, stride_rate, device=None):
        crop_size = to_2tuple(crop_size)
        ori_rows, ori_cols, _ = img.shape
        processed_pred = np.zeros((ori_rows, ori_cols, self.class_num))

        for s in self.multi_scales:     #[1]
            ### img resize to scale
            img_scale = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

            new_rows, new_cols, _ = img_scale.shape
            # H, W, C --> 480, 640, #num_class
            # if scale 1.25 --> img_scale.shape = 600, 800, 3
            processed_pred += self.scale_process_rgbX(img_scale, (ori_rows, ori_cols),
                                                        crop_size, stride_rate, device)

        # for per pixel select the class with maximum score
        pred = processed_pred.argmax(2) # 480, 640, 1
        return pred

    def scale_process_rgbX(self, img, ori_shape, crop_size, stride_rate, device=None):
        new_rows, new_cols, c = img.shape           #480, 640, 3
        long_size = new_cols if new_cols > new_rows else new_rows   #640  #800
        print(f'@@@@@img shape: {img.shape} crop size:{crop_size}')
        # new_rows = 600 # new_cols = 800    
        if new_cols <= crop_size[1] or new_rows <= crop_size[0]:
                  
            input_data, margin = self.process_image_rgbX(img, crop_size)
            ### input_data, input_modal_x ---> C, H, W
            score = self.val_func_process_rgbX(input_data, device) 
            #if scale < 1, then discard score for padded portions.
            score = score[:, margin[0]:(score.shape[1] - margin[1]), margin[2]:(score.shape[2] - margin[3])]
        
        ### if scale = 1.25, image --> 600, 800, 3
        else: 
            # stride = 320, 427
            stride = (int(np.ceil(crop_size[0] * stride_rate)), int(np.ceil(crop_size[1] * stride_rate)))
            # img_pad --> 600, 800, 3
            img_pad, margin = pad_image_to_shape(img, crop_size, cv2.BORDER_CONSTANT, value=0)
            # modal_x_pad, margin = pad_image_to_shape(modal_x, crop_size, cv2.BORDER_CONSTANT, value=0)

            pad_rows = img_pad.shape[0]     # 600
            pad_cols = img_pad.shape[1]     # 800
            r_grid = int(np.ceil((pad_rows - crop_size[0]) / stride[0])) + 1    #2
            c_grid = int(np.ceil((pad_cols - crop_size[1]) / stride[1])) + 1    #2
            data_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(device)
            
            # data_scale --> 40, 600, 800


            # stride = 320, 427
            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride[0] #0 320 
                    s_y = grid_yidx * stride[1] # 0 
                    e_x = min(s_x + crop_size[0], pad_cols) # min(480, 800) = 480 #min(800, 800) = 800
                    e_y = min(s_y + crop_size[1], pad_rows) # min(640, 600) = 600
                    s_x = e_x - crop_size[0]        # 0 320
                    s_y = e_y - crop_size[1]        # -40
                    img_sub = img_pad[s_y:e_y, s_x: e_x, :]
                    print(f'$$$$ img sub: {img_sub.shape} crop size:{crop_size} $$$$')
                    input_data, tmargin = self.process_image_rgbX(img_sub, crop_size)
                    temp_score = self.val_func_process_rgbX(input_data, device)
                    
                    temp_score = temp_score[:, tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                            tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score
            score = data_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)      # H, W, C
        data_output = cv2.resize(score.cpu().numpy(), (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)

        return data_output

    # multi-device problem exists here
    # input_data --> 3 x 480 x 640
    # input_modal_x --> 1 x 480 x 640

    # device  = 0 
    def val_func_process_rgbX(self, input_data, device=None):
        
        input_data = np.ascontiguousarray(input_data[None, :, :, :], dtype=np.float32)
        input_data = torch.FloatTensor(input_data).cuda(device) # 1, C, H, W, 

        # input_modal_x = np.ascontiguousarray(input_modal_x[None, :, :, :], dtype=np.float32)
        # input_modal_x = torch.FloatTensor(input_modal_x).cuda(device)   # 1, C, H, W

        # self.val_func --> model
        ## say model --> nn.DataParallel(device = [0, 1])
        ### self.val_func --> device 1
        with torch.cuda.device(input_data.get_device()):
            

            # send model to device --> input device
            self.val_func.to(input_data.get_device())
            self.val_func.eval()
            #print(f'device model:{self.val_func.get_device()} input_device:{input_data.get_device()}')
            with torch.no_grad():
                score = self.val_func(input_data)    # 1, C, H, W
                # print(f'score_flip: {score.shape}')
                score = score[0]                # C, H, W
                if self.is_flip:
                    input_data = input_data.flip(-1)
                    # input_modal_x = input_modal_x.flip(-1)
                    score_flip = self.val_func(input_data)
                    
                    score_flip = score_flip[0]
                    score += score_flip.flip(-1)
                score = torch.exp(score)            # e^score
        
        return score

    # for rgbd segmentation
    # when scaled image size <= crop_size (original size) condition
    def process_image_rgbX(self, img, crop_size=None):
        p_img = img
        # p_modal_x = modal_x
    
        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), amodal_xis=2)
    
        # p_img = normalize(p_img, self.norm_mean, self.norm_std)
    
        if crop_size is not None:       # 480, 640
            p_img, margin = pad_image_to_shape(p_img, crop_size, cv2.BORDER_CONSTANT, value=0)
            print(f'^^^^ p_img: {p_img.shape} margin:{margin} ^^^^^ ')
            p_img = p_img.transpose(2, 0, 1)   # C, H, W
            # margin --> length of padding on four side
            return p_img, margin
    
        p_img = p_img.transpose(2, 0, 1) # 3 H W
    
        return p_img