



import numpy as np
import scipy.io as sio
import os

import cv2
import h5py
import copy # for deepcopy in update_label_ids_to_40_class()

from label_filter import Label_Filter

# Torch Related
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.encoders.dual_segformer import RGBXTransformer
from models.builder import EncoderDecoder as segmodel
from config import config

from utils.pyt_utils import load_model




class NYUD(Dataset):

    def __init__(self, 
                 data_mat,
                 class_mapping):

        #  loading data path 
        if not (os.path.isfile(data_mat) or os.path.isfile(class_mapping)):
            print("File does not exist ")
        else:
            self.file = h5py.File(data_mat)        
            print("mat File loaded")

        images = self.file['images']
        depths = self.file['depths']
        labels = self.file['labels']

        #print("images shape: ", images.shape, type(images))
        #print("depths shape: ", depths.shape, type(depths))
        #print("labels shape: ", labels.shape, type(labels))
        self.images = np.array(images[0:].transpose(3,2,1,0))
        self.depths = np.array(depths[0:].transpose(2,1,0))
        self.labels = np.array(labels[0:].transpose(2,1,0))
        # label = label.transpose(1,0)

        #print("images shape: ", self.images.shape, self.images.dtype)
        #print("depths shape: ", self.depths.shape, self.depths.dtype)
        #print("labels shape: ", self.labels.shape, self.labels.dtype)

        self.file.close()

        self.label_filter = Label_Filter(class_mapping)
        
        pass
    def convert_to_tensor(self, img):

        img_torch = torch.from_numpy(img)
        img_torch = img_torch.type(torch.FloatTensor)
        img_torch = img_torch.permute(-1, 0, 1)
        return img_torch
    
    def __getitem__(self, index):
        # if index > self.size:
        #     raise ValueError
        sample = {}
        img = self.images[:,:,:,index]
        img = np.array(img, dtype='f')
        #print("img type: ", img.dtype, img.shape)
        depth_map = self.depths[:,:,index]
        #print("depth type: ",depth_map.dtype, depth_map.shape)
        label = self.labels[:,:,index]   
        label = self.update_label_ids_to_40_class(label)
        #print("16bit label:",np.unique(label))
        label = np.array(label, dtype=np.uint8)
        #print("8bit label:",np.unique(label), label.dtype, label.shape)

        modal_x = np.zeros_like(img)
        modal_x[:,:,0] = depth_map
        modal_x[:,:,1] = depth_map
        modal_x[:,:,2] = depth_map
        
        #depth_map = np.expand_dims(depth_map, axis=2)
        label = np.expand_dims(label, axis=2)
        

        img = img / 255.0
        modal_x = modal_x / np.max(modal_x) 
        
        img = self.convert_to_tensor(img)
        modal_x = self.convert_to_tensor(modal_x)
        label = self.convert_to_tensor(label)


        #exit()

        
             
        # Augmentation
        
        #sample['label'] = self.update_label_ids_to_40_class(sample['label'])
        
        
        return {'image': img, 'depth': modal_x, 'label': label}
    
    
    def get_original_label_names(self, label):
        id_list = np.unique(label)
        names = []
        for id_ in id_list:
            names.append(self.label_filter.get_label_name_orig(id_))
        return names    
    def get_original_label_ids(self, label):
        return np.unique(label)
    def update_label_names_to_40_class(self, label):
        id_list = np.unique(label)
        new_label_ids = []
        for id_ in id_list:
            new_label_ids.append(self.label_filter.get_label_name_40(id_))     
        return new_label_ids
    def update_label_ids_to_40_class(self, label):
        # #print(type(label))
        unique = np.unique(label)
        label_copy = copy.deepcopy(label)

        count1 = {}
        count2 = {}
        count3 = {}

        # testing shape1
        for item in unique:
            count1[item] = np.count_nonzero(label == item)
        # testing shape2
        for item in np.unique(label_copy):
            count2[item] = np.count_nonzero(label_copy == item) 
        # #print(cur_item, replace_with)
                        
        # replace label ids with 40 class ids
        id_list = np.unique(label_copy)
        for id_ in id_list:
            # get current label and to-be-updated label
            cur_item = id_
            replace_with = self.label_filter.get_label_id_40(id_)
            label_copy[label_copy == cur_item] = replace_with
        
        #testing shape3
        for item in np.unique(label_copy):
            count3[item] = np.count_nonzero(label_copy == item) 
                    
 
        # #print('previous count ', count1)
        # # #print('previous unique ', unique)
        # #print('after deepcopy ', count2)
        # # #print('previous unique ', unique)
        # #print('final count ', count3)
        
        # #print(type(label_copy))
        return label_copy


    def __len__(self):
        return self.images.shape[3]

    pass


if __name__=="__main__":
    ## main
    data_dir = '../../data/nuydv2'
    #print(os.path.join(data_dir, 'nyu_depth_v2_labeled.mat'))
    data_mat = os.path.join(data_dir, 'nyu_depth_v2_labeled.mat')
    class_mapping_csv = os.path.join(data_dir, 'classMapping.csv')
    # labels that remain in the final image after filtering
    # other labels are set to background
    network=segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    model = load_model(network, "../../Results/saved_models/NYUDV2_CMX+Segformer-B2.pth")
    #model = RGBXTransformer(img_size=(480, 640))

    dataset = NYUD(data_mat, 
                class_mapping_csv)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    nth_sample = 100
    depth_values = []
    label_values = []



    for i, sample_data in enumerate(dataloader):
        
        #sample_data = dataset.__getitem__(i)

        #  show image from the sample_data
        img = sample_data['image']
        
        """Conversion of depth needs to be included""" 
        depth = sample_data['depth']
        
        #depth_values.extend(np.unique(depth))
        #cv2.imwrite('depth.png', depth)
        label = sample_data['label']
        with torch.no_grad():
            seg_outputs = model(img, depth)

        for out in seg_outputs:
            print(f'seg output: {out.size()}')

        
        
        if i==10:
            exit()
        ##print(f'img shape: {img.size()} depth shape: {depth.size()} label shape: {label.size()}')
        
        
        #label_values.extend(np.unique(label))
    # #print("label:" , list(set(label_values)), len(list(set(label_values))))

    # depth_values = list(set(depth_values))
    # depth_values = np.asarray(depth_values)
    # #print("max:", np.max(depth_values), np.min(depth_values), depth_values.shape[0])

        



    # from matplotlib import pyplot as plt

    # fig, axs = plt.subplots(nrows=1,ncols=3)
    # axs[0].imshow(img)
    # axs[1].imshow(depth)
    # axs[2].imshow(label)
    # axs[0].title.set_text('image')
    # axs[1].title.set_text('depth')
    # axs[2].title.set_text('label')

    # plt.show()
