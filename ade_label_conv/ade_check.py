import numpy as np
from ade import ADE20KSegmentation
from label_conversion import convertFromADE, convert
import pickle
import os
import cv2

from scipy.io import loadmat
import csv

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

file  = open('./index_ade20k.pkl', 'rb')

data = pickle.load(file)

print(data.keys())

scene = data['scene']

# print(scene)

def ade20k_map_color_label(colors_path, names_path):
    colors = loadmat(colors_path)['colors']
    colors = np.concatenate([np.zeros(shape=(1,3)), colors])
    names = {0: "no class"}
    with open(names_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]
    return names, colors

def ade20k_map_color_mask(raw_mask):
    names, colors = ade20k_map_color_label("./color150.mat", "./objectInfo150.csv")
    # print('##############\n', names.keys())
    # print('\n$$$$$$$$$$$$$$$\n', colors)

    uniques, counts = np.unique(raw_mask, return_counts=True)
    print(f'uniques:{uniques} count:{counts}')
    masks = []
    for idx in np.argsort(counts)[::-1]:
        print('index:',idx)
        index_label = uniques[idx]
        
        label_mask = raw_mask == index_label

        print('index label, label_mask: ',index_label,label_mask.shape)
        ratio = counts[idx]/raw_mask.size *100
        masks.append({"class": index_label, "name": names[index_label], "color":colors[index_label], "mask": label_mask, "ratio": ratio})
    return masks

if __name__=='__main__':
    
    
    root = "/home/UFAD/mdmahfuzalhasan/Documents/Projects/local-attention-model/data/ade20k"
    train_dataset = ADE20KSegmentation(root=root, split='train', mode='train')
    print(f'train:{len(train_dataset)}')

    val_dataset = ADE20KSegmentation(root=root, split='val', mode='val')
    print('val dataset: ',len(val_dataset))
    exit()

    # color = loadmat('./color150.mat')

    # print(color, color.shape)

    # exit()
    uniques = []
    for i in range(len(val_dataset)):
        sample = val_dataset.__getitem__(i)
        path = sample['id']

        base_path = path.split('validation/')[0]
        mask_path = path.split('validation/')[1]
        mask_folder = mask_path[:mask_path.rindex('/')]
        mask_name = mask_path[mask_path.rindex('/')+1:]
        # print('base path , mask path: ',base_path, mask_path)
        # print('mask folder , mask name: ',mask_folder, mask_name)

        new_path = os.path.join(base_path, "validation/converted_mask", mask_folder)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        mask_name = mask_name[:mask_name.rindex('.')]+'_converted'+'.png'
        save_path = os.path.join(new_path, mask_name)
        # print('new path: ', new_path)
        # exit()
        if i % 500 == 499:
            print("################")
            print(f'sample processed {i}')
            print('total #unique_vals in ',i, ' images:', len(uniques), ' min max:', min(uniques),', ', max(uniques))
            # exit()

        # if i%1000 == 999:
        #     print(f'sample processed{i}')
        #     print(len(uniques))

        mask = sample['label']
        converted_mask = convertFromADE(mask)

        cv2.imwrite(save_path, converted_mask)
        

        uniques_current = list(np.unique(converted_mask))
        uniques.extend(uniques_current)
        uniques = list(set(uniques))
        if i % 500 == 499:
            print('shape of converted mask: ',converted_mask.shape)
            
        
        # print('current #unique_vals at ',i, '=',len(uniques_current), uniques_current)
    #     # print(mask.shape)
    #     # exit()
    #     uniques_current = np.unique(mask)
    #     uniques_current = list(uniques_current)
    #     print('current #unique_vals at ',i, '=',len(uniques_current))
    #     uniques.extend(uniques_current)
    #     uniques = list(set(uniques))
    #     # # print('#################### \n')
    #     # if uniques is None:
    #     #     uniques = uniques_current
    #     # else:
    #     #     uniques_set = set(uniques)
    #     #     uniques_current_set = set(uniques_current)

    #     #     uniques_set.

    #     #     # uniques_set = set(map(tuple, uniques))
    #     #     # uniques_current_set = set(map(tuple, uniques_current))
    #     #     # union_set = uniques_set.union(uniques_current_set)
    #     #     # uniques = [list(sublist) for sublist in union_set]

        
    # unique_colors_dataset_val = []
    # for i in range(len(val_dataset)):
    #     sample = val_dataset.__getitem__(i)
    #     if i%1000 == 0:
    #         print(f'sample processed{i}')

    #     mask = sample['label']
    #     uniques, _ = np.unique(mask, return_counts=True)
    #     unique_colors_dataset_val.extend(list(uniques))
    # print(unique_colors_dataset_val, len(unique_colors_dataset_val))

    
