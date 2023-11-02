import numpy as np
from ade import ADE20KSegmentation
import pickle
import os

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
    # train_dataset = ADE20KSegmentation()
    root = "/home/UFAD/mdmahfuzalhasan/Documents/Projects/local-attention-model/data/ade20k"
    train_dataset = ADE20KSegmentation(root=root, split='train', mode='train')

    val_dataset = ADE20KSegmentation(root=root, split='val', mode='val')
    
    unique_colors_dataset = []
    uniques = None
    for i in range(len(train_dataset)):
        sample = train_dataset.__getitem__(i)
        if i % 100 == 0:
            print(f'sample processed{i}')

        if i%1000 == 999:
            print(f'sample processed{i}')
            print(len(uniques))

        mask = sample['label']
        print(mask.shape)
        exit()
        uniques_current = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
        uniques_current = list(uniques_current)
        print('current unique: ',i,uniques_current, len(uniques_current))
        # print('#################### \n')
        if uniques is None:
            uniques = uniques_current
        else:
            uniques_set = set(map(tuple, uniques))
            uniques_current_set = set(map(tuple, uniques_current))
            union_set = uniques_set.union(uniques_current_set)
            uniques = [list(sublist) for sublist in union_set]

        
    unique_colors_dataset_val = []
    for i in range(len(val_dataset)):
        sample = val_dataset.__getitem__(i)
        if i%1000 == 0:
            print(f'sample processed{i}')

        mask = sample['label']
        uniques, _ = np.unique(mask, return_counts=True)
        unique_colors_dataset_val.extend(list(uniques))
    print(unique_colors_dataset_val, len(unique_colors_dataset_val))

    
