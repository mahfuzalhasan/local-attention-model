import os
import sys

import shutil

import numpy as np

data_path = './data/ade20k/images'

save_path = './label_ade20k'

for split in os.listdir(data_path):
    for section in os.listdir(os.path.join(data_path, split)):
        for subsec in os.listdir(os.path.join(data_path, split, section)):
            images = os.path.join(data_path, split, section, subsec)
            for files in os.listdir(images):
                if 'converted' in files:
                    label_path = os.path.join(images, files)
                    destination_dir = os.path.join(save_path, images)
                    if not os.path.exists(destination_dir):
                        os.makedirs(destination_dir)
                    shutil.copy(label_path, destination_dir)


