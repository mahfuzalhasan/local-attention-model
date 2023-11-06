"""Pascal ADE20K Semantic Segmentation Dataset."""
import os
import torch
import numpy as np

from PIL import Image
import cv2
# import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segbase import SegmentationDataset

import torchvision
from torchvision import transforms


class ADE20KSegmentation(SegmentationDataset):
    """ADE20K Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to ADE20K folder. Default is './datasets/ade'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    """
    BASE_DIR = 'images'
    NUM_CLASS = 150

    def __init__(self, root='./data/ade20k', split='train', mode=None, transform=None, **kwargs):
        super(ADE20KSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        root = os.path.join(root, self.BASE_DIR)
        print(root)
        assert os.path.exists(root), "{root} does not exist. Please fix the dataset path."
        self.images, self.masks = _get_ade20k_pairs(root, split) # TODO: fix annotation path for our setup . What does this return? ->
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        print('Found {} images in the folder {}'.format(len(self.images), root))
        if transform is None:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),      # PIL --> 0-1
                    transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
                ])
        self.palette = self.get_palette()
        self.palette_class_idx_list = self.map_palette_to_class_idx()
        self.index_mapping = np.loadtxt('/home/UFAD/mdmahfuzalhasan/Documents/Projects/local-attention-model/data/ade20k/mapFromADE.txt')
        
    def convert(self, lab, index_mapping):
        # Create an output array with the same shape as lab
        lab_out = np.zeros_like(lab)
        
        # Iterate over the unique values in lab
        unique_values = np.unique(lab)
        print('unique_values', unique_values)
        for unique_value in unique_values:
            # Find the corresponding index in index_mapping
            index = np.where(index_mapping[:, 0] == unique_value)
            if len(index[0]) > 0:
                index = index[0][0]
                mapped_value = index_mapping[index, 1]
                lab_out[lab == unique_value] = mapped_value
        print('lab_out unique_values', np.unique(lab_out))
        return lab_out


    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        
        # mask = cv2.imread(self.masks[index],  cv2.IMREAD_GRAYSCALE)
        mask = np.array(Image.open(self.masks[index]))
        # mask = self.convert(mask, self.index_mapping)
        # print(f'initial mask:{mask.shape} unique:{np.unique(mask)}')
        # exit()
        # mask = Image.open(self.masks[index])
        # mask = np.array(mask)

        # pixels = set(list(mask.getdata()))
        # print('unique in original mask: ',pixels)
        
        # # synchrosized transform
        # if self.mode == 'train':
        #     img, mask = self._sync_transform(img, mask)
        # elif self.mode == 'val':
        #     img, mask = self._val_sync_transform(img, mask)
        # else:
        #     assert self.mode == 'testval'
        #     img, mask = self._img_transform(img), self._mask_transform(mask)
        # # general resize, normalize and to Tensor
        # # pixels = set(list(mask.getdata()))
        # print('img size: ',img.size, type(img))
        # print('mask size: ',mask.shape, type(mask))
        # # print('unique in original mask after crop: ',torch.unique(mask, dim=2))
        # if self.transform is not None:
        #     img = self.transform(img)
        sample = {}
        sample['image'] = img
        sample['label'] = mask
        sample['id'] = self.masks[index]
        return sample


    def map_palette_to_class_idx(self):
        '''
        Returns a list of [1-decremented 3D palette, 1D class_idx] pairs.
        Decrement of 1 implies incorporating the background. We ignore it as background in later processing.
        '''
        # class_idx = [i for i in range(1, 151)]
        # assert len(palette) == len(class_idx)
        palette_class_idx_list = [[[0, 0, 0], 0]]
        
        for i,p in enumerate(self.palette):
            palette_class_idx_list.append([p, i+1])

        # print(palette_class_idx_list, len(palette_class_idx_list))
        return palette_class_idx_list


    
    # label_rgb --> PIL
    def map_label_rgb_to_class_idx(self, label_rgb):
        '''iterate label_rgb and map to class_idx'''
        print('label pil: ',label_rgb.size, type(label_rgb))
        pixels = set(list(label_rgb.getdata()))
        print('unique values in final mask: ', pixels)
        label_rgb = np.array(label_rgb, dtype=np.uint8)
        print('rgb unique data numpy: ',np.unique(label_rgb))
        class_idx = np.zeros(label_rgb.shape, dtype=np.uint8)  # Initialize with zeros
        # print(class_idx.shape)
        # exit()
        self.palette.insert(0, [0, 0, 0])
        self.palette = [tuple(i) for i in self.palette]
        # self.palette = np.asarray(self.palette)
        # print(self.palette.shape)
        # exit()
        # self.palette_class_idx_list = np.asarray(self.palette_class_idx_list, dtype=np.uint8)
        # print(np.asarray(self.palette_class_idx_list).shape)
        # exit()

        for i, p in enumerate(self.palette):
            print("pixel to check: ",p)
            indices = np.where(label_rgb==p)
            print('matched value: ',label_rgb[indices])
            for index in indices:
                
                print('index: ',index)
            exit()
            # print('index: ',indices[0].shape)

            class_idx[indices] = i            
            # print(f'class:{i} sum:{torch.sum(color_mask)} shape:{color_mask.shape}')
            # exit()
            # # color_mask = torch.all(color_mask, dim=-1)  # [512, 512] # TODO: what is happening here ?? This is most likely the reason the code is not working
            # class_idx[color_mask] = i
        print(f'unique in mask final overall: {np.unique(class_idx)}')
        print(f'unique in mask final in channel 0: {np.unique(class_idx[:, :, 0])}')
        print(f'unique in mask final in channel 1: {np.unique(class_idx[:, :, 1])}')
        print(f'unique in mask final in channel 2: {np.unique(class_idx[:, :, 2])}')
        exit()
        return class_idx


    # input mask --> PIL -->  512, 512, 3
    def _mask_transform(self, mask):
        # print("it's coming home")
        # pixels = set(list(mask.getdata()))
        # print('unique in original mask after crop: ',pixels)
        # palette = self.get_palette()
        # palette.append([0, 0, 0])

        # print(len(palette))

        # self.map_label_rgb_to_class_idx(mask)
        # exit()
        return torch.LongTensor(np.array(mask).astype('int32') - 1)

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    @property
    def classes(self):
        """Category names."""
        return ("wall", "building, edifice", "sky", "floor, flooring", "tree",
                "ceiling", "road, route", "bed", "windowpane, window", "grass",
                "cabinet", "sidewalk, pavement",
                "person, individual, someone, somebody, mortal, soul",
                "earth, ground", "door, double door", "table", "mountain, mount",
                "plant, flora, plant life", "curtain, drape, drapery, mantle, pall",
                "chair", "car, auto, automobile, machine, motorcar",
                "water", "painting, picture", "sofa, couch, lounge", "shelf",
                "house", "sea", "mirror", "rug, carpet, carpeting", "field", "armchair",
                "seat", "fence, fencing", "desk", "rock, stone", "wardrobe, closet, press",
                "lamp", "bathtub, bathing tub, bath, tub", "railing, rail", "cushion",
                "base, pedestal, stand", "box", "column, pillar", "signboard, sign",
                "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink",
                "skyscraper", "fireplace, hearth, open fireplace", "refrigerator, icebox",
                "grandstand, covered stand", "path", "stairs, steps", "runway",
                "case, display case, showcase, vitrine",
                "pool table, billiard table, snooker table", "pillow",
                "screen door, screen", "stairway, staircase", "river", "bridge, span",
                "bookcase", "blind, screen", "coffee table, cocktail table",
                "toilet, can, commode, crapper, pot, potty, stool, throne",
                "flower", "book", "hill", "bench", "countertop",
                "stove, kitchen stove, range, kitchen range, cooking stove",
                "palm, palm tree", "kitchen island",
                "computer, computing machine, computing device, data processor, "
                "electronic computer, information processing system",
                "swivel chair", "boat", "bar", "arcade machine",
                "hovel, hut, hutch, shack, shanty",
                "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, "
                "motorcoach, omnibus, passenger vehicle",
                "towel", "light, light source", "truck, motortruck", "tower",
                "chandelier, pendant, pendent", "awning, sunshade, sunblind",
                "streetlight, street lamp", "booth, cubicle, stall, kiosk",
                "television receiver, television, television set, tv, tv set, idiot "
                "box, boob tube, telly, goggle box",
                "airplane, aeroplane, plane", "dirt track",
                "apparel, wearing apparel, dress, clothes",
                "pole", "land, ground, soil",
                "bannister, banister, balustrade, balusters, handrail",
                "escalator, moving staircase, moving stairway",
                "ottoman, pouf, pouffe, puff, hassock",
                "bottle", "buffet, counter, sideboard",
                "poster, posting, placard, notice, bill, card",
                "stage", "van", "ship", "fountain",
                "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
                "canopy", "washer, automatic washer, washing machine",
                "plaything, toy", "swimming pool, swimming bath, natatorium",
                "stool", "barrel, cask", "basket, handbasket", "waterfall, falls",
                "tent, collapsible shelter", "bag", "minibike, motorbike", "cradle",
                "oven", "ball", "food, solid food", "step, stair", "tank, storage tank",
                "trade name, brand name, brand, marque", "microwave, microwave oven",
                "pot, flowerpot", "animal, animate being, beast, brute, creature, fauna",
                "bicycle, bike, wheel, cycle", "lake",
                "dishwasher, dish washer, dishwashing machine",
                "screen, silver screen, projection screen",
                "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase",
                "traffic light, traffic signal, stoplight", "tray",
                "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, "
                "dustbin, trash barrel, trash bin",
                "fan", "pier, wharf, wharfage, dock", "crt screen",
                "plate", "monitor, monitoring device", "bulletin board, notice board",
                "shower", "radiator", "glass, drinking glass", "clock", "flag")

    def get_palette(self):
        PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                [102, 255, 0], [92, 0, 255]]
        return PALETTE
def _get_ade20k_pairs(folder, mode='train'):
    img_paths = []
    mask_paths = []
    print('folder: ',folder)
    if mode == 'train':
        root_dir = os.path.join(folder, 'training')
        assert os.path.exists(root_dir), "{root_dir} does not exist. Please fix train path."
    else:
        root_dir = os.path.join(folder, 'validation')
        assert os.path.exists(root_dir), "{root_dir} does not exist. Please fix val path."
        
    for filename in os.listdir(root_dir):
        basename, _ = os.path.splitext(filename)
        file_path = os.path.join(root_dir, filename)
        for scene in os.listdir(file_path):
            scene_path = os.path.join(file_path, scene)
            for filename in os.listdir(scene_path):
                if filename.endswith(".jpg"):
                    imgpath = os.path.join(scene_path, filename)
                    basename, _ = os.path.splitext(imgpath)
                    maskpath = basename + '_seg.png'

                    if os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask:', maskpath)

        
    # print(f'img_paths: {len(img_paths)}')
    # print(f'mask_paths: {len(mask_paths)}')
    '''
    train img+mask pair count: 25574 
    val img+mask pair count: 2000 
    '''
    return img_paths, mask_paths


if __name__ == '__main__':
    # train_dataset = ADE20KSegmentation()
    root = "/home/UFAD/mdmahfuzalhasan/Documents/Projects/local-attention-model/data/ade20k"
    train_dataset = ADE20KSegmentation(root=root, split='train', mode='train')

    classes = train_dataset.classes

    print(len(classes))


    sample = train_dataset.__getitem__(1)
    img, mask, path = sample['image'], sample['label'], sample['id']
    print(f'img: {img.shape}')
    print(f'mask: {mask.shape}')
    print('unique values: ',torch.unique(mask), mask.size())

    mask = mask.detach().cpu().numpy()
    print('unique numpy: ',np.unique(mask))
    cv2.imwrite('mask.jpg', mask)

    

    

    # img = img.detach().cpu().numpy()
    # print('unique numpy img: ',np.unique(img))
    # cv2.imwrite('img.jpg', img)

    print(f'path: {path}')



    

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(img)
    # ax1.set_title('Image')
    # ax2.imshow(mask)
    # ax2.set_title('Mask')
    # plt.show()
    
