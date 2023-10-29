"""Pascal ADE20K Semantic Segmentation Dataset."""
import os
import torch
import numpy as np

from PIL import Image
import cv2
# import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.segbase import SegmentationDataset

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

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        pixels = set(list(mask.getdata()))
        print('unique in original mask: ',pixels)
        
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and to Tensor
        # pixels = set(list(mask.getdata()))
        print('unique in original mask after crop: ',torch.unique(mask))
        if self.transform is not None:
            img = self.transform(img)
        sample = {}
        sample['image'] = img
        sample['label'] = mask
        sample['id'] = os.path.basename(self.images[index])
        return sample

    def _mask_transform(self, mask):
        print("it's coming home")
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
        
        '''
        -- image path should be like this: ../data/ade20k/images/training/**/**/ADE_train_00000001.jpg
        -- mask path should be like this: ../data/ade20k/images/training/**/**/ADE_train_00000001*seg.png
        - file_path should be like this: ../data/ade20k/images/training/**/**/
        '''
        
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
    
