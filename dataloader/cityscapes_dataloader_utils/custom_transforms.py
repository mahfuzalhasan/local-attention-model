import torch
import random
import numpy as np
import torchvision
import skimage

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0., 0.), std=(1., 1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        img = sample['image']
        img = np.array(img).astype(np.float32)

        img /= 255.0
        img -= self.mean[:3]
        img /= self.std[:3]

        mask = sample['label']
        mask = np.array(mask).astype(np.float32)
        # print('################label in Augmentation###########')
        # print(np.unique(mask))
        # print('################################################')

        depth = sample['depth']
        if not isinstance(depth, list):
            depth = np.array(depth).astype(np.float32)
            depth /= 255.0
            depth -= self.mean[3:]
            depth /= self.std[3:]

        return {'image': img,
                'depth': depth,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        mask = sample['label']
        mask = np.array(mask).astype(np.float32)
        mask = torch.from_numpy(mask).float()

        depth = sample['depth']
        if not isinstance(depth, list):
            depth = np.array(depth).astype(np.float32)
            if len(depth.shape) == 3:
                depth = depth.transpose((2, 0, 1))
            depth = torch.from_numpy(depth).float()

        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        coin_flip = random.random()
        mask = sample['label']
        depth = sample['depth']
        img = sample['image']

        if coin_flip < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if not isinstance(depth, list):
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        rotate_degree = random.uniform(-1*self.degree, self.degree)

        img = sample['image']
        img = img.rotate(rotate_degree, Image.BILINEAR)

        mask = sample['label']
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        depth = sample['depth']
        if not isinstance(depth, list):
             depth = depth.rotate(rotate_degree, Image.BILINEAR)

        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        coin_flip = random.random()
        rand_radius = random.random()
        mask = sample['label']

        img = sample['image']
        depth = sample['depth']
        if coin_flip < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=rand_radius))
            if not isinstance(depth, list):
                depth = depth.filter(ImageFilter.GaussianBlur(
                radius=rand_radius))

        return {'image': img,
                'depth': depth,
                'label': mask}
# class RandomResizeCrop(object):
#     def __init__(self, img_size, scale, ratio):
#         self.img_size = img_size
#         self.scale = scale
#         self.ratio = ratio
#         self.crop_transform = transforms.RandomResizedCrop(size=(300, 600))

#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         depth = sample['depth']




class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        depth = sample['depth']

        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        #### [256, 1024]
        
        w, h = img.size #[2048, 1024]

        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size     
            ow = int(1.0 * w * oh / h)  
        mask = mask.resize((ow, oh), Image.NEAREST) 
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w_resize, h_resize = mask.size      #2400, 1200
        x1 = random.randint(0, w_resize - self.crop_size) # 1376
        y1 = random.randint(0, h_resize - self.crop_size)   # 176

        #### (1376, 176, 2400, 1200) --> 1024 x 1024
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        img = img.resize((ow, oh), Image.BILINEAR)
        if short_size < self.crop_size:
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        if not isinstance(depth, list):
            depth = depth.resize((ow, oh), Image.BILINEAR)
            if short_size < self.crop_size:
                depth = ImageOps.expand(depth, border=(0, 0, padw, padh), fill=0)
            depth = depth.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'depth': depth,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):

        img = sample['image']
        mask = sample['label']      --> 2048, 1024
        depth = sample['depth']

        w, h = img.size
        if w > h:
            oh = self.crop_size     #1024
            ow = int(1.0 * w * oh / h) #2048
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)

        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w_resize, h_resize = mask.size
        x1 = int(round((w_resize - self.crop_size) / 2.))       #[512, 0 ]
        y1 = int(round((h_resize - self.crop_size) / 2.))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        img = img.resize((ow, oh), Image.BILINEAR)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        if not isinstance(depth, list):
            depth = depth.resize((ow, oh), Image.BILINEAR)
            depth = depth.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'depth': depth,
                'label': mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        mask = sample['label']
        depth = sample['depth']

        img = sample['image']
        assert img.size == mask.size
        img = img.resize(self.size, Image.BILINEAR)

        mask = mask.resize(self.size, Image.NEAREST)

        if not isinstance(depth, list):
            depth = depth.resize(self.size, Image.BILINEAR)

        return {'image': img,
                'depth': depth,
                'label': mask}

# Based on method from
#  Structure-Revealing Low-Light Image EnhancementVia Robust Retinex Model.
#          Li et al. Transactions on Image Processing, 2018.
# "We synthesize low-light images by  first  applying Gamma correction(withγ=2.2)  (...)
#  and  then  adding Poisson noise and white Gaussian noise to Gamma corrected images.
#  In our work, we use the built-in function of MATLAB imnoise to generate Poisson noise.
#  For Gaussian noise, we use σ=5  to  simulate  the  noise  level  in  most  natural  low-light images."
class RandomDarken(object):
    def __init__(self, cfg, darken):
        self.darken = darken
        self.gaussian_var = cfg.DATASET.DARKEN.GAUSSIAN_SIGMA*cfg.DATASET.DARKEN.GAUSSIAN_SIGMA
        self.poisson = cfg.DATASET.DARKEN.POISSON

    def __call__(self, sample):
        mask = sample['label']
        depth = sample['depth']
        img = sample['image']

        if self.darken:
            # Darken Image
            gamma = 1.0 + random.random() * 1.2
            gain = 1 - random.random() / 2.0
            img = torchvision.transforms.functional.adjust_gamma(img, gamma, gain)

            # Add noise
            img_arr = np.array(img).astype(np.float32) / 255.

            # Shot noise, proportional to number of photons measured
            if self.poisson:
                img_arr = skimage.util.random_noise(img_arr, mode='poisson', clip=True)
            # Temperature noise, constant for sensor at temperature
            if self.gaussian_var > 0:
                img_arr = skimage.util.random_noise(img_arr, mode='gaussian', var=self.gaussian_var, clip=True)
            img = Image.fromarray(np.uint8(img_arr * 255.))

        return {'image': img,
                'depth': depth,
                'label': mask}

class Darken(object):
    def __init__(self, cfg):
        #, gamma=2.0, gain=0.5, gaussian_m = 5./255.
        self.darken = cfg.DATASET.DARKEN.DARKEN  # size: (h, w)
        self.gamma = cfg.DATASET.DARKEN.GAMMA
        self.gain = cfg.DATASET.DARKEN.GAIN
        self.gaussian_var = cfg.DATASET.DARKEN.GAUSSIAN_SIGMA * cfg.DATASET.DARKEN.GAUSSIAN_SIGMA
        self.poisson = cfg.DATASET.DARKEN.POISSON

    def __call__(self, sample):
        mask = sample['label']
        depth = sample['depth']
        img = sample['image']

        if self.darken:
            # Darken Image
            img = torchvision.transforms.functional.adjust_gamma(img, self.gamma, self.gain)

            img_arr = np.array(img).astype(np.float32) / 255.

            # Add noise
            # Shot noise, proportional to number of photons measured
            if self.poisson:
                img_arr = skimage.util.random_noise(img_arr, mode='poisson', clip=True)
            # Temperature noise, constant for sensor at temperature
            if self.gaussian_var > 0:
                img_arr = skimage.util.random_noise(img_arr, mode='gaussian', var=self.gaussian_var, clip=True)

            img = Image.fromarray(np.uint8(img_arr * 255.))

        return {'image': img,
                'depth': depth,
                'label': mask}

## Reverses gamma correction and gain to show the effect of adding noise to the image. Noise is not-reversed
class UnDarken(object):
    def __init__(self, cfg):
        #, gamma=2.0, gain=0.5, gaussian_m = 5./255.
        self.darken = cfg.DATASET.DARKEN.DARKEN  # size: (h, w)
        self.gamma = 1.0 / cfg.DATASET.DARKEN.GAMMA # To reverse gamma correction, take the gamma root

        if cfg.DATASET.DARKEN.GAIN == 0:
            self.gain = 0.0 #No way to reverse
        else:
            self.gain = 1.0 / cfg.DATASET.DARKEN.GAIN # To reverse gain, multiply by the inverse

    def __call__(self, sample):
        mask = sample['label']
        depth = sample['depth']
        img = sample['image']

        if self.darken:
            # Darken Image
            img = torchvision.transforms.functional.adjust_gamma(img, self.gamma, self.gain)
            img = Image.fromarray(np.uint8(img))

        return {'image': img,
                'depth': depth,
                'label': mask}