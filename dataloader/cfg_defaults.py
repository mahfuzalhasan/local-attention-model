from yacs.config import CfgNode as CN
import torch

_C = CN()

_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
_C.SYSTEM.GPU_IDS = [0]
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 4
# Disable CUDA
_C.SYSTEM.NO_CUDA = False
# Random Seed
_C.SYSTEM.SEED = 1

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 1
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1

_C.DATASET = CN()
# ['pascal', 'coco', 'cityscapes', 'sunrgbd']
_C.DATASET.NAME = 'coco'
_C.DATASET.N_CLASSES = 81
# Root directory of dataset
_C.DATASET.ROOT = 'datasets/coco/'
# whether to use SBD dataset
_C.DATASET.USE_STD = True
# Base image size
_C.DATASET.BASE_SIZE = 513
# Crop image size
_C.DATASET.CROP_SIZE = 513
# Use RGB-D input
_C.DATASET.MODE = 'RGBD' #['RGBD', 'RGB', 'RGB_HHA']
_C.DATASET.SYNTHETIC = False
# What kind of annotations to use
_C.DATASET.ANNOTATION_TYPE = 'semantic' #['instance', 'bbox']
# Don't use any transformations including normalization. This flag is used for data statistics.
_C.DATASET.NO_TRANSFORMS = False
# Only use normalization. This flag is used to write normalized depth images to uint8 files.
_C.DATASET.NORMALIZE_ONLY = False

# Artifially darken input images
_C.DATASET.DARKEN = CN()
_C.DATASET.DARKEN.DARKEN = False
# Parameters for darkening filter include gamma correction, gain, and gaussian noise
_C.DATASET.DARKEN.GAMMA = 2.0
_C.DATASET.DARKEN.GAIN = 0.5
_C.DATASET.DARKEN.GAUSSIAN_SIGMA = 0.01
_C.DATASET.DARKEN.POISSON = True

# Use Box-Cox Transform on Depth Data
_C.DATASET.POWER_TRANSFORM = False
# Box-Cox Lambda
_C.DATASET.PT_LAMBDA = -0.5

# Variables specific to coco loader
_C.DATASET.COCO = CN()
_C.DATASET.COCO.CATEGORIES = 'coco' #['coco', 'pascal', 'sunrgbd']

# Variables specific to cityscapes loader
_C.DATASET.CITYSCAPES = CN()
_C.DATASET.CITYSCAPES.GT_MODE = 'gtFine' #['gtCoarse', 'gtFine']
_C.DATASET.CITYSCAPES.TRAIN_SET = 'train_extra' #['train_extra', 'train']
_C.DATASET.CITYSCAPES.DEPTH_DIR = 'disparity' #['disparity', 'VNL_Monocular', 'HHA', 'completed_depth']

# Mixes the labels and images randomly to generate baseline educated guess about the pixel classes
# i.e. road at the bottom, sky at the top
_C.DATASET.SCRAMBLE_LABELS = False

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  C = _C.clone()

  C.SYSTEM.CUDA = not C.SYSTEM.NO_CUDA and torch.cuda.is_available()
  return C