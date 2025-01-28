#%% Imports
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torchvision.transforms.functional as transF
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torchviz import make_dot
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth = 160)
np.set_printoptions(linewidth = 160)
np.set_printoptions(precision=4)
np.set_printoptions(suppress='True')

from utils_custom_tvision_functions import get_dataloaders, get_dataloaders_aligned, plot_curve, plot_curve_multiple
from AVS_config_M1_celeba import AVS_config as AVS_config_for_M1
from AVS_config_M2 import AVS_config as AVS_config_for_M2
from AVS_config_M3_celeba import AVS_config as AVS_config_for_M3
from AVS_model_M1_vgg import customizable_VGG as VGG_for_M1
from AVS_model_M2 import context_network
from AVS_model_M3_vgg import customizable_VGG as VGG_for_M3
from utils_custom_tvision_functions import imshow, plotregions, plotspots, plotspots_at_regioncenters, region_iou, region_area
# LSTM
from config_lstm import AVS_config as config_for_LSTM
from model_lstm import customizable_LSTM as LSTM
from weight_init import weight_init
# Debug
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import pdb
# PiCIE-related
from commons import *
import argparse
from modules import fpn
import seaborn as sns
import pickle

#%% Plot configs
sns.set_style('whitegrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=24, titleweight='bold', labelweight='bold', edgecolor='black')     # fontsize of the axes title
plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=22)    # fontsize of the tick labels
plt.rc('ytick', labelsize=22)    # fontsize of the tick labels
plt.rc('legend', fontsize=22, edgecolor='black')    # legend fontsize
plt.rc('font', size=22, weight='bold')          # controls default text sizes
plt.rc('lines', linewidth=4)     # linewidth
plt.rc('figure', figsize=(14,8), dpi=300, frameon=True) # figsize
plt.rcParams.update({'xtick.labelsize': 16})
plt.rcParams.update({'ytick.labelsize': 16})


#%% Instantiate parameters, dataloaders, and model
# Parameters
config_1 = AVS_config_for_M1
config_2 = AVS_config_for_M2
config_3 = AVS_config_for_M3
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda")

# SEED instantiation
SEED = config_3.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
#random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Dataloaders
train_loader, loss_weights    = get_dataloaders_aligned(config_3, loader_type=config_3.train_loader_type) # train (162770)/trainval split
valid_loader                  = get_dataloaders_aligned(config_3, loader_type=config_3.valid_loader_type) # val (19867)/test (19962) split


#%% Swap patches functionality
# write the code for cropping patches around landmarks and swapping patches
def landmark_shuffle(img_tensor, landmarks, num_distortion, patch_size_half, scaling_factor):
    landmark_indices = random.sample(range(5), num_distortion) # make sure the indices are unique to each other
    landmark_1, landmark_2 = landmarks[landmark_indices] 
    # swap patches
    scaling_factor = 1
    # mind the image borders, and overlaps. if the crop region hits the border, resize the crop regions when replacing
    patch_1_ymin = max(0, math.floor(landmark_1[1].item() - patch_size_half*scaling_factor)) # floor and ceil so that max-min != 0
    patch_1_ymax = min(255, math.ceil(landmark_1[1].item() + patch_size_half*scaling_factor))
    patch_1_xmin = max(0, math.floor(landmark_1[0].item() - patch_size_half*scaling_factor))
    patch_1_xmax = min(255, math.ceil(landmark_1[0].item() + patch_size_half*scaling_factor))

    patch_2_ymin = max(0, math.floor(landmark_2[1].item() - patch_size_half*scaling_factor))
    patch_2_ymax = min(255, math.ceil(landmark_2[1].item() + patch_size_half*scaling_factor))
    patch_2_xmin = max(0, math.floor(landmark_2[0].item() - patch_size_half*scaling_factor))
    patch_2_xmax = min(255, math.ceil(landmark_2[0].item() + patch_size_half*scaling_factor))

    # bi-linear interpolation to resize the crop region, can add anti-alias to make the output for PIL images and tensors closer
    patch_1 = transF.resized_crop(img_tensor.clone(), patch_1_ymin, patch_1_xmin, 
                patch_1_ymax - patch_1_ymin, patch_1_xmax - patch_1_xmin, 
                (patch_2_ymax - patch_2_ymin, patch_2_xmax - patch_2_xmin)) 

    patch_2 = transF.resized_crop(img_tensor.clone(), patch_2_ymin, patch_2_xmin,
                patch_2_ymax - patch_2_ymin, patch_2_xmax - patch_2_xmin, 
                (patch_1_ymax - patch_1_ymin, patch_1_xmax - patch_1_xmin))

    img_tensor[:, patch_2_ymin:patch_2_ymax, patch_2_xmin:patch_2_xmax] = patch_1
    img_tensor[:, patch_1_ymin:patch_1_ymax, patch_1_xmin:patch_1_xmax] = patch_2 

    return img_tensor


#%% Corrupt half of all images. 
# method 1: within each batch, pick half of batch indices and corrupt corresponding images. save the corrupt indices in a dictionary. save a dictionary that serves as label for correct and corrupte images as well.
label_dict = {}

for i, (indices, images, landmarks) in enumerate(valid_loader):
    # visualization
    # save_dir = "/home/nano01/a/tao88/3.6"
    # imshow(images[0])
    # plotspots(landmarks[0], color='r') # landmarks correct
    # plt.savefig(os.path.join(save_dir, 'img_{}.png'.format(0)))

    batch_size = indices.shape[0]
    selected_indices = random.sample(indices.tolist(), batch_size // 2)

    # if we want to corrupt
    for img_idx in range(batch_size // 2):
        corrupted_image = landmark_shuffle(images[img_idx], 
                                           landmarks[img_idx], 
                                           config_3.num_distortion,
                                           config_3.patch_size_half,
                                           config_3.scaling_factor)
        # visualization
        # imshow(corrupted_image)
        # plotspots(landmarks[0], color='r')
        # plt.savefig(os.path.join(save_dir, 'img_{}_corrupted.png'.format(0)))
        # pdb.set_trace()
        label_dict[indices[img_idx].item()] = 1 # 1 for corrupted (fake)
        save_image(corrupted_image, '/home/nano01/a/tao88/celebA_raw/valset_corrupted/imgs/img_{}.jpg'.format(indices[img_idx]))
    
    # if we don't want to corrupt
    for img_idx in range(batch_size // 2, batch_size):
        label_dict[indices[img_idx].item()] = 0 # 0 for correct (true)
        save_image(images[img_idx], '/home/nano01/a/tao88/celebA_raw/valset_corrupted/imgs/img_{}.jpg'.format(indices[img_idx]))


# # save the dict as txt
with open('corrupt_lbl_dict.pkl', 'wb') as f:
    pickle.dump(label_dict, f)

# to load
# with open('/home/nano01/a/tao88/foveation_grammar_detection/corrupt_lbl_dict.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)
pdb.set_trace()

