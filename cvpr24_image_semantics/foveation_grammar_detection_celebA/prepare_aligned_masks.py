import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torchviz import make_dot
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

torch.set_printoptions(linewidth = 160)
np.set_printoptions(linewidth = 160)
np.set_printoptions(precision=4)
np.set_printoptions(suppress='True')

from utils_custom_tvision_functions import get_dataloaders, plot_curve
from AVS_config_M1_celeba import AVS_config as AVS_config_for_M1
from AVS_config_M2 import AVS_config as AVS_config_for_M2
from AVS_config_M3_celeba import AVS_config as AVS_config_for_M3
# from config_lstm import AVS_config as config_for_LSTM # LSTM configs
from AVS_model_M1_vgg import customizable_VGG as VGG_for_M1
from AVS_model_M2 import context_network
from AVS_model_M3_vgg import customizable_VGG as VGG_for_M3
# from model_lstm import customizable_LSTM as LSTM
from utils_custom_tvision_functions import imshow, plotregions, plotspots, plotspots_at_regioncenters, region_iou, region_area
import pdb
import random

# 1.24 - tao88
from torchvision.utils import save_image
from commons import *
from utils import *
import argparse
from torchPCA import PCA
import PIL
from torch.utils.tensorboard import SummaryWriter

#%% Instantiate parameters, dataloaders, and model
# Parameters
config_1 = AVS_config_for_M1
config_2 = AVS_config_for_M2
config_3 = AVS_config_for_M3
# config_4 = config_for_LSTM
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# SEED instantiation
SEED            = config_3.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
#random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Dataloaders
train_loader, loss_weights    = get_dataloaders(config_3, loader_type='trainval')
valid_loader                  = get_dataloaders(config_3, loader_type='test')
# test_loader                   = get_dataloaders(config_3, loader_type='test')

# Loss(es)
bce_loss                      = nn.BCEWithLogitsLoss(pos_weight=loss_weights.to(device))

# Model
model_1 = VGG_for_M1(config_1).to(device)
ckpt_1  = torch.load(config_3.ckpt_dir_model_M1)
model_1.load_state_dict(ckpt_1['model'])
for p in model_1.parameters():
    p.requires_grad_(False)
model_1.eval()
print("Model M1:\n", "Loaded from: {}\n".format(config_3.ckpt_dir_model_M1), model_1)

model_2 = context_network(in_channels=config_2.in_num_channels, res_size = config_2.low_res, avg_size = config_2.avg_size).to(device)
ckpt_2  = torch.load(config_3.ckpt_dir_model_M2)
model_2.load_state_dict(ckpt_2['model'])
for p in model_2.parameters():
    p.requires_grad_(False)
model_2.eval()
print("Model M2:\n", "Loaded from: {}\n".format(config_3.ckpt_dir_model_M2), model_2)

model_3 = VGG_for_M3(config_3).to(device)
print("Model M3:\n")
ckpt_3 = torch.load(config_3.ckpt_dir_model_M3)
model_3.load_state_dict(ckpt_3['model'])
for p in model_3.parameters():
    p.requires_grad_(False)
model_3.eval()
print("Loaded from: {}\n".format(config_3.ckpt_dir_model_M3))
print(model_3)


#%% 2.2 - tao88: import PiCIE models
# PiCIE arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--restart_path', type=str)
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed for reproducability.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers.')
    parser.add_argument('--restart', action='store_true', default=False)
    parser.add_argument('--num_epoch', type=int, default=10) 
    parser.add_argument('--repeats', type=int, default=0)  

    # Train. 
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--res', type=int, default=320, help='Input size.')
    parser.add_argument('--res1', type=int, default=320, help='Input size scale from.')
    parser.add_argument('--res2', type=int, default=640, help='Input size scale to.')
    parser.add_argument('--batch_size_cluster', type=int, default=256)
    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optim_type', type=str, default='Adam')
    parser.add_argument('--num_init_batches', type=int, default=30)
    parser.add_argument('--num_batches', type=int, default=30)
    parser.add_argument('--kmeans_n_iter', type=int, default=30)
    parser.add_argument('--in_dim', type=int, default=128)
    parser.add_argument('--X', type=int, default=80)

    # Loss. 
    parser.add_argument('--metric_train', type=str, default='cosine')   
    parser.add_argument('--metric_test', type=str, default='cosine')
    parser.add_argument('--K_train', type=int, default=27) # COCO Stuff-15 / COCO Thing-12 / COCO All-27
    parser.add_argument('--K_test', type=int, default=27) 
    parser.add_argument('--no_balance', action='store_true', default=False)
    parser.add_argument('--mse', action='store_true', default=False)

    # Dataset. 
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--equiv', action='store_true', default=False)
    parser.add_argument('--min_scale', type=float, default=0.5)
    parser.add_argument('--stuff', action='store_true', default=False)
    parser.add_argument('--thing', action='store_true', default=False)
    parser.add_argument('--jitter', action='store_true', default=False)
    parser.add_argument('--grey', action='store_true', default=False)
    parser.add_argument('--blur', action='store_true', default=False)
    parser.add_argument('--h_flip', action='store_true', default=False)
    parser.add_argument('--v_flip', action='store_true', default=False)
    parser.add_argument('--random_crop', action='store_true', default=False)
    parser.add_argument('--val_type', type=str, default='train')
    parser.add_argument('--version', type=int, default=7)
    parser.add_argument('--fullcoco', action='store_true', default=False)

    # Eval-only
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_path', type=str)

    # Cityscapes-specific.
    parser.add_argument('--cityscapes', action='store_true', default=False)
    parser.add_argument('--label_mode', type=str, default='gtFine')
    parser.add_argument('--long_image', action='store_true', default=False)

    # tao88 - Celeba-specific
    parser.add_argument('--celeba', action='store_true', default=False)
    parser.add_argument('--full_res_img_size', type=tuple, default=(256, 256)) # (height, width), like res1 and res2
    parser.add_argument('--num_classes', type=int, default=40)
    parser.add_argument('--selected_attributes', type=str, default='all')
    parser.add_argument('--correct_imbalance', action='store_true', default=False)
    parser.add_argument('--at_least_true_attributes', type=int, default=0)
    parser.add_argument('--treat_attributes_as_classes', action='store_true', default=False)
    parser.add_argument('--landmark_shuffle', action='store_true', default=False)

    # tao88 - clustering specifc
    parser.add_argument('--with_mask', action='store_true', default=False)
    # tao88 - 1.15
    parser.add_argument('--model_finetuned', action='store_true', default=False)
    parser.add_argument('--finetuned_model_path',type=str, default='')
    parser.add_argument('--FPN_with_classifier', action='store_true', default=False)
    
    return parser.parse_args()
args = parse_arguments()
# Setup the path to save.
if not args.pretrain:
    args.save_root += '/scratch'
# tao88
if args.with_mask:
    args.save_root += '/with_mask'
if args.augment:
    args.save_root += '/augmented/res1={}_res2={}/jitter={}_blur={}_grey={}'.format(args.res1, args.res2, args.jitter, args.blur, args.grey)
if args.equiv:
    args.save_root += '/equiv/h_flip={}_v_flip={}_crop={}/min_scale\={}'.format(args.h_flip, args.v_flip, args.random_crop, args.min_scale)
if args.no_balance:
    args.save_root += '/no_balance'
if args.mse:
    args.save_root += '/mse'

args.save_model_path = os.path.join(args.save_root, args.comment, 'K_train={}_{}'.format(args.K_train, args.metric_train))
args.save_eval_path  = os.path.join(args.save_model_path, 'K_test={}_{}'.format(args.K_test, args.metric_test))

def compute_dist(featmap, metric_function, euclidean_train=True):
    centroids = metric_function.module.weight.data
    if euclidean_train:
        return - (1 - 2*metric_function(featmap)\
                    + (centroids*centroids).sum(dim=1).unsqueeze(0)) # negative l2 squared 
    else:
        return metric_function(featmap)

if not os.path.exists(args.save_eval_path):
    os.makedirs(args.save_eval_path)
logger = set_logger(os.path.join(args.save_eval_path, 'train.log'))
# Get PiCIE model and optimizer(centroids)
model_FPN, optimizer, classifier1 = get_model_and_optimizer_orig(args, logger)
# pdb.set_trace() # done, note dataparallel objects
# 2.2 - tao88: tensorboard
# writer = SummaryWriter(comment="visualize_image_0_glimpse_labels")


#%% AVS-specific functions, methods, and parameters
from AVS_functions import location_bounds, extract_and_resize_glimpses_for_batch

lower, upper = location_bounds(glimpse_w=config_3.glimpse_size_init[0], input_w=config_3.full_res_img_size[0]) #TODO: make sure this works on both x and y coordinates


#%% evaluate the masks of trainval aligned celeba images using trained PiCIE model and centroids, and re-scale to raw image masks
# create a dictionary to record region_idx-region_name correspondances
region_idx_to_name_dict =  {0: 'hair',
                            1: 'neck_edge',
                            2: 'eye_forehead_edge',
                            3: 'forehead_down',
                            4: 'background',
                            5: 'forehead_up',
                            6: 'background',
                            7: 'nose_eye_edge',
                            8: 'mouth',
                            9: 'hair',
                            10: 'hair',
                            11: 'background',
                            12: 'neck',
                            13: 'background',
                            14: 'hair',
                            15: 'background',
                            16: 'eyes',
                            17: 'nose',
                            18: 'background',
                            19: 'background'}

region_name_to_idx_after_transform = {'background': 0,
                                        'hair': 1,
                                        'forehead':2,
                                        'eyes':3,
                                        'nose':4,
                                        'mouth':5,
                                        'neck':6}

model_FPN.eval()
model_FPN.cuda()
classifier1.eval()
classifier1.cuda()


#%% Evaluate the aligned images for the masks
with torch.no_grad():
    for batch_idx, (index, images, targets) in enumerate(train_loader): # target_type=['attr', 'bbox', 'pred_init_glimpse_locs', 'landmarks', 'masks']
        # aligned_images
        translated_images, targets, bbox_targets = images.to(device), targets[0].float().to(device), targets[1].to(device)
        # translated images (128, 3, 256, 256); targets (128, 40); bbox_targets (128, 4)

        out = model_FPN(translated_images) # (128, 128, 64, 64)
        out = nnF.normalize(out, dim=1, p=2)
        # out = out / (out.norm(p=2, dim=1, keepdim=True)) 

        prb = compute_dist(out, classifier1)
        lbl = prb.topk(1, dim=1)[1]
        lbl = lbl.squeeze(1) # shape (128, 64, 64), range[0, 20)

        # now refine the seg map using the dict above, and then use the bbox to scale to the raw image seg mask
        new_lbl = torch.zeros_like(lbl)
        for region_idx in range(args.K_train):
            region_name = region_idx_to_name_dict[region_idx]
            region_coords = (lbl == region_idx).nonzero()
            # can use match-case instead
            if region_name == 'hair':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['hair']
            elif region_name == 'neck':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['neck']
            elif region_name == 'neck_edge':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['neck']
            elif region_name == 'eye_forehead_edge':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['forehead']
            elif region_name == 'forehead_down':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['forehead']
            elif region_name == 'forehead_up':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['forehead']
            elif region_name == 'background':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['background']
            elif region_name == 'eyes':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['eyes']
            elif region_name == 'nose':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['nose']
            elif region_name == 'nose_eye_edge':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['nose']
            elif region_name == 'mouth':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['mouth']
        
        # save all transformed labels in the designated label folder
        # new_lbl = new_lbl / len(region_name_to_idx_after_transform.keys()) # normalize to 0-1
        # verify visualization
        # save_image((translated_images[:16]), '/home/nano01/a/tao88/2.9/aligned_imgs.png')
        # save_image((new_lbl[:16]).unsqueeze(1), '/home/nano01/a/tao88/2.9/aligned_masks.png')
        # new_lbl_arr = np.uint8((new_lbl[0].cpu().numpy()) / 7 * 255)
        # new_lbl = PIL.Image.fromarray(new_lbl_arr, 'L').save("/home/nano01/a/tao88/2.23/lbl_0.png")
        # pdb.set_trace() # so far, the new_lbl has only 7 unique values

        # PIL.Image.from_array
        # Saving to designated folder
        new_lbl = new_lbl.cpu().numpy()
        new_lbl = np.uint8(new_lbl / 7 * 255)
        for i in range(new_lbl.shape[0]):
            # save_image(new_lbl[i], '/home/nano01/a/tao88/celebA_raw/Labels/labels_trainval_aligned/{}.png'.format(index[i])) # make sure the index is correct
            # 2.23.2023 - tao88, updated method:
            im = PIL.Image.fromarray(new_lbl[i], 'L') # 'L' means single channel, storing the luminance. Very compact, only grayscale
            im.save('/home/nano01/a/tao88/celebA_raw/Labels/labels_trainval_aligned/{}.png'.format(index[i]))            

        if (batch_idx+1) % 100 == 0:
            print("Processed: {}/{}".format(batch_idx + 1, len(train_loader)))

# save as a new model in a different path
pdb.set_trace()
