'''
Created on 2/6/2023
@tao88
Use modified Resnet+FPN so that it outputs pixel-wise features, finetune using the segmentation masks as supervision
'''
#%% Imports
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torchviz import make_dot
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth = 160)
np.set_printoptions(linewidth = 160)
np.set_printoptions(precision=4)
np.set_printoptions(suppress='True')

# M1-M3
from utils_custom_tvision_functions import get_dataloaders, plot_curve, plot_curve_multiple
from AVS_config_M1_celeba import AVS_config as AVS_config_for_M1
from AVS_config_M2 import AVS_config as AVS_config_for_M2
from AVS_config_M3_celeba import AVS_config as AVS_config_for_M3
from AVS_model_M1_vgg import customizable_VGG as VGG_for_M1
from AVS_model_M2 import context_network
from AVS_model_M3_vgg import customizable_VGG as VGG_for_M3

# LSTM
from config_lstm import AVS_config as config_for_LSTM
from model_lstm import customizable_LSTM as LSTM
from weight_init import weight_init

# Others
from utils_custom_tvision_functions import imshow, plotregions, plotspots, plotspots_at_regioncenters, region_iou, region_area
# debug
from torchvision.utils import save_image
import pdb
# PiCIE-related
from commons import *
import argparse
import matplotlib.pyplot as plt
from modules import fpn 
import seaborn as sns


#%% Plot configs
sns.set_style('whitegrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=24, titleweight='bold', labelweight='bold', edgecolor='black')     # fontsize of the axes title
plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=22)   # fontsize of the tick labels
plt.rc('ytick', labelsize=22)   # fontsize of the tick labels
plt.rc('legend', fontsize=22, edgecolor='black')    # legend fontsize
plt.rc('font', size=22, weight='bold')              # controls default text sizes
plt.rc('lines', linewidth=1)    # linewidth
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
SEED            = config_3.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
#random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# GPU Debug
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # used when using single card

# Dataloaders
# train_loader, loss_weights    = get_dataloaders(config_3, loader_type=config_3.train_loader_type)
valid_loader                  = get_dataloaders(config_3, loader_type=config_3.valid_loader_type)

# Loss(es)
# bce_loss                    = nn.BCEWithLogitsLoss(pos_weight=loss_weights.to(device))  
ce_loss                     = nn.CrossEntropyLoss()


#%% Resnet+FPN model for context learning in each glimpse
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducability.')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--FPN_with_classifier', action='store_true', default=False)

    parser.add_argument('--restart', action='store_true', default=False)
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_path', type=str, default='.')
    parser.add_argument('--model_finetuned', action='store_true', default=False)
    parser.add_argument('--finetuned_model_path',type=str, default='')

    parser.add_argument('--in_dim', type=int, default=128)
    parser.add_argument('--K_train', type=int, default=20)

    parser.add_argument('--num_epoch', type=int, default=10) 
    parser.add_argument('--optim_type', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    return parser.parse_args()

args = parse_arguments()
args.save_model_path = args.eval_path # 2.20.2023 - use 20 epochs model

# Set logger
logger = set_logger('./test_Resnet.log')

# Initialize model (import the finetuned model and load its trained weights)
# model_FPN, optimizer = get_model_and_optimizer(args, logger) # note dataparallel objects (fixed 2.13.2023)
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma = 0.1, milestones=[40, 80])

model_FPN = fpn.PanopticFPN(args)
if config_3.data_parallel:
    model_FPN = nn.DataParallel(model_FPN) # comment for using one device
model_FPN = model_FPN.cuda()
checkpoint  = torch.load(args.eval_path)
if config_3.data_parallel:
    model_FPN.load_state_dict(checkpoint['state_dict'])
else:
    # If model_FPN is not data-parallel
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    model_FPN.load_state_dict(new_state_dict)
print("Model FPN:\n", "Loaded from: {}\n".format(args.eval_path))
# logger.info('Loaded checkpoint. [epoch {}]'.format(checkpoint['epoch']))
for p in model_FPN.parameters():
    p.requires_grad_(False)
model_FPN.eval()
# pdb.set_trace() # Done, successfully loaded 20-epoch glimpse classification model


#%% LSTM-related
# LSTM configs
config_4 = config_for_LSTM
# LSTM model (M4)
model_4 = LSTM(config_4)
if config_3.data_parallel:
    model_4 = nn.DataParallel(model_4)
model_4.cuda()
# model_4.apply(weight_init)
ckpt_4 = torch.load(config_4.ckpt_dir_model_M4)
model_4.load_state_dict(ckpt_4['state_dict'])
for p in model_4.parameters():
    p.requires_grad_(False)
model_4.eval()
print("Model M4:\n", "Loaded from: {}\n".format(config_4.ckpt_dir_model_M4))
print(model_4)
# pdb.set_trace() # Done, imported all models (M1-M3, FPN, M4)


#%% AVS-specific functions, methods, and parameters
from AVS_functions import location_bounds, extract_and_resize_glimpses_for_batch, crop_five

lower, upper = location_bounds(glimpse_w=config_3.glimpse_size_init[0], input_w=config_3.full_res_img_size[0]) #TODO: make sure this works on both x and y coordinates


#%% Test on the models
with torch.no_grad():
    for i, (index, images, targets) in enumerate(valid_loader): # target_type=['attr', 'bbox', 'pred_init_glimpse_locs', 'landmarks', 'masks']
        if i == len(valid_loader) - 1:
            continue # 2.15.2023 - tao88, in order to account for error in compute_rl_loss
        translated_images, targets, bbox_targets, corrupt_labels = images.to(device), targets[0].float().to(device), targets[1].to(device), targets[-1].to(device) # translated images (128, 3, 256, 256); targets (128, 40); bbox_targets (128, 4); corrupt_labels (128), 0 for correct, 1 for corrnano01upted
        # masks = torch.round(masks * 7).long() # preprocess the aligned masks

        # 3.12.2023 - tao88: visualize batch of images before crop
        if i == 0:
            for img_id in range(16):
                if not config_3.landmark_shuffle:
                    if not config_3.use_corrupted_testset:
                        save_path = '/home/nano01/a/tao88/3.21_fivepatch/correct/img_{}'.format(img_id)
                    else:
                        save_path = '/home/nano01/a/tao88/3.21_fivepatch/corrupted_testset/img_{}'.format(img_id)
                else:
                    save_path = '/home/nano01/a/tao88/3.21_fivepatch/corrupted/img_{}'.format(img_id)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_image(translated_images[img_id], os.path.join(save_path, 'input_img.png'))
        
        # pdb.set_trace()

        # 3.10.2023 - tao88: crop the 5 patches around center of the input image
        batch_patches = crop_five(translated_images, 
                                  left_coords=[64,128,64,128,96], 
                                  top_coords=[64,64,128,128,96], 
                                  widths=[64,64,64,64,64], 
                                  heights=[64,64,64,64,64],              
                                  resized_height=256, 
                                  resized_width=256) # (128, 5, 3, 256, 256)

        # pass each of the patches through the FPN network to predict masks, and collect the predicted masks in batch_masks
        for patch_id in range(5):
            patches = batch_patches[:, patch_id, :, :] # (128, 3, 256, 256)
            masks_predicted = model_FPN(patches) # (128, 7, 64, 64)
            lbl_predicted = masks_predicted.topk(1, dim=1)[1] # (128, 1, 64, 64)
            if patch_id == 0:
                batch_masks = lbl_predicted
            else:
                batch_masks = torch.cat((batch_masks, lbl_predicted), dim=1)
        
            # 3.13.2023 - visualize the patches and patch predictions
            if i == 0:
                for img_id in range(16):
                    if not config_3.landmark_shuffle:
                        if not config_3.use_corrupted_testset:
                            save_path = '/home/nano01/a/tao88/3.21_fivepatch/correct/img_{}'.format(img_id)
                        else:
                            save_path = '/home/nano01/a/tao88/3.21_fivepatch/corrupted_testset/img_{}'.format(img_id)
                    else:
                        save_path = '/home/nano01/a/tao88/3.21_fivepatch/corrupted/img_{}'.format(img_id)
                    if not os.path.exists(save_path): os.makedirs(save_path)
                    # patches
                    save_image(patches[img_id], os.path.join(save_path, 'img_{}_patch_{}.png'.format(img_id, patch_id)))
                    # patch masks
                    fig = plt.figure(figsize=(1,1))
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(lbl_predicted[img_id].clone().detach().squeeze(0).cpu().numpy(),
                                interpolation='nearest',
                                cmap='Paired',
                                vmin=0,
                                vmax=6)
                    plt.savefig(os.path.join(save_path, 'img_{}_patch_{}_mask.png'.format(img_id, patch_id)))
                    plt.close(fig)

        # batch masks now has shape (128, 5, 64, 64), process the masks into semantics
        num_pixels = batch_masks.shape[2] * lbl_predicted.shape[3]
        batch_masks_flattened = batch_masks.flatten(start_dim=2, end_dim=3) # (128, 5, 4096)
        mask_semantics = torch.nn.functional.one_hot(batch_masks_flattened, num_classes=7).sum(dim=2) # (128, 5, 7)
        mask_semantics_ratio = (mask_semantics.float()) / num_pixels # (128, 5, 7), normalize to ratio by dividing by 4096

        # pass mask_semantics_ratio to LSTM, sequence length is 5
        h0, c0 = model_4._init_hidden(translated_images.shape[0])
        output_lstm, (hn, cn) = model_4(mask_semantics_ratio, (h0, c0))
        
        # record dist curves for each image in dist_batch_imgs
        dist_batch_imgs = torch.norm((output_lstm[:, :-1, :] - mask_semantics_ratio[:, 1:, :]), dim=2, keepdim=False) # (128, 4)
        dist_batch_imgs = dist_batch_imgs.cpu().numpy()

        # visualize the distance curves for indivisual images
        for sample_id in range(16):
            if not config_3.landmark_shuffle:
                if not config_3.use_corrupted_testset:
                    save_path = '/home/nano01/a/tao88/3.21_fivepatch/correct/img_{}'.format(sample_id)
                else:
                    save_path = '/home/nano01/a/tao88/3.21_fivepatch/corrupted_testset/img_{}'.format(sample_id)
            else:
                save_path = '/home/nano01/a/tao88/3.21_fivepatch/corrupted/img_{}'.format(sample_id)
            plot_curve(np.arange(1, 5), dist_batch_imgs[sample_id], 'Dist btw. prediction and glimpse semantics', 'Glimpse iteration', 'L2 distance', os.path.join(save_path, 'plot_dist.png'))

        # 3.8.2023 - tao88: try first plotting together the 16 curves of the 1st batch, blue for correct, red for corrupted
        x, y = [np.arange(1, 5) for _ in range(translated_images.shape[0])], dist_batch_imgs
        colors = ['b', 'r']
        corruptness = ['correct', 'corrupted']
        seen = set()
        plt.figure()
        for j in range(len(x)):
            corrupt_label = corrupt_labels[j].item()
            if corrupt_label not in seen:
                seen.add(corrupt_label)
                plt.plot(x[j], y[j], c=colors[corrupt_label], label=corruptness[corrupt_label])
            else:
                plt.plot(x[j], y[j], c=colors[corrupt_label])
        plt.title('Dist btw. prediction and glimpse semantics')
        plt.xlabel('Glimpse iteration')
        plt.ylabel('L2 distance')
        plt.legend()
        plt.savefig('/home/nano01/a/tao88/3.21_fivepatch/curves_all.png')

        # 3.9.2023 - tao88: plot an average distance curve for all correct images, and another for all corrupted
        dist_correct_imgs, dist_corrupted_imgs = [], []
        for j in range(translated_images.shape[0]):
            corrupt_label = corrupt_labels[j].item()
            if corrupt_label == 0:
                dist_correct_imgs.append(dist_batch_imgs[j])
            else:
                dist_corrupted_imgs.append(dist_batch_imgs[j])
        dist_correct_imgs = np.average(np.array(dist_correct_imgs), axis=0) # compute average curve for all correct images
        dist_corrupted_imgs = np.average(np.array(dist_corrupted_imgs), axis=0) # compute average curve for all corrupted images
        plt.figure()
        plt.plot(np.arange(1, 5), dist_correct_imgs, c='b', label=corruptness[0], linewidth=4)
        plt.plot(np.arange(1, 5), dist_corrupted_imgs, c='r', label=corruptness[1], linewidth=4)
        plt.title('Averaged dist btw. prediction and glimpse semantics')
        plt.xlabel('Glimpse iteration')
        plt.ylabel('L2 distance')
        plt.legend()
        plt.savefig('/home/nano01/a/tao88/3.21_fivepatch/curves_together_avg.png')

        pdb.set_trace()

        if (i+1) % 100 == 0:
            print("Current process: batch: {}/{}".format(i+1, len(valid_loader)))
            # print("Loss: {}".format(train_loss_glmipse_mask))

pdb.set_trace()