'''
Created on 2/6/2023
@tao88
Use modified Resnet+FPN so that it outputs pixel-wise features, finetune using the segmentation masks as supervision
'''

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

from utils_custom_tvision_functions import get_dataloaders, plot_curve
from AVS_config_M1_celeba import AVS_config as AVS_config_for_M1
from AVS_config_M2 import AVS_config as AVS_config_for_M2
from AVS_config_M3_celeba import AVS_config as AVS_config_for_M3
from AVS_model_M1_vgg import customizable_VGG as VGG_for_M1
from AVS_model_M2 import context_network
from AVS_model_M3_vgg import customizable_VGG as VGG_for_M3
from utils_custom_tvision_functions import imshow, plotregions, plotspots, plotspots_at_regioncenters, region_iou, region_area
# debug
from torchvision.utils import save_image
import pdb
# PiCIE-related
from commons import *
import argparse
import matplotlib.pyplot as plt

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
train_loader, loss_weights    = get_dataloaders(config_3, loader_type=config_3.train_loader_type) # trainval split
valid_loader                  = get_dataloaders(config_3, loader_type=config_3.valid_loader_type) # test split

# Loss(es)
# bce_loss                      = nn.BCEWithLogitsLoss(pos_weight=loss_weights.to(device))  
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
args.save_model_path = args.eval_path

# Set logger
logger = set_logger('./train_Resnet.log')

# Initialize model (import the finetuned model and load its trained weights)
model_FPN, optimizer = get_model_and_optimizer_loadpart(args, logger) # load FPN params, initialize classifier params
logger.info('Adam optimizer is used.')
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma = 0.1, milestones=[40, 80])

# If initialized from middle of training
# model_FPN = fpn.PanopticFPN(args)
# model_FPN = nn.DataParallel(model_FPN) # comment for same device
# model_FPN = model_FPN.cuda()
# checkpoint  = torch.load(args.eval_path)
# args.start_epoch = checkpoint['epoch']
# model_FPN.load_state_dict(checkpoint['state_dict'])
# logger.info('Loaded checkpoint. [epoch {}]'.format(args.start_epoch))
# logger.info('Adam optimizer is used.')
# optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model_FPN.parameters()), lr=args.lr)
# optimizer.load_state_dict(checkpoint['optimizer'])
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma = 0.1, milestones=[40, 80])

# First try training on glimpses without freezing any layers
# check number of parameters and number of parameters that require gradient
print("Number of params: {}".format(sum(p.numel() for p in model_FPN.parameters())))
print("Number of trainable params: {}".format(sum(p.numel() for p in model_FPN.parameters() if p.requires_grad)))

model_FPN.train()
# pdb.set_trace() # done


#%% AVS-specific functions, methods, and parameters
from AVS_functions import location_bounds, extract_and_resize_glimpses_for_batch, crop_five, extract_and_resize_masks

lower, upper = location_bounds(glimpse_w=config_3.glimpse_size_init[0], input_w=config_3.full_res_img_size[0]) #TODO: make sure this works on both x and y coordinates

#%% Train the model
log_dict = {"train_loss_crop_mask": []} # 3.14.2023 - tao88
for epoch in range(0, args.num_epoch):
    # if epoch > 0: break # for debug
    print("\nEpoch: {}/{}".format(epoch+1, args.num_epoch))
    # initiate epoch losses
    train_loss_crop_mask = 0.0
# =============================================================================
#   TRAINING
# =============================================================================    
    for i, (index, images, targets) in enumerate(train_loader): # target_type=['attr', 'bbox', 'pred_init_glimpse_locs', 'landmarks', 'masks']
        if i == len(train_loader) - 1:
            continue # 2.15.2023 - tao88, in order to account for error in compute_rl_loss
        translated_images, targets, bbox_targets, masks = images.to(device), targets[0].float().to(device), targets[1].to(device), targets[4].to(device) # translated images (128, 3, 256, 256); targets (128, 40); bbox_targets (128, 4); masks (128, 1, 256, 256)
        masks = torch.round(masks * 7).long() # round the aligned masks
        loss_crop_mask = torch.tensor([0.0]).cuda()

        # 2.13.2023 - visualize some aligned images and their masks to make sure of correspondance
        # if i == 0:
        #     for img_id in range(16):
        #         save_path = '/home/nano01/a/tao88/3.14_traincrop/img_{}'.format(img_id)
        #         if not os.path.exists(save_path): os.makedirs(save_path)
        #         # images
        #         save_image(translated_images[img_id], os.path.join(save_path, 'aligned_img_{}.png'.format(img_id)))
        #         # aligned masks
        #         fig = plt.figure(figsize=(1,1))
        #         ax = plt.Axes(fig, [0., 0., 1., 1.])
        #         ax.set_axis_off()
        #         fig.add_axes(ax)
        #         ax.imshow(masks[img_id].squeeze(0).cpu().numpy(),
        #                   interpolation='nearest',
        #                   cmap='Paired',
        #                   vmin=0,
        #                   vmax=6)
        #         plt.savefig(os.path.join(save_path, 'aligned_mask_{}.png'.format(img_id)))
        #         plt.close(fig)

        # 3.14.2023 - tao88: crop five patches around the center of aligned images, default size is 64x64, reshaped to 256x256
        batch_patches = crop_five(translated_images, resized_height=256, resized_width=256) # (128, 5, 3, 256, 256)
        masks_extracted_resized = extract_and_resize_masks(masks, resized_height=64, resized_width=64) # (128, 5, 1, 64, 64)

        # pass each of the patches through the FPN network to predict masks
        for patch_id in range(5):
            patches = batch_patches[:, patch_id, :, :, :] # (128, 3, 256, 256)
            masks_predicted = model_FPN(patches) # (128, 7, 64, 64)
            # lbl_predicted = masks_predicted.clone().detach().topk(1, dim=1)[1] # (128, 1, 64, 64)
            mask_patches = masks_extracted_resized[:, patch_id, :, :, :] # (128, 1, 64, 64)
            loss_crop_mask += ce_loss(masks_predicted, mask_patches.squeeze(1)) #(128, 7, 64, 64), (128, 64, 64)

            # 3.14.2023 - visualize the patches and mask patches
            # if i == 0:
            #     for img_id in range(16):
            #         save_path = '/home/nano01/a/tao88/3.14_traincrop/img_{}'.format(img_id)
            #         if not os.path.exists(save_path): os.makedirs(save_path)
            #         # patches
            #         save_image(patches[img_id], os.path.join(save_path, 'img_{}_patch_{}.png'.format(img_id, patch_id)))
            #         # patch masks
            #         fig = plt.figure(figsize=(1,1))
            #         ax = plt.Axes(fig, [0., 0., 1., 1.])
            #         ax.set_axis_off()
            #         fig.add_axes(ax)
            #         ax.imshow(mask_patches[img_id].squeeze(0).cpu().numpy(),
            #                   interpolation='nearest',
            #                   cmap='Paired',
            #                   vmin=0,
            #                   vmax=6)
            #         plt.savefig(os.path.join(save_path, 'img_{}_patch_{}_mask.png'.format(img_id, patch_id)))
            #         plt.close(fig)
        # pdb.set_trace()

        # backward the CE loss between predicted masks and extracted and resized masks
        optimizer.zero_grad()
        loss_crop_mask.backward()
        optimizer.step()
        train_loss_crop_mask += loss_crop_mask.item()

        # pdb.set_trace()
        if (i+1) % 100 == 0:
            print("Current process: batch: {}/{}   Current batch loss: {}".format(i+1, len(train_loader), loss_crop_mask.item()))
    
    # End of epoch, lr_scheduler update
    # if optimizer.param_groups[0]['lr'] > config_3.lr_min:
    lr_scheduler.step()
    
    # Print losses
    print("Epoch Train Loss (crop_mask_model): {:.3f}\n".format(train_loss_crop_mask / (i+1)))
    log_dict['train_loss_crop_mask'].append(train_loss_crop_mask/(i+1))
    
    # Storing results
    # 2.17.2023 - tao88
    print("Saving model: epoch: {}/{}\n".format(epoch+1, args.num_epoch))
    if (epoch+1) % 10 == 0: 
        torch.save({'epoch': epoch+1, 
                    'args' : args,
                    'log_dict': log_dict,
                    'state_dict': model_FPN.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    },
                    os.path.join('./crop_mask_models', 'checkpoint_{}.pth.tar'.format(epoch+1)))
            
    torch.save({'epoch': epoch+1, 
                'args' : args,
                'log_dict': log_dict,
                'state_dict': model_FPN.state_dict(),
                'optimizer' : optimizer.state_dict(),
                },
                os.path.join('./crop_mask_models', 'checkpoint_latest_{}.pth.tar'.format(epoch+1)))
pdb.set_trace()