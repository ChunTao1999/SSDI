#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 22:38:55 2021
Verified on Wed May 25 2022

@author: tibrayev
"""

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
train_loader, loss_weights    = get_dataloaders(config_3, loader_type='train')
valid_loader                  = get_dataloaders(config_3, loader_type='valid')
test_loader                   = get_dataloaders(config_3, loader_type='test')

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

# Set logger
if not os.path.exists(args.save_eval_path):
    os.makedirs(args.save_eval_path)
logger = set_logger(os.path.join(args.save_eval_path, 'train.log'))

# Get PiCIE model and optimizer(centroids)
model_FPN, optimizer, classifier1 = get_model_and_optimizer(args, logger) # note dataparallel objects
for p in model_FPN.parameters():
    p.requires_grad_(False)
model_FPN.eval()

# 2.2 - tao88: tensorboard
# writer = SummaryWriter(comment="visualize_image_0_glimpse_labels")

#%% AVS-specific functions, methods, and parameters
from AVS_functions import location_bounds, extract_and_resize_glimpses_for_batch

lower, upper = location_bounds(glimpse_w=config_3.glimpse_size_init[0], input_w=config_3.full_res_img_size[0]) #TODO: make sure this works on both x and y coordinates


#%% Evaluate the model
model_3.clear_memory()
test_loss                   = 0.0
test_loss_classification    = 0.0
test_loss_glimpse_change    = 0.0
acc_correct_attr            = 0
acc_localization            = 0
acc_attr_localized          = 0
test_ave_iou                = 0.0
total_attr                  = 0
total_samples               = 0
eval_loader                 = test_loader
batch_to_capture            = len(eval_loader) - 1
captured_batch_images       = None
captured_batch_gt_bboxes    = None
glimpses_locs_dims_array    = []
glimpses_iou_array          = []
glimpses_array              = []
similarity_array            = []
rewards_array               = []
iou_array                   = []
with torch.no_grad():
    for i, (index, images, targets) in enumerate(eval_loader):
        translated_images, targets, bbox_targets                    = images.to(device), targets[0].float().to(device), targets[1].to(device)        
# =============================================================================
#       DATA STRUCTURES to keep track of glimpses
# =============================================================================
        glimpses_locs_dims      = torch.zeros((targets.shape[0], 4), dtype=torch.int).to(device)

        glimpses_extracted_resized = torch.zeros((translated_images.shape[0],   translated_images.shape[1], 
                                                  config_3.glimpse_size_fixed[1], config_3.glimpse_size_fixed[0])).to(device) # glimpse_size_fixed[width, height]

        rewards = []
        actions = []
        simscores = []
# =============================================================================
#       M2 Stage: getting initial predicted locations        
# =============================================================================
        # M2 predicts the center of the initial point-of-interest
        pred_locations          = model_2(translated_images) # predictions are in range of (-1, 1)
        initial_location        = torch.clamp(pred_locations, min=lower, max=upper)
        initial_location[:, 0]  = ((initial_location[:, 0] + 1.0)/2.0)*config_3.full_res_img_size[0] # convert from (-1, 1) range to (0, img_width) range
        initial_location[:, 1]  = ((initial_location[:, 1] + 1.0)/2.0)*config_3.full_res_img_size[1] # convert from (-1, 1) range to (0, img_height) range
    
        # then, we translate the center of the initial point-of-interest into (x_TopLeftCorner, y_TopLeftCorner, width, height)       
        glimpses_locs_dims[:, 0]    = initial_location[:, 0] + 0.5 - (config_3.glimpse_size_init[0]/2.0) #!!!: we add 0.5 for printing purposes only. See base_version/AVS_train_a2.py for explanation
        glimpses_locs_dims[:, 1]    = initial_location[:, 1] + 0.5 - (config_3.glimpse_size_init[1]/2.0)
        glimpses_locs_dims[:, 2]    = config_3.glimpse_size_init[0]
        glimpses_locs_dims[:, 3]    = config_3.glimpse_size_init[1] 

        # 2.7.2023 - tao88: change initialization point, for image 0, glimpses_loc_dims=[122, 136, 20, 20]
        # glimpses_locs_dims[0] = torch.tensor([153, 164, 20, 20])
        # pdb.set_trace()

        if i == batch_to_capture: # last batch
            captured_batch_images = translated_images.clone().detach()
            captured_batch_gt_bboxes = bbox_targets.clone().detach()
            glimpses_locs_dims_array.append(glimpses_locs_dims.clone().detach())
            iou = region_iou(glimpses_locs_dims.clone().detach(), bbox_targets).diag()
            glimpses_iou_array.append(iou.clone().detach())
# =============================================================================
#       M3 Stage
# =============================================================================
        model_3.clear_memory()
        loss_classification     = None
        loss_glimpse_change     = None
    
        glimpse_wise_face_part_prob = []
        for g in range(config_3.num_glimpses):
            # Extract and resize the batch of glimpses based on their current locations and dimensions.
            glimpses_extracted_resized, glimpses_extracted_resized_FPN = extract_and_resize_glimpses_for_batch(translated_images, glimpses_locs_dims, config_3.glimpse_size_fixed[1], config_3.glimpse_size_fixed[0], args.res, args.res) # glimpse_size_fixed[width, height]

            # 2.7.2023 - tao88: glimpses_extracted_resized_FPN has shape (128, 3, 256, 256)

            if i == batch_to_capture:
                glimpses_array.append(glimpses_extracted_resized.clone().detach())

            # 2.2 - tao88: use with_latent in model_3 to get the features of the glimpses
            output_predictions, glimpses_change_actions = model_3(glimpses_extracted_resized)
            glimpse_spatial_map = model_FPN(glimpses_extracted_resized_FPN) # (128, 128, 64, 64)
            glimpse_spatial_map = nnF.normalize(glimpse_spatial_map, dim=1, p=2)
            prb = compute_dist(glimpse_spatial_map, classifier1) # (128, 20, 64, 64)
            lbl = prb.topk(1, dim=1)[1] # (128, 1, 64, 64)
            # lbl = lbl.squeeze(1) # (128, 64, 64)

            # 2.7.2023 - tao88: visualize the glimpses of image 0, as well as the lbls these glimpses. check whether it's consistent over foveation glimpses
            save_image(glimpses_extracted_resized_FPN[0], "/home/nano01/a/tao88/2.10/glimpses/img_0_glimpse_{}.png".format(g))
            save_image(lbl[0] / 20, "/home/nano01/a/tao88/2.10/img_0_glimpse_{}_label.png".format(g))
            # pdb.set_trace()


            # # PCA reduction from 1024-d to 128-d, batch size needs to be configured greater than 128
            # mypca = PCA # try adding FC(1024, 128) and finetune on attributes instead
            # glimpse_features = mypca.Decomposition(glimpse_features, 128) # note the two dimensions are 1024 and 128
            # # mypca.explained_variance()
            # glimpse_features = F.normalize(glimpse_features, dim=1, p=2) # some problem with F.normalize
            # glimpse_features = glimpse_features / (glimpse_features.norm(p=2, dim=1, keepdim=True)) 
            # glimpse_features = glimpse_features.unsqueeze(2).unsqueeze(3)
            # prb = compute_dist(glimpse_features, classifier1) 
            # prb = prb.squeeze()
            # glimpse_wise_face_part_prob.append(prb[0].detach().cpu().numpy())
            # # print('glimpse_{}'.format(g))
            # # print(prb[0]) # 20-d

        
            # Estimate classification loss.
            if g == (config_3.num_glimpses - 1):
                loss_classification  = bce_loss(output_predictions, targets)

            # Estimate rewards for RL agent.
            #@Version 6 extension E:
            similarity      = torch.relu(torch.cosine_similarity(output_predictions >= config_3.attr_detection_th, targets)).clone().detach()
            action_cost     = glimpses_change_actions.mean(dim=-1).clone().detach()
            if g == 0:
                max_sim_val     = similarity.clone().detach()   # maximum value of similarity detected over all glimpses
                max_sim_gid     = torch.zeros_like(similarity)  # glimpse at which maximum value of similarity is detected
            else:
                new_max_sim     = similarity >= max_sim_val
                max_sim_val[new_max_sim] = similarity[new_max_sim]
                max_sim_gid[new_max_sim] = g
            actions.append(action_cost.clone().detach())
            simscores.append(similarity.clone().detach())
            if i == 0:
                similarity_array.append(similarity)
            else:
                similarity_array[g] = torch.cat([similarity_array[g], similarity])
            
            if g == (config_3.num_glimpses - 1):
                for g_replay in range(config_3.num_glimpses):
                    g_reward                = torch.zeros_like(similarity)
                    if g_replay > 0:
                        # mask_a -> keeps track of whether the maximum similarity was not achieved yet
                        mask_a              = g_replay <= max_sim_gid
                        mask_a_not          = torch.logical_not(mask_a)
                        # mask_b -> keeps track of whether the action improved similarity
                        mask_b_pos          = simscores[g_replay - 1] < simscores[g_replay]
                        mask_b_neg          = simscores[g_replay - 1] > simscores[g_replay] # !!!: Note that these two masks exclude the case of equal similarity scores
                        mask_b_nochange     = simscores[g_replay - 1] == simscores[g_replay]
                        # mask_c -> keeps track of whether the action was to STOP
                        mask_c_stop         = actions[g_replay - 1] == 0 # !!!: STOP action is the only action with mean 0
                        mask_c_move         = torch.logical_not(mask_c_stop)
                        
                        # mask_1 -> rule_1 rewards
                        mask_1              = torch.logical_and(mask_a, mask_b_pos)
                        # mask_2 -> rule_2 punishments
                        mask_2              = torch.logical_and(mask_a, mask_b_neg)
                        # mask_3 -> rule_3 punishments
                        mask_3              = torch.logical_and(mask_a, mask_c_stop)
                        # mask_4 -> rule_4 rewards
                        mask_4              = torch.logical_and(mask_a_not, mask_c_stop)
                        # mask_5 -> rule_5 punishments
                        mask_5              = torch.logical_and(mask_a_not, mask_c_move)
                        

                        # Rule 1: before max similarity is achieved, improvement in similarity is rewarded!
                        # Rule 2: before max similarity is achieved, degradation in similarity is punished!
                        # Rule 3: before max similarity is achieved, any STOP action is punished!
                        # Rule 4: after max similarity is achieved, reward ONLY STOP action (i.e. "0,0,0,0" action)
                        # Rule 5: after max similarity is achieved, punish any other action
                        
                        g_reward[mask_1]    = 1.0
                        g_reward[mask_2]    = -actions[g_replay - 1][mask_2]
                        g_reward[mask_3]    = -1.0
                        g_reward[mask_4]    = 1.0
                        g_reward[mask_5]    = -actions[g_replay - 1][mask_5]
                        
                    rewards.append(g_reward.clone().detach())
                    if i == 0:
                        rewards_array.append(g_reward)
                    else:
                        rewards_array[g_replay] = torch.cat([rewards_array[g_replay], g_reward])
            model_3.store_all_rewards(rewards)
        
            # Change the glimpse locations and dimensions.
            x_min_current   = (glimpses_locs_dims[:, 0]).clone().detach()
            x_max_current   = (glimpses_locs_dims[:, 0]+glimpses_locs_dims[:, 2]).clone().detach()
            y_min_current   = (glimpses_locs_dims[:, 1]).clone().detach()
            y_max_current   = (glimpses_locs_dims[:, 1]+glimpses_locs_dims[:, 3]).clone().detach()
        
            # Check so that glimpses do not go out of the image boundaries.
            x_min_new       = torch.clamp(x_min_current - glimpses_change_actions[:, 0]*config_3.glimpse_size_step[0], min=0)
            x_max_new       = torch.clamp(x_max_current + glimpses_change_actions[:, 1]*config_3.glimpse_size_step[0], max=config_3.full_res_img_size[1]) #(height, width) as used in transforms.Resize
            y_min_new       = torch.clamp(y_min_current - glimpses_change_actions[:, 2]*config_3.glimpse_size_step[1], min=0)
            y_max_new       = torch.clamp(y_max_current + glimpses_change_actions[:, 3]*config_3.glimpse_size_step[1], max=config_3.full_res_img_size[0]) #(height, width) as used in transforms.Resize
        
            # Store the new glimpse locations and dimensions.
            glimpses_locs_dims[:, 0] = x_min_new.clone().detach()
            glimpses_locs_dims[:, 1] = y_min_new.clone().detach()
            glimpses_locs_dims[:, 2] = x_max_new.clone().detach() - glimpses_locs_dims[:, 0]
            glimpses_locs_dims[:, 3] = y_max_new.clone().detach() - glimpses_locs_dims[:, 1]
            if i == batch_to_capture:
                glimpses_locs_dims_array.append(glimpses_locs_dims.clone().detach())
                iou = region_iou(glimpses_locs_dims.clone().detach(), bbox_targets).diag()
                glimpses_iou_array.append(iou.clone().detach())
            
            # 1.26 - tao88, iou_array not useful
            # if i == 0:
            #     iou_array.append(iou)
            # else:
            #     iou_array[g] = torch.cat([iou_array[g], iou])

        # 2.2 - tao88
        # glimpse_wise_face_part_prob = np.asarray(glimpse_wise_face_part_prob) # (16, 20)
        # for k in range(20):
        #     plt.figure()
        #     plt.plot(range(16), glimpse_wise_face_part_prob[:,k])
        #     plt.savefig('/home/nano01/a/tao88/PiCIE_results/face_part_index_{}.png'.format(k))
        pdb.set_trace()

        # Estimate the RL agent loss.
        loss_glimpse_change = model_3.compute_rl_loss()
        loss                = loss_glimpse_change + loss_classification

        pred_labels         = (output_predictions >= config_3.attr_detection_th).float()
        total_attr         += targets.numel()
        total_samples      += targets.size(0)
        test_loss          += loss.item()
        test_loss_classification   += loss_classification.item()
        test_loss_glimpse_change   += loss_glimpse_change.item()
        
        iou = region_iou(glimpses_locs_dims.clone().detach(), bbox_targets).diag()
        test_ave_iou        += iou.sum().item()

        correct_attr         = pred_labels == targets
        correct_tp_loc       = iou >= config_3.iou_th
        acc_correct_attr    += correct_attr.sum().item()
        acc_localization    += correct_tp_loc.sum().item()
        acc_attr_localized  += correct_attr[correct_tp_loc, :].sum().item()

print("Validation Loss: {:.3f} | {:.3f} | {:.3f}\n".format(
    (test_loss/(i+1)), (test_loss_classification/(i+1)), (test_loss_glimpse_change/(i+1)))) 
print("Attribute Acc: {:.4f} [{}/{}] | GT Loc: {:.4f} [{}/{}] | Attr Loc: {:.4f} [{}/{}]\n.".format(
    (100.*acc_correct_attr/total_attr), acc_correct_attr, total_attr,
    (100.*acc_localization/total_samples), acc_localization, total_samples,
    (100.*acc_attr_localized/total_attr), acc_attr_localized, total_attr))
print("IoU@{}: Average: {:.3f} | TPR: {:.4f} [{}/{}]\n".format(
    config_3.iou_th, (test_ave_iou/total_samples), 
    (1.*acc_localization/total_samples), acc_localization, total_samples))

#%% Qualitative inspection.
plt.close('all')
sample_id = 16
imshow(translated_images[sample_id])
plotregions(bbox_targets[sample_id].unsqueeze(0), color='r')
plotregions(glimpses_locs_dims_array[0][sample_id].unsqueeze(0))
plotregions(glimpses_locs_dims_array[1][sample_id].unsqueeze(0), color='darkorange')
plotregions(glimpses_locs_dims_array[2][sample_id].unsqueeze(0), color='k')
plotregions(glimpses_locs_dims_array[3][sample_id].unsqueeze(0), color='y')
plotregions(glimpses_locs_dims_array[4][sample_id].unsqueeze(0), color='m')
plotregions(glimpses_locs_dims_array[5][sample_id].unsqueeze(0), color='b')
plotregions(glimpses_locs_dims_array[6][sample_id].unsqueeze(0), color='w')
plotregions(glimpses_locs_dims_array[7][sample_id].unsqueeze(0), color='c')
plotregions(glimpses_locs_dims_array[8][sample_id].unsqueeze(0))
plotregions(glimpses_locs_dims_array[9][sample_id].unsqueeze(0), color='darkorange')
plotregions(glimpses_locs_dims_array[10][sample_id].unsqueeze(0), color='k')
plotregions(glimpses_locs_dims_array[11][sample_id].unsqueeze(0), color='y')
plotregions(glimpses_locs_dims_array[12][sample_id].unsqueeze(0), color='m')
plotregions(glimpses_locs_dims_array[13][sample_id].unsqueeze(0), color='b')
plotregions(glimpses_locs_dims_array[14][sample_id].unsqueeze(0), color='w')
plotregions(glimpses_locs_dims_array[15][sample_id].unsqueeze(0), color='c')
plotregions(glimpses_locs_dims_array[16][sample_id].unsqueeze(0))

#%% Qualitative inspection EXTRA

def imshow(ax, input, normalize=True):
    input_to_show = input.cpu().clone().detach()
    if normalize:
        input_to_show = (input_to_show - input_to_show.min())/(input_to_show.max() - input_to_show.min())
    if input_to_show.ndim == 4 and input_to_show.size(1) == 3:
        ax.imshow(input_to_show[0].permute(1,2,0))
    elif input_to_show.ndim == 4 and input_to_show.size(1) == 1:
        ax.imshow(input_to_show[0,0])
    elif input_to_show.ndim == 3 and input_to_show.size(0) == 3:
        ax.imshow(input_to_show.permute(1,2,0))
    elif input_to_show.ndim == 3 and input_to_show.size(0) == 1:
        ax.imshow(input_to_show[0])
    elif input_to_show.ndim == 2:
        ax.imshow(input_to_show)
    else:
        raise ValueError("Input with {} dimensions is not supported by this function!".format(input_to_show.ndim))

def plotregions(ax, list_of_regions, glimpse_size = None, color='g', **kwargs):
    if glimpse_size is None:
        for region in list_of_regions:
            xmin = region[0].item()
            ymin = region[1].item()
            width = region[2].item()
            height = region[3].item()
            # Add the patch to the Axes
            # FYI: Rectangle doc says the first argument defines bottom left corner. However, in reality it changes based on plt axis. 
            # So, if the origin of plt (0,0) is at top left, then (x,y) specify top left corner. 
            # Essentially, (x,y) needs to point to x min and y min of bbox.
            ax.add_patch(Rectangle((xmin,ymin), width, height, edgecolor=color, facecolor='none', **kwargs))
    elif glimpse_size is not None:
        if isinstance(glimpse_size, tuple):
            width, height = glimpse_size
        else:
            width = height = glimpse_size
        for region in list_of_regions:
            xmin = region[0].item()
            ymin = region[1].item()
            ax.add_patch(Rectangle((xmin,ymin), width, height, edgecolor=color, facecolor='none', **kwargs))

plt.rcParams.update({'xtick.labelsize': 16})
plt.rcParams.update({'ytick.labelsize': 16})


for sample_id in range(0, translated_images.size(0)):
    save_dir = './results/celeba/for_paper/test_samples/target_batch_{}_test_sample_{}/'.format(batch_to_capture, sample_id)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    full_img = captured_batch_images[sample_id]
    gt_bbox  = captured_batch_gt_bboxes[sample_id].unsqueeze(0)
    
    for t in range(0, len(glimpses_array)):
        # Figure
        fig = plt.figure(constrained_layout=True, figsize=(7, 4))
        subfigs = fig.subfigures(1, 2, wspace=0.0)
        fig.suptitle('Foveated iteration at {}'.format('$t={}$'.format(t+1)), fontsize='xx-large')
        
        # Left subfigure
        foveated_region = glimpses_locs_dims_array[t][sample_id].unsqueeze(0)
        IoU = glimpses_iou_array[t][sample_id].item()
        axsLeft = subfigs[0].subplots()
        axsLeft.set_xticks([0, 255])
        axsLeft.xaxis.set_ticks_position('none') 
        axsLeft.set_yticks([0, 255])
        axsLeft.yaxis.set_ticks_position('none') 
        imshow(axsLeft, full_img)
        plotregions(axsLeft, foveated_region, color='b', linewidth=4)
        if t == len(glimpses_array)-1:
            plotregions(axsLeft, gt_bbox, color='r', linewidth=4, linestyle='--')
            subfigs[0].suptitle('Foveated region with GT bbox\non original image(IoU={:.2f})'.format(IoU), fontsize='x-large')
        else:
            subfigs[0].suptitle('Foveated region on\noriginal image (IoU={:.2f})'.format(IoU), fontsize='x-large')
        
        # Right subfigure
        foveated_glimpse = glimpses_array[t][sample_id].unsqueeze(0)
        axsRight = subfigs[1].subplots()
        axsRight.set_xticks([0, 95])
        axsRight.xaxis.set_ticks_position('none') 
        axsRight.set_yticks([0, 95])
        axsRight.yaxis.set_ticks_position('none') 
        imshow(axsRight, foveated_glimpse)
        subfigs[1].suptitle('Resized foveated glimpse\nas perceived by M2 models', fontsize='x-large')
        
        #plt.show()
        plt.savefig(save_dir + 'foveated_iteration_t_{}.png'.format(t+1))
        plt.close('all')




