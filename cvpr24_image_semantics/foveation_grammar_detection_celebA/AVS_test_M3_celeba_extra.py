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


#%% Instantiate parameters, dataloaders, and model
# Parameters
config_1 = AVS_config_for_M1
config_2 = AVS_config_for_M2
config_3 = AVS_config_for_M3
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
ckpt_3 = torch.load(config_3.ckpt_dir)
model_3.load_state_dict(ckpt_3['model'])
for p in model_3.parameters():
    p.requires_grad_(False)
model_3.eval()
print("Loaded from: {}\n".format(config_3.ckpt_dir))
print(model_3)


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
glimpses_locs_dims_array    = []
similarity_array            = []
rewards_array               = []
iou_array                   = []
eval_loader                 = test_loader
with torch.no_grad():
    for i, (images, targets) in enumerate(eval_loader):
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
        if i == len(eval_loader) - 1:
            glimpses_locs_dims_array.append(glimpses_locs_dims.clone().detach())
# =============================================================================
#       M3 Stage
# =============================================================================
        model_3.clear_memory()
        loss_classification     = None
        loss_glimpse_change     = None
    
        for g in range(config_3.num_glimpses):
            # Extract and resize the batch of glimpses based on their current locations and dimensions.
            glimpses_extracted_resized = extract_and_resize_glimpses_for_batch(translated_images, glimpses_locs_dims,
                                                                               config_3.glimpse_size_fixed[1], config_3.glimpse_size_fixed[0]) # glimpse_size_fixed[width, height]

            # Process the batch of extracted and resized glimpses through the network.
            # glimpses_change_actions[batch_size, 4] - where second dimension is 4-sized tuple, 
            # representing (dx-, dx+, dy-, dy+) changes of every glimpse in x and y directions
            output_predictions, glimpses_change_actions = model_3(glimpses_extracted_resized)
        
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
            if i == len(eval_loader) - 1:
                glimpses_locs_dims_array.append(glimpses_locs_dims.clone().detach())
            iou = region_iou(glimpses_locs_dims.clone().detach(), bbox_targets).diag()
            if i == 0:
                iou_array.append(iou)
            else:
                iou_array[g] = torch.cat([iou_array[g], iou])
    
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

#%% Estimating average iterative statistics
plt.rcParams.update({'font.size': 35})
plt.rcParams.update({"font.family":"Times New Roman"})

similarity_array_mean = []
similarity_array_std = []
for j in range(len(similarity_array)):
    similarity_array_mean.append(torch.mean(similarity_array[j]))
    similarity_array_std.append(torch.std(similarity_array[j]))
    
similarity_array_mean = torch.stack(similarity_array_mean).cpu()
similarity_array_std = torch.stack(similarity_array_std).cpu()



iou_array_mean = []
iou_array_std = []
for j in range(len(iou_array)):
    iou_array_mean.append(torch.mean(iou_array[j]))
    iou_array_std.append(torch.std(iou_array[j]))
    
iou_array_mean = torch.stack(iou_array_mean).cpu()
iou_array_std = torch.stack(iou_array_std).cpu()



rewards_array_tensor = torch.stack(rewards_array)
rewards_array_cumulative = torch.cumsum(rewards_array_tensor, dim=0)

rewards_array_mean = []
rewards_array_std = []
for j in range(len(rewards_array_cumulative)):
    rewards_array_mean.append(torch.mean(rewards_array_cumulative[j]))
    rewards_array_std.append(torch.std(rewards_array_cumulative[j]))
    
rewards_array_mean = torch.stack(rewards_array_mean).cpu()
rewards_array_std = torch.stack(rewards_array_std).cpu()



plt.figure()
plt.xlabel('Foveated glimpse iteration')
plt.ylabel('Average cosine similarity')
plt.xlim([1, 16])
plt.plot(range(1, len(similarity_array_mean)+1), similarity_array_mean.numpy(), linewidth=5, color='dodgerblue')
plt.fill_between(range(1, len(similarity_array_mean)+1), ((similarity_array_mean-similarity_array_std).numpy()), ((similarity_array_mean+similarity_array_std).numpy()), alpha=.1, color='dodgerblue')


plt.figure()
plt.xlabel('Foveated glimpse iteration')
plt.ylabel('Average intersection over union (IoU)')
plt.xlim([1, 16])
plt.plot(range(1, len(iou_array_mean)+1), iou_array_mean.numpy(), linewidth=5, color='darkviolet')
plt.fill_between(range(1, len(iou_array_mean)+1), ((iou_array_mean-iou_array_std).numpy()), ((iou_array_mean+iou_array_std).numpy()), alpha=.1, color='darkviolet')



plt.figure()
plt.xlabel('Foveated glimpse iteration')
plt.ylabel('Average cumulative RL rewards')
plt.xlim([1, 16])
plt.plot(range(1, len(rewards_array_mean)+1), rewards_array_mean.numpy(), linewidth=5, color='g')
plt.fill_between(range(1, len(rewards_array_mean)+1), ((rewards_array_mean-rewards_array_std).numpy()), ((rewards_array_mean+rewards_array_std).numpy()), alpha=.1, color='g')



