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
from torchvision.utils import save_image
import pdb

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
train_loader, loss_weights    = get_dataloaders(config_3, loader_type=config_3.train_loader_type)
valid_loader                  = get_dataloaders(config_3, loader_type=config_3.valid_loader_type)
# pdb.set_trace()
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
if config_3.initialize_M3 == 'from_M1': # This is in the current config
    model_1_sd = model_1.state_dict()
    model_3_sd = model_3.state_dict()
    for (k_3, v_3), (k_1, v_1) in zip(model_3_sd.items(), model_1_sd.items()):
        if 'features' in k_3 or 'classifier' in k_3:
            model_3_sd[k_3] = model_1_sd[k_1].clone().detach()
            print("Initialized M3 layer ({}) from M1 layer ({})".format(k_3, k_1))
    model_3.load_state_dict(model_3_sd)
elif config_3.initialize_M3 == 'from_checkpoint':
    ckpt_3 = torch.load(config_3.ckpt_dir_model_M3)
    model_3.load_state_dict(ckpt_3['model'])
    print("Loaded from: {}\n".format(config_3.ckpt_dir_model_M3))
elif config_3.initialize_M3 == 'from_scratch':
    print("Training M3 from scratch!\n")
else:
    raise ValueError("Unknown value for argument config.initialize_M3: ({}). ".format(config_3.initialize_M3) +
                     "Acceptable options are: ('from_M1', 'from_checkpoint', 'from_scratch').")
for n, p in model_3.named_parameters():
    if 'features' in n or 'classifier' in n:
        p.requires_grad_(False)
print(model_3)
# pdb.set_trace()


# Optimizer
optimizer = torch.optim.Adam(model_3.parameters(), lr=config_3.lr_start, weight_decay=config_3.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma = 0.1, milestones=config_3.milestones)

# Logger dictionary
log_dict     = {'train_loss':[], 
                'train_loss_classification': [], 
                'train_loss_glimpse_change': [], 
                'train_acc_correct_attr':[],
                'train_acc_localization':[],
                'train_acc_attr_localized':[], 
                'test_loss':[],
                'test_loss_classification': [],
                'test_loss_glimpse_change': [],
                'test_acc_correct_attr':[],
                'test_acc_localization':[],
                'test_acc_attr_localized':[]}
if not os.path.exists(config_3.save_dir): os.makedirs(config_3.save_dir)

#%% AVS-specific functions, methods, and parameters
from AVS_functions import location_bounds, extract_and_resize_glimpses_for_batch

lower, upper = location_bounds(glimpse_w=config_3.glimpse_size_init[0], input_w=config_3.full_res_img_size[0]) #TODO: make sure this works on both x and y coordinates

#%% Face semantic part masks (by Chun)
#eyes_mask = [1, 3, 12, 15, 23]
#nose_mask = [7, 27]
#cheek_mask = [19, 29, 30]
#mouth_mask = [0, 6, 14, 16, 21, 22, 24, 36]
#hair_mask = [4, 5, 8, 9, 11, 17, 28, 32, 33, 35]
#general_mask = [2, 10, 13, 18, 20, 25, 26, 31, 34, 37, 38, 39]

#%% Train the model
for epoch in range(0, config_3.epochs):
    print("Epoch: {}/{}\n".format(epoch+1, config_3.epochs))
# =============================================================================
#   TRAINING
# =============================================================================
    model_3.train()
    train_loss                  = 0.0
    train_loss_classification   = 0.0
    train_loss_glimpse_change   = 0.0
    acc_correct_attr            = 0
    acc_localization            = 0
    acc_attr_localized          = 0
    train_ave_iou               = 0.0
    total_attr                  = 0
    total_samples               = 0
    glimpses_locs_dims_array    = []
    for i, (images, targets) in enumerate(train_loader):
        translated_images, targets, bbox_targets                    = images.to(device), targets[0].float().to(device), targets[1].to(device)
# =============================================================================
#       DATA STRUCTURES to keep track of glimpses
# =============================================================================
        # Data structure to keep track of glimpse locations and dimensions
        # glimpses_locs_dims[batch_size, 4] - where second dimension is 4-sized tuple,
        # representing (x_TopLeftCorner, y_TopLeftCorner, width, height) of each glimpse in the batch
        glimpses_locs_dims      = torch.zeros((targets.shape[0], 4), dtype=torch.int).to(device)
        
        # Data structure to store actual batch of extracted and resized glimpses to be fetched to the network
        # !!!: in order to be able to process the batch of different sized glimpses in one feedforward path through the network,
        # after extracting glimpses from each individual image (according to locations and dimensions of corresponding glimpses)
        # we are resizing all of them to some fixed, pre-determined fixed size, which is determined by config.glimpse_size_fixed parameter!
        # !!!: glimpses_extracted_resized[batch_size, input_channels, fixed_glimpse_height, fixed_glimpse_width]
        glimpses_extracted_resized = torch.zeros((translated_images.shape[0],   translated_images.shape[1], 
                                                  config_3.glimpse_size_fixed[1], config_3.glimpse_size_fixed[0])).to(device) # glimpse_size_fixed[width, height]

        rewards = []
        actions = []
        simscores = []
        # pdb.set_trace()
# =============================================================================
#       M2 Stage: getting initial predicted locations        
# =============================================================================
        # M2 predicts the center of the initial point-of-interest
        pred_locations          = model_2(translated_images) # predictions are in range of (-1, 1)
        initial_location        = torch.clamp(pred_locations, min=lower, max=upper)
        initial_location[:, 0]  = ((initial_location[:, 0] + 1.0)/2.0)*config_3.full_res_img_size[0] # convert from (-1, 1) range to (0, img_width) range
        initial_location[:, 1]  = ((initial_location[:, 1] + 1.0)/2.0)*config_3.full_res_img_size[1] # convert from (-1, 1) range to (0, img_height) range

        # initial_location = torch.rand_like(initial_location.float())
        # initial_location[:, 0] *= config_3.full_res_img_size[1]
        # initial_location[:, 1] *= config_3.full_res_img_size[0]
        # initial_location = initial_location.int()
        
        # then, we translate the center of the initial point-of-interest into (x_TopLeftCorner, y_TopLeftCorner, width, height)       
        glimpses_locs_dims[:, 0]    = initial_location[:, 0] + 0.5 - (config_3.glimpse_size_init[0]/2.0) #!!!: we add 0.5 for printing purposes only. See base_version/AVS_train_a2.py for explanation
        glimpses_locs_dims[:, 1]    = initial_location[:, 1] + 0.5 - (config_3.glimpse_size_init[1]/2.0)
        glimpses_locs_dims[:, 2]    = config_3.glimpse_size_init[0]
        glimpses_locs_dims[:, 3]    = config_3.glimpse_size_init[1]
        if i == 0:
            glimpses_locs_dims_array.append(glimpses_locs_dims.clone().detach())
        
# =============================================================================
#       M3 Stage
# =============================================================================
        optimizer.zero_grad()
        model_3.clear_memory()
        loss_classification     = torch.tensor([0.0]) #None
        loss_glimpse_change     = None
        
        for g in range(config_3.num_glimpses):
            # Extract and resize the batch of glimpses based on their current locations and dimensions.
            glimpses_extracted_resized = extract_and_resize_glimpses_for_batch(translated_images, glimpses_locs_dims,
                                                                               config_3.glimpse_size_fixed[1], config_3.glimpse_size_fixed[0]) # glimpse_size_fixed[width, height]

            # 1.30 - tao88, save glimpses
            # save_image(glimpses_extracted_resized[0], '/home/nano01/a/tao88/1.30/glimpse_{}.jpg'.format(g))
            # pdb.set_trace()
            # Process the batch of extracted and resized glimpses through the network.
            # glimpses_change_actions[batch_size, 4] - where second dimension is 4-sized tuple, 
            # representing (dx-, dx+, dy-, dy+) changes of every glimpse in x and y directions
            output_predictions, glimpses_change_actions = model_3(glimpses_extracted_resized) # glimpses_change_actions has values (0, 1)
            
            # Estimate classification loss.
            # if g == (config_3.num_glimpses - 1):
            #     loss_classification  = bce_loss(output_predictions, targets)
            # else:
            #     loss_classification += bce_loss(output_predictions, targets)
                
            # Estimate rewards for RL agent.
            #@Version 6 extension E:
            # change of similarity metric
            
            similarity      = torch.relu(torch.cosine_similarity(output_predictions >= config_3.attr_detection_th, targets)).clone().detach()
            action_cost     = glimpses_change_actions.mean(dim=-1).clone().detach()
            # pdb.set_trace()
            if g == 0:
                max_sim_val     = similarity.clone().detach()   # maximum value of similarity detected over all glimpses
                max_sim_gid     = torch.zeros_like(similarity)  # glimpse at which maximum value of similarity is detected (initialize at 0 for the first glimpse)
            else:
                new_max_sim     = similarity >= max_sim_val
                max_sim_val[new_max_sim] = similarity[new_max_sim]
                max_sim_gid[new_max_sim] = g
            # if i == 0:
            #     print("batch: {}, glimpse: {}, cosine_sim: \n{}\n".format(i, g, similarity))
            #     print("max_sim_gid".format(max_sim_gid))
            actions.append(action_cost.clone().detach())
            simscores.append(similarity.clone().detach())
            
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
                        # pdb.set_trace()
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
                    # if i == 0:
                    #     print("glimpse: {}, rewards: \n{}\n".format(g_replay, rewards[-1]))
    
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
            if i == 0:
                glimpses_locs_dims_array.append(glimpses_locs_dims.clone().detach())
        
        pdb.set_trace()

        # Estimate the RL agent loss.
        loss_glimpse_change = model_3.compute_rl_loss()
        # Backpropagate the errors for all iterations.
        loss                = loss_glimpse_change #+ loss_classification
        loss.backward()
        optimizer.step()
        

        pred_labels         = (output_predictions >= config_3.attr_detection_th).float()
        total_attr         += targets.numel()
        total_samples      += targets.size(0)
        train_loss         += loss.item()
        train_loss_classification   += loss_classification.item()
        train_loss_glimpse_change   += loss_glimpse_change.item()
        
        iou = region_iou(glimpses_locs_dims.clone().detach(), bbox_targets).diag()
        train_ave_iou       += iou.sum().item()
        
        correct_attr         = pred_labels == targets
        correct_tp_loc       = iou >= config_3.iou_th
        acc_correct_attr    += correct_attr.sum().item()
        acc_localization    += correct_tp_loc.sum().item()
        acc_attr_localized  += correct_attr[correct_tp_loc, :].sum().item()


    print("Train Loss: {:.3f} | {:.3f} | {:.3f}\n".format(
        (train_loss/(i+1)), (train_loss_classification/(i+1)), (train_loss_glimpse_change/(i+1))))
    print("Attribute Acc: {:.4f} [{}/{}] | GT Loc: {:.4f} [{}/{}] | Attr Loc: {:.4f} [{}/{}]\n.".format(
        (100.*acc_correct_attr/total_attr), acc_correct_attr, total_attr,
        (100.*acc_localization/total_samples), acc_localization, total_samples,
        (100.*acc_attr_localized/total_attr), acc_attr_localized, total_attr))
    log_dict['train_loss'].append(train_loss/(i+1))
    log_dict['train_loss_classification'].append(train_loss_classification/(i+1))
    log_dict['train_loss_glimpse_change'].append(train_loss_glimpse_change/(i+1))
    log_dict['train_acc_correct_attr'].append(100.*acc_correct_attr/total_attr)
    log_dict['train_acc_localization'].append(100.*acc_localization/total_samples)
    log_dict['train_acc_attr_localized'].append(100.*acc_attr_localized/total_attr)
    print("IoU@{}: Average: {:.3f} | TPR: {:.4f} [{}/{}]\n".format(
        config_3.iou_th, (train_ave_iou/total_samples),
        (1.*acc_localization/total_samples), acc_localization, total_samples))


# =============================================================================
#   EVALUATION
# =============================================================================
    model_3.eval()
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
    with torch.no_grad():
        for i, (images, targets) in enumerate(valid_loader):
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
            if i == len(valid_loader) - 1:
                glimpses_locs_dims_array.append(glimpses_locs_dims.clone().detach())
# =============================================================================
#       M3 Stage
# =============================================================================
            model_3.clear_memory()
            loss_classification     = torch.tensor([0.0]) #None
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
                # if g == (config_3.num_glimpses - 1):
                #     loss_classification  = bce_loss(output_predictions, targets)
                # else:
                #     loss_classification += bce_loss(output_predictions, targets)
                
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
                # if i == 0:
                #     print("batch: {}, glimpse: {}, cosine_sim: \n{}\n".format(i, g, similarity))
                #     print("max_sim_gid".format(max_sim_gid))
                actions.append(action_cost.clone().detach())
                simscores.append(similarity.clone().detach())
                
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
                        # if i == 0:
                        #     print("glimpse: {}, rewards: \n{}\n".format(g_replay, rewards[-1]))
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
                if i == len(valid_loader) - 1:
                    glimpses_locs_dims_array.append(glimpses_locs_dims.clone().detach())
        
            # Estimate the RL agent loss.
            loss_glimpse_change = model_3.compute_rl_loss()
            loss                = loss_glimpse_change #+ loss_classification

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
    log_dict['test_loss'].append(test_loss/(i+1))
    log_dict['test_loss_classification'].append(test_loss_classification/(i+1))
    log_dict['test_loss_glimpse_change'].append(test_loss_glimpse_change/(i+1))
    log_dict['test_acc_correct_attr'].append(100.*acc_correct_attr/total_attr)
    log_dict['test_acc_localization'].append(100.*acc_localization/total_samples)
    log_dict['test_acc_attr_localized'].append(100.*acc_attr_localized/total_attr)
    print("IoU@{}: Average: {:.3f} | TPR: {:.4f} [{}/{}]\n".format(
        config_3.iou_th, (test_ave_iou/total_samples), 
        (1.*acc_localization/total_samples), acc_localization, total_samples))

        
    if optimizer.param_groups[0]['lr'] > config_3.lr_min:
        lr_scheduler.step()

    # Storing results
    # ckpt = {}
    # ckpt['model']   = model_3.state_dict()
    # ckpt['log']     = log_dict
    # torch.save(ckpt, config_3.ckpt_dir)

# # Plotting statistics
# plot_dir = config_3.save_dir
# if not os.path.exists(plot_dir): os.makedirs(plot_dir)
# epochs = range(1, config_3.epochs+1)
# plot_curve(epochs, log_dict['train_loss'], 'training loss', 'epoch', 'train loss',     plot_dir + 'train_loss.png')
# plot_curve(epochs, log_dict['train_acc'],  'Performance',   'epoch', 'train accuracy', plot_dir + 'train_acc.png')
# plot_curve(epochs, log_dict['test_acc'],   'Performance',   'epoch', 'test accuracy',  plot_dir + 'test_acc.png')
# plot_curve(epochs, log_dict['test_loss'],  'testing loss',  'epoch', 'test loss',      plot_dir + 'test_loss.png')