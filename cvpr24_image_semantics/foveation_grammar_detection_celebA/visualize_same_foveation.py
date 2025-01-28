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
from modules import fpn 

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
train_loader, loss_weights    = get_dataloaders(config_3, loader_type=config_3.train_loader_type)
valid_loader                  = get_dataloaders(config_3, loader_type=config_3.valid_loader_type)

# Loss(es)
# bce_loss                      = nn.BCEWithLogitsLoss(pos_weight=loss_weights.to(device))  
ce_loss                     = nn.CrossEntropyLoss()

# Model
model_1 = VGG_for_M1(config_1)
model_1 = nn.DataParallel(model_1) # tao88
model_1 = model_1.to(device)
ckpt_1  = torch.load(config_3.ckpt_dir_model_M1)
new_ckpt_1 = OrderedDict() # tao88
for k, v in ckpt_1['model'].items():
    name = "".join(["module.", k])
    new_ckpt_1[name] = v
model_1.load_state_dict(new_ckpt_1)
for p in model_1.parameters():
    p.requires_grad_(False)
model_1.eval()
print("Model M1:\n", "Loaded from: {}\n".format(config_3.ckpt_dir_model_M1))

model_2 = context_network(in_channels=config_2.in_num_channels, res_size = config_2.low_res, avg_size = config_2.avg_size)
model_2 = nn.DataParallel(model_2) # tao88
model_2 = model_2.to(device)
ckpt_2  = torch.load(config_3.ckpt_dir_model_M2)
new_ckpt_2 = OrderedDict() # tao88
for k, v in ckpt_2['model'].items():
    name = "".join(["module.", k])
    new_ckpt_2[name] = v
model_2.load_state_dict(new_ckpt_2)
for p in model_2.parameters():
    p.requires_grad_(False)
model_2.eval()
print("Model M2:\n", "Loaded from: {}\n".format(config_3.ckpt_dir_model_M2))

model_3 = VGG_for_M3(config_3)
model_3 = nn.DataParallel(model_3) # tao88
model_3 = model_3.to(device)
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
    new_ckpt_3 = OrderedDict() # tao88
    for k, v in ckpt_3['model'].items():
        name = "".join(["module.", k])
        new_ckpt_3[name] = v
    model_3.load_state_dict(new_ckpt_3)
    # print("Loaded from: {}\n".format(config_3.ckpt_dir_model_M3))
elif config_3.initialize_M3 == 'from_scratch':
    print("Training M3 from scratch!\n")
else:
    raise ValueError("Unknown value for argument config.initialize_M3: ({}). ".format(config_3.initialize_M3) +
                     "Acceptable options are: ('from_M1', 'from_checkpoint', 'from_scratch').")
for n, p in model_3.named_parameters():
    if 'features' in n or 'classifier' in n:
        p.requires_grad_(False)
print("Model M3:\n", "Loaded from: {}\n".format(config_3.ckpt_dir_model_M3))

# Optimizer
# optimizer = torch.optim.Adam(model_3.parameters(), lr=config_3.lr_start, weight_decay=config_3.weight_decay)
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma = 0.1, milestones=config_3.milestones)

# Logger dictionary
# log_dict     = {'train_loss':[], 
#                 'train_loss_glimpse_mask': [], # 2.13.2023 - tao88
#                 'train_loss_glimpse_change': [], 
#                 'train_acc_correct_attr':[],
#                 'train_acc_localization':[],
#                 'train_acc_attr_localized':[], 
#                 'test_loss':[],
#                 'test_loss_glimpse_change': [],
#                 'test_acc_correct_attr':[],
#                 'test_acc_localization':[],
#                 'test_acc_attr_localized':[]}
log_dict = {"train_loss_glimpse_mask": []} # 2.13.2023 - tao88
if not os.path.exists(config_3.save_dir): os.makedirs(config_3.save_dir)


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
# model_FPN, optimizer = get_model_and_optimizer(args, logger) # note dataparallel objects (fixed 2.13.2023)
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma = 0.1, milestones=[40, 80])

model_FPN = fpn.PanopticFPN(args)
model_FPN = nn.DataParallel(model_FPN) # comment for same device
model_FPN = model_FPN.cuda()
checkpoint = torch.load(args.eval_path)
model_FPN.load_state_dict(checkpoint['state_dict'])
logger.info('Loaded checkpoint. [epoch {}]'.format(checkpoint['epoch']))

# First try training on glimpses without freezing any layers
# check number of parameters and number of parameters that require gradient
print("Number of params: {}".format(sum(p.numel() for p in model_FPN.parameters())))
print("Number of trainable params: {}".format(sum(p.numel() for p in model_FPN.parameters() if p.requires_grad)))

model_FPN.eval()
# pdb.set_trace() # done


#%% AVS-specific functions, methods, and parameters
from AVS_functions import location_bounds, extract_and_resize_glimpses_for_batch

lower, upper = location_bounds(glimpse_w=config_3.glimpse_size_init[0], input_w=config_3.full_res_img_size[0]) #TODO: make sure this works on both x and y coordinates

#%% Train the model
for epoch in range(0, args.num_epoch):
    # if epoch > 0: break # for debug
    print("\nEpoch: {}/{}".format(epoch+1, args.num_epoch))
# =============================================================================
#   TRAINING
# =============================================================================
    model_3.eval()
    model_3.module.clear_memory() # tao88, add ".module"
    # train_loss_glmipse_mask     = 0.0 # 2.13.2023 - tao88
    test_loss                   = 0.0
    test_loss_classification    = 0.0
    test_loss_glimpse_change    = 0.0
    acc_correct_attr            = 0
    acc_localization            = 0
    acc_attr_localized          = 0
    test_ave_iou                = 0.0
    total_attr                  = 0
    total_samples               = 0
    captured_batch_images       = None
    captured_batch_gt_bboxes    = None
    glimpses_locs_dims_array    = []
    glimpses_iou_array          = []
    glimpses_array              = []
    similarity_array            = []
    rewards_array               = []
    iou_array                   = []
    
    for i, (index, images, targets) in enumerate(train_loader): # target_type=['attr', 'bbox', 'pred_init_glimpse_locs', 'landmarks', 'masks']
        if i == len(train_loader) - 1:
            continue # 2.15.2023 - tao88, in order to account for error in compute_rl_loss
        translated_images, targets, bbox_targets, masks = images.to(device), targets[0].float().to(device), targets[1].to(device), targets[4].unsqueeze(1).to(device) # translated images (128, 3, 256, 256); targets (128, 40); bbox_targets (128, 4); masks (128, 1, 256, 256)
        # pdb.set_trace()
        masks = torch.round(masks * 7).long() # preprocess the aligned masks

        # 2.13.2023 - visualize some wild images and their wild masks to make sure of correspondance
        if i == 0:
            save_image(translated_images[0], '/home/nano01/a/tao88/2.17/same_foveation/corrupted_img.png')
            fig = plt.figure(figsize=(10,10))
            fig.add_subplot(1, 1, 1)
            plt.imshow(masks[0].squeeze(0).cpu().numpy())
            plt.savefig('/home/nano01/a/tao88/2.17/same_foveation/corrupted_mask.png')

# =============================================================================
#       DATA STRUCTURES to keep track of glimpses
# =============================================================================
        glimpses_locs_dims      = torch.zeros((targets.shape[0], 4), dtype=torch.int).to(device)

        glimpses_extracted_resized = torch.zeros((translated_images.shape[0],   translated_images.shape[1], 
                                                  config_3.glimpse_size_fixed[1], config_3.glimpse_size_fixed[0])) # glimpse_size_fixed[width, height]

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

# =============================================================================
#       M3 Stage
# =============================================================================
        model_3.module.clear_memory()
        loss_glimpse_mask = torch.tensor([0.0]).cuda()
        # loss_classification     = None
        loss_glimpse_change = None
    
        # 2.17.2023 - tao88, load the saved_glimpses_locs_dims_array for correct images
        glimpses_locs_dims_array = torch.load("/home/nano01/a/tao88/2.17/glimpses_locs_dims_array.pt")
        # pdb.set_trace()

        for g in range(config_3.num_glimpses):
            glimpses_locs_dims = glimpses_locs_dims_array[g]
            # if i == 0:
            #     glimpses_locs_dims_array.append(glimpses_locs_dims.clone().detach())
            # Extract and resize the batch of glimpses based on their current locations and dimensions.
            glimpses_extracted_resized, glimpses_extracted_resized_FPN = extract_and_resize_glimpses_for_batch(translated_images, glimpses_locs_dims, config_3.glimpse_size_fixed[1], config_3.glimpse_size_fixed[0], 256, 256, copies=2) # glimpse_size_fixed[width, height]

            masks_extracted_resized = extract_and_resize_glimpses_for_batch(masks, glimpses_locs_dims, 64, 64, copies=1, interpolation_mode=transforms.InterpolationMode.NEAREST) # use the correct interpolation for mask reshape

            # visualize the match between glimpses and their cropped_and_resized_masks
            if i == 0:
                save_image(glimpses_extracted_resized[0], "/home/nano01/a/tao88/2.17/same_foveation/img_corrupted_0_g_{}.png".format(g))
                fig = plt.figure(figsize=(10,10))
                fig.add_subplot(1, 1, 1)
                plt.imshow(masks_extracted_resized[0].squeeze(0).cpu().numpy())
                plt.savefig('/home/nano01/a/tao88/2.17/same_foveation/img_corrupted_0_gmask_{}.png'.format(g))

            # we now have the glimpses and their corresponding cropped masks. We pass the glimpses throught model_FPN, and finetune model_FPN with cropped masks as labels
            masks_predicted = model_FPN(glimpses_extracted_resized_FPN) # (128, 20, 64, 64)
            # loss_glimpse_mask += ce_loss(masks_predicted, masks_extracted_resized.squeeze(1)) # (128, 20, 64, 64), (128, 64, 64)
            # pdb.set_trace()
            
            # 2.17 - tao88: visualize glimpse classification
            lbl_predicted = masks_predicted.topk(1, dim=1)[1]
            # pdb.set_trace()
            if i == 0:
                fig = plt.figure(figsize=(10,10))
                fig.add_subplot(1, 1, 1)
                plt.imshow(lbl_predicted[0].squeeze(0).cpu().numpy())
                plt.savefig('/home/nano01/a/tao88/2.17/same_foveation/img_corrupted_0_gmask_pred_{}.png'.format(g))

            output_predictions, glimpses_change_actions = model_3(glimpses_extracted_resized)
        
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
            model_3.module.store_all_rewards(rewards)
        
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
        
        #  # Backward the glimpse mask classification loss
        # optimizer.zero_grad()
        # loss_glimpse_mask.backward() # this takes long time
        # optimizer.step()
        # train_loss_glmipse_mask += loss_glimpse_mask.item()

        # Save the glimpses_locs_dims_array
        # torch.save(glimpses_locs_dims_array, "/home/nano01/a/tao88/2.17/glimpses_locs_dims_array.pt")
        pdb.set_trace()

        if (i+1) % 100 == 0:
            print("Current process: batch: {}/{}".format(i+1, len(train_loader)))
            # print("Loss: {}".format(train_loss_glmipse_mask))

        # Estimate the RL agent loss.
        loss_glimpse_change = model_3.module.compute_rl_loss()
        loss                = loss_glimpse_change

        pred_labels         = (output_predictions >= config_3.attr_detection_th).float()
        total_attr         += targets.numel()
        total_samples      += targets.size(0)
        test_loss          += loss.item()
        test_loss_glimpse_change   += loss_glimpse_change.item()
        
        iou = region_iou(glimpses_locs_dims.clone().detach(), bbox_targets).diag()
        test_ave_iou        += iou.sum().item()

        correct_attr         = pred_labels == targets
        correct_tp_loc       = iou >= config_3.iou_th
        acc_correct_attr    += correct_attr.sum().item()
        acc_localization    += correct_tp_loc.sum().item()
        acc_attr_localized  += correct_attr[correct_tp_loc, :].sum().item()
    
    # End of epoch
    # lr_scheduler update
    # if optimizer.param_groups[0]['lr'] > config_3.lr_min:
    #     lr_scheduler.step()
    
    # Print losses
    # print("Train Loss (glimse mask classification): {:.3f}\n".format(train_loss_glmipse_mask / (i+1)))
    # log_dict['train_loss_glimpse_mask'].append(train_loss_glmipse_mask/(i+1))
    
    # Storing results
    # print("Saving model: epoch: {}/{}\n".format(epoch+1, config_3.epochs))
    # # if (epoch+1) % 2 == 0: 
    # # save every epoch
    # torch.save({'epoch': epoch+1, 
    #             'args' : args,
    #             'state_dict': model_FPN.state_dict(),
    #             'optimizer' : optimizer.state_dict(),
    #             },
    #             os.path.join('./glimpse_classification_models', 'checkpoint_{}.pth.tar'.format(epoch)))
            
    # torch.save({'epoch': epoch+1, 
    #             'args' : args,
    #             'state_dict': model_FPN.state_dict(),
    #             'optimizer' : optimizer.state_dict(),
    #             },
    #             os.path.join('./glimpse_classification_models', 'checkpoint.pth.tar'))
pdb.set_trace()