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
train_loader, loss_weights    = get_dataloaders(config_3, loader_type=config_3.train_loader_type)
valid_loader                  = get_dataloaders(config_3, loader_type=config_3.valid_loader_type)

# Loss(es)
bce_loss                    = nn.BCEWithLogitsLoss(pos_weight=loss_weights.to(device))  
ce_loss                     = nn.CrossEntropyLoss()


#%% Foveation models
if not config_3.data_parallel:
    model_1 = VGG_for_M1(config_1).to(device)
    ckpt_1  = torch.load(config_3.ckpt_dir_model_M1)
    model_1.load_state_dict(ckpt_1['model'])
    for p in model_1.parameters():
        p.requires_grad_(False)
    model_1.eval()
    print("Model M1:\n", "Loaded from: {}\n".format(config_3.ckpt_dir_model_M1))
    # print(model_1)

    model_2 = context_network(in_channels=config_2.in_num_channels, res_size = config_2.low_res, avg_size = config_2.avg_size).to(device)
    ckpt_2  = torch.load(config_3.ckpt_dir_model_M2)
    model_2.load_state_dict(ckpt_2['model'])
    for p in model_2.parameters():
        p.requires_grad_(False)
    model_2.eval()
    print("Model M2:\n", "Loaded from: {}\n".format(config_3.ckpt_dir_model_M2))
    # print(model_2)

    model_3 = VGG_for_M3(config_3).to(device)
    print("Model M3:\n")
    ckpt_3 = torch.load(config_3.ckpt_dir_model_M3)
    model_3.load_state_dict(ckpt_3['model'])
    for p in model_3.parameters():
        p.requires_grad_(False)
    model_3.eval()
    print("Loaded from: {}\n".format(config_3.ckpt_dir_model_M3))
    # print(model_3)
else:
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
from AVS_functions import location_bounds, extract_and_resize_glimpses_for_batch

lower, upper = location_bounds(glimpse_w=config_3.glimpse_size_init[0], input_w=config_3.full_res_img_size[0]) #TODO: make sure this works on both x and y coordinates


#%% Test on the models
with torch.no_grad():
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
    captured_batch_images       = None
    captured_batch_gt_bboxes    = None
    glimpses_locs_dims_array    = []
    glimpses_iou_array          = []
    glimpses_array              = []
    similarity_array            = []
    rewards_array               = []
    iou_array                   = []
    # pdb.set_trace()

    for i, (index, images, targets) in enumerate(valid_loader): # target_type=['attr', 'bbox', 'pred_init_glimpse_locs', 'landmarks', 'masks']
        if i == len(valid_loader) - 1:
            continue # 2.15.2023 - tao88, in order to account for error in compute_rl_loss
        translated_images, targets, bbox_targets, corrupt_labels = images.to(device), targets[0].float().to(device), targets[1].to(device), targets[-1].to(device) # translated images (128, 3, 256, 256); targets (128, 40); bbox_targets (128, 4); corrupt_labels (128), 0 for correct, 1 for corrnano01upted

        # masks = torch.round(masks * 7).long() # preprocess the aligned masks

        # pdb.set_trace()
        # 2.13.2023 - visualize some aligned images and their aligned masks to make sure of correspondance
        if i == 0:
            for img_id in range(16):
                if not config_3.landmark_shuffle:
                    if not config_3.use_corrupted_testset:
                        save_path = '/home/nano01/a/tao88/3.9_LSTM/correct/img_{}'.format(img_id)
                    else:
                        save_path = '/home/nano01/a/tao88/3.9_LSTM/corrupted_testset/img_{}'.format(img_id)
                else:
                    save_path = '/home/nano01/a/tao88/3.9_LSTM/corrupted/img_{}'.format(img_id)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                save_image(translated_images[img_id], os.path.join(save_path, 'input_img.png'))
                # fig = plt.figure(figsize=(10,10))
                # fig.add_subplot(1, 1, 1)
                # plt.imshow(masks[img_id].squeeze(0).cpu().numpy(),
                #            interpolation='nearest',
                #            cmap='Paired',
                #            vmin=0,
                #            vmax=6)
                # plt.savefig(os.path.join(save_path, 'input_aligned_mask.png'))
        # pdb.set_trace()


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

# =============================================================================
#       M3 Stage
# =============================================================================
        model_3.clear_memory()
        loss_glimpse_mask = torch.tensor([0.0]).cuda()
        # loss_classification     = None
        loss_glimpse_change = None
    
        # 2.27.2023 - tao88, add functionality of LSTM
        if config_3.data_parallel:
            h_0, c_0 = model_4.module._init_hidden(translated_images.shape[0]) # (1, 128, 7), (1, 128, 11)
        else:
            h_0, c_0 = model_4._init_hidden(translated_images.shape[0]) # (1, 128, 7), (1, 128, 11)

        # 2.26.2023 - tao88: initiate list for distance curves for batch of images
        # dist_16_imgs = [[] for _ in range(16)]
        dist_batch_imgs = [[] for _ in range(translated_images.shape[0])]

        # Start glimpse iteration
        for g in range(config_3.num_glimpses):
            # Extract and resize the batch of glimpses based on their current locations and dimensions.
            glimpses_extracted_resized, glimpses_extracted_resized_FPN = extract_and_resize_glimpses_for_batch(translated_images, glimpses_locs_dims, config_3.glimpse_size_fixed[1], config_3.glimpse_size_fixed[0], 256, 256, copies=2) # glimpse_size_fixed[width, height]

            # masks_extracted_resized = extract_and_resize_glimpses_for_batch(masks, glimpses_locs_dims, 64, 64, copies=1, interpolation_mode=transforms.InterpolationMode.NEAREST) # use the correct interpolation for mask reshape

            # 2.27.2023 - tao88: visualize the match between glimpses and their cropped_and_resized_masks
            if i == 0:
                for img_id in range(16):
                    if not config_3.landmark_shuffle:
                        if not config_3.use_corrupted_testset:
                            save_path = '/home/nano01/a/tao88/3.9_LSTM/correct/img_{}'.format(img_id)
                        else:
                            save_path = '/home/nano01/a/tao88/3.9_LSTM/corrupted_testset/img_{}'.format(img_id)
                    else:
                        save_path = '/home/nano01/a/tao88/3.9_LSTM/corrupted/img_{}'.format(img_id)
                    save_image(glimpses_extracted_resized[img_id], os.path.join(save_path, 'img_{}_g_{}.png'.format(img_id, g)))
                    # fig = plt.figure(figsize=(10,10))
                    # fig.add_subplot(1, 1, 1)
                    # plt.imshow(masks_extracted_resized[img_id].squeeze(0).cpu().numpy(),
                    #            interpolation='nearest',
                    #            cmap='Paired',
                    #            vmin=0,
                    #            vmax=6)
                    # plt.savefig(os.path.join(save_path, 'img_{}_gmask_{}.png'.format(img_id, g)))

            # we now have the glimpses and their corresponding cropped masks. We pass the glimpses throught model_FPN, and finetune model_FPN with cropped masks as labels
            masks_predicted = model_FPN(glimpses_extracted_resized_FPN) # (128, 7, 64, 64)
            # loss_glimpse_mask += ce_loss(masks_predicted, masks_extracted_resized.squeeze(1)) # (128, 7, 64, 64), (128, 64, 64)
            
            # 2.17 - tao88: visualize predicted glimpse classification
            lbl_predicted = masks_predicted.clone().detach().topk(1, dim=1)[1] # (128, 1, 64, 64)
            if i == 0:
                for img_id in range(16):
                    if not config_3.landmark_shuffle:
                        if not config_3.use_corrupted_testset:
                            save_path = '/home/nano01/a/tao88/3.9_LSTM/correct/img_{}'.format(img_id)
                        else:
                            save_path = '/home/nano01/a/tao88/3.9_LSTM/corrupted_testset/img_{}'.format(img_id)
                    else:
                        save_path = '/home/nano01/a/tao88/3.9_LSTM/corrupted/img_{}'.format(img_id)
                    fig = plt.figure(figsize=(10,10))
                    fig.add_subplot(1, 1, 1)
                    plt.imshow(lbl_predicted[img_id].squeeze(0).cpu().numpy(),
                               interpolation='nearest',
                               cmap='Paired',
                               vmin=0,
                               vmax=6)
                    plt.axis('off')
                    plt.savefig(os.path.join(save_path, 'img_{}_gmask_pred_{}.png'.format(img_id, g)))

            # For foveation
            output_predictions, glimpses_change_actions = model_3(glimpses_extracted_resized)
        
            # 2.27.2023 - tao88: from lbl_predicted, transform into semantics predicted and semantics_ratio predicted
            num_pixels = lbl_predicted.shape[2] * lbl_predicted.shape[3]
            lbl_predicted_flattened = lbl_predicted.flatten(start_dim=2, end_dim=3) # (128, 1, 4096)
            g_semantics_pred = torch.nn.functional.one_hot(lbl_predicted_flattened).sum(dim=2) # (128, 1, 7)
            # glimpse semantics ratio and glimpse action
            g_semantics_pred = (g_semantics_pred.float()) / num_pixels # (128, 1, 7)
            g_actions_pred = glimpses_change_actions.clone().detach().unsqueeze(1) # (128, 1, 4)
            g_concat_pred = torch.cat((g_semantics_pred, g_actions_pred), 2) # (128, 1, 11)

            # 2.27.2023 - tao88: let trained LSTM model predict next glimpse semantics and measure distance to gt next glimpse semantics
            if g == 0:
                output_lstm, (h, c) = model_4(g_concat_pred, (h_0, c_0)) # (128, 1, 7), (1, 128, 7), (1, 128, 11)
            else:
                # compute l2 dist between output_lstm and next g_semantics, only for visualization
                for sample_id in range(translated_images.shape[0]):
                    dist_batch_imgs[sample_id].append((output_lstm[sample_id]-g_semantics_pred[sample_id]).norm().item())
                # predict next glimpse semantics
                output_lstm, (h, c) = model_4(g_concat_pred, (h, c))
            # pdb.set_trace() # Done

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
        
        #  # Backward the glimpse mask classification loss
        # optimizer.zero_grad()
        # loss_glimpse_mask.backward() # this takes long timeplt.rc
        # optimizer.step()
        # train_loss_glmipse_mask += loss_glimpse_mask.item()

        # 2.26.2023 - tao88: plot the distance between p_t and r_(t+1), for individual images
        for sample_id in range(16):
            if not config_3.landmark_shuffle:
                if not config_3.use_corrupted_testset:
                    save_path = '/home/nano01/a/tao88/3.9_LSTM/correct/img_{}'.format(sample_id)
                else:
                    save_path = '/home/nano01/a/tao88/3.9_LSTM/corrupted_testset/img_{}'.format(sample_id)
            else:
                save_path = '/home/nano01/a/tao88/3.9_LSTM/corrupted/img_{}'.format(sample_id)
            plot_curve(np.arange(1, 16), dist_batch_imgs[sample_id], 'Dist btw. prediction and glimpse semantics', 'Glimpse iteration', 'L2 distance', os.path.join(save_path, 'plot_dist.png'))

        # 3.8.2023 - tao88: try first plotting together the 16 curves of the 1st batch, blue for correct, red for corrupted
        x, y = [np.arange(1, 16) for _ in range(translated_images.shape[0])], dist_batch_imgs
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
        plt.savefig('/home/nano01/a/tao88/3.9_LSTM/curves_all.png')

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
        plt.plot(np.arange(1, 16), dist_correct_imgs, c='b', label=corruptness[0], linewidth=4)
        plt.plot(np.arange(1, 16), dist_corrupted_imgs, c='r', label=corruptness[1], linewidth=4)
        plt.title('Averaged dist btw. prediction and glimpse semantics')
        plt.xlabel('Glimpse iteration')
        plt.ylabel('L2 distance')
        plt.legend()
        plt.savefig('/home/nano01/a/tao88/3.9_LSTM/curves_together_avg.png')
        pdb.set_trace()


        if (i+1) % 100 == 0:
            print("Current process: batch: {}/{}".format(i+1, len(train_loader)))
            # print("Loss: {}".format(train_loss_glmipse_mask))

        # Estimate the RL agent loss.
        loss_glimpse_change = model_3.compute_rl_loss()
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
    
pdb.set_trace()
# %%
