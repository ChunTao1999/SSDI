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

from utils_custom_tvision_functions import get_dataloaders, plot_curve
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
from model_lstm_masks import customizable_LSTM as LSTM_masks
from weight_init import weight_init
# Debug
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import pdb
# PiCIE-related
from commons import *
import argparse
from modules import fpn


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
train_loader, loss_weights    = get_dataloaders(config_3, loader_type=config_3.train_loader_type) # trainval split
valid_loader                  = get_dataloaders(config_3, loader_type=config_3.valid_loader_type) # test split

# Loss(es)
bce_loss                    = nn.BCEWithLogitsLoss(pos_weight=loss_weights.to(device))  
ce_loss                     = nn.CrossEntropyLoss()

# Foveation models (no need foveation)

#%% LSTM-related
# LSTM configs
config_4 = config_for_LSTM

# LSTM model (M4)
model_4 = LSTM_masks(config_4)
if config_3.data_parallel: # for now, don't use DataParallel with LSTM model
    model_4 = nn.DataParallel(model_4)
model_4.cuda()
model_4.apply(weight_init)
print("Model M4: \n", "Training M4 from scratch!\n")
print(model_4)
# check number of parameters and number of parameters that require gradient
print("Number of params: {}".format(sum(p.numel() for p in model_4.parameters())))
print("Number of trainable params: {}".format(sum(p.numel() for p in model_4.parameters() if p.requires_grad)))
model_4.train()
# pdb.set_trace() # done

# LSTM losses
mse_loss = nn.MSELoss(reduction='mean')
cosine_embedding_loss = nn.CosineEmbeddingLoss(reduction='mean')
# Contrastive Loss function
def contrastive_loss(p_tensor, r_tensor):
    p_norm = p_tensor / p_tensor.norm(dim=1)[:,None]
    r_norm = r_tensor / r_tensor.norm(dim=1)[:,None]
    sim_mat = torch.mm(p_norm, r_norm.transpose(0, 1))
    temp_para = 1
    contr_loss = 0
    for i in range(p_tensor.shape[0]): # for each p, calculate a contrastive loss across all r's
        nom = torch.exp(sim_mat[i,i] / temp_para)
        denom = torch.sum(torch.exp(sim_mat[i,:] / temp_para)) - nom
        p_loss = -torch.log(nom / denom)
        contr_loss += p_loss
    contr_loss /= p_tensor.shape[0]
    return contr_loss

#%% LSTM Optimizer
optimizer_M4 = torch.optim.Adam(model_4.parameters(), lr=config_4.lr_start, weight_decay=config_4.weight_decay)
lr_scheduler_M4 = torch.optim.lr_scheduler.MultiStepLR(optimizer_M4, gamma=config_4.gamma, milestones=config_4.milestones, verbose=True)

#%% Resnet+FPN model for glimpse context (no need, use gt masks for LSTM training)
# def parse_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducability.')
#     parser.add_argument('--arch', type=str, default='resnet18')
#     parser.add_argument('--pretrain', action='store_true', default=False)
#     parser.add_argument('--FPN_with_classifier', action='store_true', default=False)

#     parser.add_argument('--restart', action='store_true', default=False)
#     parser.add_argument('--eval_only', action='store_true', default=False)
#     parser.add_argument('--eval_path', type=str, default='.')
#     parser.add_argument('--model_finetuned', action='store_true', default=False)
#     parser.add_argument('--finetuned_model_path',type=str, default='')

#     parser.add_argument('--in_dim', type=int, default=128)
#     parser.add_argument('--K_train', type=int, default=20)

#     parser.add_argument('--num_epoch', type=int, default=10) 
#     parser.add_argument('--optim_type', type=str, default='Adam')
#     parser.add_argument('--lr', type=float, default=1e-4)
#     return parser.parse_args()

# args = parse_arguments()
# args.save_model_path = args.eval_path # 2.20.2023 - use 20 epochs model

# model_FPN = fpn.PanopticFPN(args)
# model_FPN = nn.DataParallel(model_FPN) # comment for same device
# model_FPN = model_FPN.cuda()
# checkpoint  = torch.load(args.eval_path)
# model_FPN.load_state_dict(checkpoint['state_dict'])
# print('Loaded checkpoint. [epoch {}]'.format(checkpoint['epoch']))
# model_FPN.eval()


#%% Logging the losses over epochs
log_dict = {"train_loss_lstm": []} # 2.20.2023 - tao88


#%% AVS-specific functions, methods, and parameters
from AVS_functions import location_bounds, extract_and_resize_glimpses_for_batch, crop_five, extract_and_resize_masks

lower, upper = location_bounds(glimpse_w=config_3.glimpse_size_init[0], input_w=config_3.full_res_img_size[0]) #TODO: make sure this works on both x and y coordinates


#%% Training the LSTM module
# log_dict and save_path
log_dict = {"train_loss_lstm": []}
save_path = './bi-lstm_models_crop_masks_lr={}'.format(config_4.lr_start)
if config_4.use_contr_loss:
    save_path = os.path.join(save_path, 'contr_loss')
else:
    save_path = os.path.join(save_path, 'mse_loss') 
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Start training
for epoch in range(0, config_3.epochs):
    print("\nEpoch: {}/{}".format(epoch+1, config_3.epochs))
    train_loss_lstm = 0.0

    for i, (indices, images, targets) in enumerate(train_loader):
        translated_images, masks = images.to(device), targets[4].to(device)
        # indices: indices of images in the dataset (128, 1);
        # images: translated images (128, 3, 256, 256); 
        # targets[4]: masks (128, 1, 256, 256); # can be set to wild or aligned masks

        # some error in compute_rl_loss for last batch
        if i == len(train_loader) - 1:
            continue

        # round the aligned masks
        # masks = torch.round(masks * 7).long() # round the aligned masks
        masks = torch.round(masks * 7) # need float to feed extracted masks directly to LSTM

        # crop the five patches from each image and each mask
        batch_patches = crop_five(translated_images, resized_height=256, resized_width=256) # (128, 5, 3, 256, 256)
        masks_extracted_resized = extract_and_resize_masks(masks, resized_height=64, resized_width=64) # (128, 5, 1, 64, 64)

        # transform from cropped masks to their semantics
        num_pixels = masks_extracted_resized.shape[3] * masks_extracted_resized.shape[4]
        masks_extracted_resized = masks_extracted_resized.flatten(start_dim=2, end_dim=4) # (128, 5, 4096)
        mask_semantics = torch.nn.functional.one_hot(masks_extracted_resized.long()).sum(dim=2) # (128, 5, 7)
        mask_semantics_ratio = (mask_semantics.float()) / num_pixels # (128, 5, 7)

        # Initialize the LSTM before glimpse iteration
        optimizer_M4.zero_grad()
        h0, c0 = model_4._init_hidden(translated_images.shape[0])
        # (1, 128, 128) (D*num_layers, N, H_out), (1, 128, 128) (D*num_layers, N, H_cell)
        loss_lstm = torch.tensor([0.0]).cuda()

        # Pass the extracted masks into the LSTM model
        output_lstm, (hn, cn) = model_4(masks_extracted_resized, (h0, c0))
        # In the Bi-directional case, output_lstm will contain a concatenation of the foward and reverse hidden states at each timestep in the sequence. Be careful in computing the LSTM loss.
        # Forward loss
        loss_lstm = mse_loss(output_lstm[:, :-1, :7], mask_semantics_ratio[:, 1:, :]) #  both (128, 4, 7)
        # Backward loss
        loss_lstm += mse_loss(output_lstm[:, 1:, 7:], mask_semantics_ratio[:, :-1, :]) # both (128, 4, 7)

        # M4 backpropagate after all patch iterations
        loss_lstm.backward()
        optimizer_M4.step()
        train_loss_lstm += loss_lstm.item()

        # Print every 100 batches
        if (i+1) % 100 == 0:
            print("Current process: batch: {}/{}, current batch loss: {}".format(i+1, len(train_loader), loss_lstm.item()))

    # Adjust lr every epoch
    # if optimizer_M4.param_groups[0]['lr'] > config_4.lr_min:
    lr_scheduler_M4.step()
    
    # Print losses
    print("Train Loss (LSTM): {:.3f}\n".format(train_loss_lstm / (i+1)))
    log_dict['train_loss_lstm'].append(train_loss_lstm / (i+1))

    # Storing models and results
    print("Saving model: epoch: {}/{}\n".format(epoch+1, config_3.epochs))
    if (epoch+1) % 10 == 0: 
        torch.save({'epoch': epoch+1, 
                    'state_dict': model_4.state_dict(),
                    'optimizer' : optimizer_M4.state_dict(),
                    'log_dict': log_dict
                    },
                    os.path.join(save_path, 'checkpoint_{}.pth.tar'.format(epoch+1)))
            
    # torch.save({'epoch': epoch+1, 
    #             'state_dict': model_4.state_dict(),
    #             'optimizer' : optimizer_M4.state_dict(),
    #             'log_dict': log_dict
    #             },
    #             os.path.join(save_path, 'checkpoint_{}.pth.tar'.format(epoch+1)))

print('\nFinished Training\n')