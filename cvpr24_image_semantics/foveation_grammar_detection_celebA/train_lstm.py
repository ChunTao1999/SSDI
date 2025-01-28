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
train_loader, loss_weights    = get_dataloaders(config_3, loader_type=config_3.train_loader_type)
# valid_loader                  = get_dataloaders(config_3, loader_type=config_3.valid_loader_type)

# Loss(es)
bce_loss                    = nn.BCEWithLogitsLoss(pos_weight=loss_weights.to(device))  
ce_loss                     = nn.CrossEntropyLoss()

# Foveation models
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


#%% LSTM-related
# LSTM configs
config_4 = config_for_LSTM

# LSTM model (M4)
model_4 = LSTM(config_4)
if config_3.data_parallel:
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

# Optimizer
optimizer_M4 = torch.optim.Adam(model_4.parameters(), lr=config_4.lr_start, weight_decay=config_4.weight_decay)
lr_scheduler_M4 = torch.optim.lr_scheduler.MultiStepLR(optimizer_M4, gamma=config_4.gamma, milestones=config_4.milestones, verbose=True)

#%% Resnet+FPN model for glimpse context (may not need)
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


#%% Logging
# Set logger
logger = set_logger('./train_Resnet.log')
log_dict = {"train_loss_lstm": []} # 2.20.2023 - tao88
# if not os.path.exists(config_3.save_dir): os.makedirs(config_3.save_dir)


#%% AVS-specific functions, methods, and parameters
from AVS_functions import location_bounds, extract_and_resize_glimpses_for_batch

lower, upper = location_bounds(glimpse_w=config_3.glimpse_size_init[0], input_w=config_3.full_res_img_size[0]) #TODO: make sure this works on both x and y coordinates


#%% Training the LSTM module
log_dict = {"train_loss_lstm": []}
save_path = './lstm_models_1e-4'
if config_4.use_contr_loss:
    save_path = os.path.join(save_path, 'contr_loss')
else:
    save_path = os.path.join(save_path, 'mse_loss') 
# Start training
for epoch in range(0, config_3.epochs):
    print("\nEpoch: {}/{}".format(epoch+1, config_3.epochs))
    train_loss_lstm = 0.0

    for i, (indices, images, targets) in enumerate(train_loader):
        translated_images = images.to(device)
        masks, glimpses, glimpse_masks, glimpse_actions = targets[4].to(device), targets[5].to(device), targets[6].squeeze().to(device), targets[7].to(device)
        # indices: indices of images in the dataset (128, 1);
        # images: translated images (128, 3, 256, 256); 
        # targets[0]: targets (128, 40); 
        # targets[1]: bbox_targets (128, 4);
        # targets[4]: masks (128, 1, 256, 256); # can be set to wild or aligned masks
        # targets[5]: glimpses (128, 16, 3, 256, 256);
        # targets[6]: glimpse_masks (128, 16, 64, 64); # if resized, (128, 16, 256, 256)
        # targets[7]: glimpse_actions (128, 16, 4)

        # some error in compute_rl_loss for last batch
        if i == len(train_loader) - 1:
            continue

        # 2.22.2023 - tao88
        # visualize some correspondances between face images, glimpses, and glimpse masks
        # save_image(translated_images[0], '/home/nano01/a/tao88/2.22/img_0/orig_img.png')
        # save_image(translated_images[1], '/home/nano01/a/tao88/2.22/img_1/orig_img.png')
        # save_image(masks[0], '/home/nano01/a/tao88/2.22/img_0/orig_mask.png')
        # save_image(masks[1], '/home/nano01/a/tao88/2.22/img_1/orig_mask.png')
        # for g in range(16):
        #     save_image(glimpses[0][g], '/home/nano01/a/tao88/2.22/img_0/g_{}.png'.format(g))
        #     save_image(glimpses[1][g], '/home/nano01/a/tao88/2.22/img_1/g_{}.png'.format(g))
        #     save_image(glimpse_masks[0][g], '/home/nano01/a/tao88/2.22/img_0/gmask_{}.png'.format(g))
        #     save_image(glimpse_masks[1][g], '/home/nano01/a/tao88/2.22/img_1/gmask_{}.png'.format(g))
        # print(glimpse_actions[0])

        # pdb.set_trace() # verified
        
        # 2.22.2023 - tao88
        # round the masks so that the values are in between 0 and 6 (7 classes)
        glimpse_masks = torch.round(glimpse_masks * 7).long() # (128, 16, 64, 64)
        num_pixels = glimpse_masks.shape[2] * glimpse_masks.shape[3]
        glimpse_masks = glimpse_masks.flatten(start_dim=2, end_dim=3) # (128, 16, 4096)
        # count occurrences
        glimpse_semantics = torch.nn.functional.one_hot(glimpse_masks).sum(dim=2) # (128, 16, 7)
        glimpse_semantics_ratio = (glimpse_semantics.float()) / num_pixels # (128, 16, 7)

        # Initiate LSTM
        optimizer_M4.zero_grad()
        if config_3.data_parallel:
            h_0, c_0 = model_4.module._init_hidden(translated_images.shape[0]) # (1, 128, 7), (1, 128, 11)
        else:
            h_0, c_0 = model_4._init_hidden(translated_images.shape[0]) # (1, 128, 7), (1, 128, 11)
        loss_lstm = torch.tensor([0.0]).cuda()

        # we may need to save the actions too (prepare the dataset again, this time on trainval split)
        for g in range(config_3.num_glimpses):
            # access the glimpse masks and glimpse actions for the corresponding glimpse
            g_semantics = torch.unsqueeze(glimpse_semantics_ratio[:,g,:], 1) # (128, 1, 7)
            g_actions = torch.unsqueeze(glimpse_actions[:,g,:], 1) # (128, 1, 4)
            g_concat = torch.cat((g_semantics, g_actions), 2) # (128, 1, 11)
            if g == 0:
                output_lstm, (h, c) = model_4(g_concat, (h_0, c_0)) # (128, 1, 7), (1, 128, 7), (1, 128, 11)
            else:
                # compute cos_sim between output_lstm and next g_semantics, only for visualization
                # cos_sim = F.cosine_similarity(output_lstm, g_semantics, dim=2).clone().detach()
                # comput loss (MSE or contrastive)
                if not config_4.use_contr_loss:
                    loss_lstm += mse_loss(output_lstm, g_semantics) 
                else:
                    loss_lstm += contrastive_loss(torch.squeeze(output_lstm), torch.squeeze(g_semantics))
                # predict next glimpse semantics
                output_lstm, (h, c) = model_4(g_concat, (h, c))

        # M4 backpropagate
        loss_lstm.backward()
        optimizer_M4.step()
        train_loss_lstm += loss_lstm.item()

        # Print every 100 batches
        if (i+1) % 100 == 0:
            print("Current process: batch: {}/{}, current batch loss: {}".format(i+1, len(train_loader), loss_lstm.item()))

    # Adjust lr every epoch
    if optimizer_M4.param_groups[0]['lr'] > config_4.lr_min:
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
                    },
                    os.path.join(save_path, 'checkpoint_{}.pth.tar'.format(epoch+1)))
            
    torch.save({'epoch': epoch+1, 
                'state_dict': model_4.state_dict(),
                'optimizer' : optimizer_M4.state_dict(),
                },
                os.path.join(save_path, 'checkpoint_latest_{}.pth.tar'.format(epoch+1)))

print('\nFinished Training\n')