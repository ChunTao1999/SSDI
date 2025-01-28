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
from AVS_config_M1_celeba import AVS_config
from AVS_model_M1_vgg import customizable_VGG as VGG
import torchvision.models.vgg as pretrained_vgg

# Debug
import pdb


#%% Instantiate parameters, dataloaders, and model
# Parameters
config = AVS_config
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# SEED instantiation
SEED            = config.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
#random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Dataloaders
train_loader, loss_weights    = get_dataloaders(config, loader_type=config.train_loader_type)
valid_loader                  = get_dataloaders(config, loader_type=config.valid_loader_type)
# pdb.set_trace()
# Loss(es)
bce_loss                      = nn.BCEWithLogitsLoss(pos_weight=loss_weights.to(device))  

# Model
model = VGG(config).to(device) # customizable_VGG
print(model)

if config.pretrained:
    if config.vgg_name == 'vgg11':
        model_pretrained = pretrained_vgg.vgg11(pretrained=True)
    elif config.vgg_name == 'vgg11_bn':
        model_pretrained = pretrained_vgg.vgg11_bn(pretrained=True)
    else:
        raise NotImplementedError("Requested pretrained model ({}), however, the script does not support it yet!".format(config.vgg_name))

    model_pretrained_sd = model_pretrained.state_dict()
    model_sd = model.state_dict()
    for (k, v), (k_pt, v_pt) in zip(model_sd.items(), model_pretrained_sd.items()):
        if 'classifier' not in k:
            model_sd[k] = model_pretrained_sd[k_pt].clone().detach()
            print("Copied ({}) from pretrained model's ({})".format(k, k_pt))
    model.load_state_dict(model_sd)
    for n, p in model.named_parameters():
        if 'classifier' not in n:
            p.requires_grad_(False)


# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_start, weight_decay=config.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma = 0.1, milestones=config.milestones)

# Logger dictionary
log_dict     = {'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[]}
if not os.path.exists(config.save_dir): os.makedirs(config.save_dir)

#%% Train the model
for epoch in range(0, config.epochs):
    print("Epoch: {}/{}\n".format(epoch+1, config.epochs))
    model.train()
    train_loss  = 0.0
    correct     = 0
    total       = 0
    for i, (images, targets) in enumerate(train_loader):
        translated_images, targets, bbox_targets            = images.to(device), targets[0].float().to(device), targets[1].to(device)
        # pdb.set_trace()
        optimizer.zero_grad()
        outputs             = model(translated_images)
        loss                = bce_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        
        pred_labels         = (outputs >= config.attr_detection_th).float()
        total              += targets.numel()
        correct            += (pred_labels == targets).sum().item()
        train_loss         += loss.item()
    
    print("Train Loss: {:.3f}, Acc: {:.4f} [{}/{}]\n".format((train_loss/(i+1)), (100.*correct/total), correct, total))
    log_dict['train_loss'].append(train_loss/(i+1))
    log_dict['train_acc'].append(100.*correct/total)
    
    model.eval()
    test_loss   = 0.0
    correct     = 0
    total       = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(valid_loader):
            translated_images, targets, bbox_targets            = images.to(device), targets[0].float().to(device), targets[1].to(device)

            outputs         = model(translated_images)
            loss            = bce_loss(outputs, targets)
            pred_labels     = (outputs >= config.attr_detection_th).float()
            total          += targets.numel()
            correct        += (pred_labels == targets).sum().item()
            test_loss      += loss.item()
            
    print("Validation Loss: {:.3f}, Acc: {:.4f} [{}/{}]\n".format((test_loss/(i+1)), (100.*correct/total), correct, total))
    log_dict['test_loss'].append(test_loss/(i+1))
    log_dict['test_acc'].append(100.*correct/total)
        
    if optimizer.param_groups[0]['lr'] > config.lr_min:
        lr_scheduler.step()

    # Storing results
    ckpt = {}
    ckpt['model']   = model.state_dict()
    ckpt['log']     = log_dict
    torch.save(ckpt, config.ckpt_dir)

# Plotting statistics
plot_dir = config.save_dir
if not os.path.exists(plot_dir): os.makedirs(plot_dir)
epochs = range(1, config.epochs+1)
plot_curve(epochs, log_dict['train_loss'], 'training loss', 'epoch', 'train loss',     plot_dir + 'train_loss.png')
plot_curve(epochs, log_dict['train_acc'],  'Performance',   'epoch', 'train accuracy', plot_dir + 'train_acc.png')
plot_curve(epochs, log_dict['test_acc'],   'Performance',   'epoch', 'test accuracy',  plot_dir + 'test_acc.png')
plot_curve(epochs, log_dict['test_loss'],  'testing loss',  'epoch', 'test loss',      plot_dir + 'test_loss.png')