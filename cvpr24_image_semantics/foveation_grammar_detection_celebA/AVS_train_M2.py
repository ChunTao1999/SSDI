
"""
Verified on May 25 2022
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
from AVS_config_M2 import AVS_config
from AVS_model_M2 import context_network



config = AVS_config
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# Loss(es)
mse_loss     = nn.MSELoss()

# Model
model = context_network(in_channels=config.in_num_channels, res_size = config.low_res, avg_size = config.avg_size).to(device)
print(model)


# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_start)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma = 0.1, milestones=config.milestones)

# Logger dictionary
log_dict     = {'train_loss':[],
                'train_acc': [],
                'test_loss':[],
                'test_acc': []}
if not os.path.exists(config.save_dir): os.makedirs(config.save_dir)

for epoch in range(0, config.epochs):
    print("Epoch: {}/{}\n".format(epoch+1, config.epochs))
    model.train()
    train_loss  = 0.0
    correct     = 0
    total       = 0
    for i, (images, targets) in enumerate(train_loader):
        images, targets, bbox_targets, init_locs = images.to(device), targets[0].float().to(device), targets[1].to(device), targets[2].type(torch.FloatTensor).to(device)
        
        true_locations       = torch.zeros_like(init_locs)
        true_locations[:,0]  = (init_locs[:,0]/float(config.full_res_img_size[0]))*2.0 - 1.0
        true_locations[:,1]  = (init_locs[:,1]/float(config.full_res_img_size[1]))*2.0 - 1.0
        optimizer.zero_grad()
        pred_locations       = model(images)
        total += targets.size(0)
        x_min_current   = (bbox_targets[:, 0]/float(config.full_res_img_size[0]))*2.0 - 1.0
        x_max_current   = ((bbox_targets[:, 0]+bbox_targets[:, 2])/float(config.full_res_img_size[0]))*2.0 - 1.0
        y_min_current   = (bbox_targets[:, 1]/float(config.full_res_img_size[0]))*2.0 - 1.0
        y_max_current   = ((bbox_targets[:, 1]+bbox_targets[:, 3])/float(config.full_res_img_size[0]))*2.0 - 1.0
        for k in range(0, targets.size(0)):
            #print(pred_locations[k,:], true_locations[k,:])
            if pred_locations[k,0]<=x_max_current[k] and pred_locations[k,0]>=x_min_current[k] and pred_locations[k,1]<=y_max_current[k] and pred_locations[k,1]>=y_min_current[k]:
                correct+=1
        
        loss                = mse_loss(pred_locations, true_locations)
        loss.backward()
        optimizer.step()
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
            images, targets, bbox_targets, init_locs     = images.to(device), targets[0].float().to(device), targets[1].to(device), targets[2].type(torch.FloatTensor).to(device)
            true_locations      = torch.zeros_like(init_locs)
            true_locations[:,0] = (init_locs[:,0]/float(config.full_res_img_size[0]))*2.0 - 1.0
            true_locations[:,1] = (init_locs[:,1]/float(config.full_res_img_size[1]))*2.0 - 1.0

            pred_locations      = model(images)
            total              += targets.size(0)
            x_min_current   = (bbox_targets[:, 0]/float(config.full_res_img_size[0]))*2.0 - 1.0
            x_max_current   = ((bbox_targets[:, 0]+bbox_targets[:, 2])/float(config.full_res_img_size[0]))*2.0 - 1.0
            y_min_current   = (bbox_targets[:, 1]/float(config.full_res_img_size[0]))*2.0 - 1.0
            y_max_current   = ((bbox_targets[:, 1]+bbox_targets[:, 3])/float(config.full_res_img_size[0]))*2.0 - 1.0
            for k in range(0, targets.size(0)):
                #print(pred_locations[k,:], true_locations[k,:])
                if pred_locations[k,0]<=x_max_current[k] and pred_locations[k,0]>=x_min_current[k] and pred_locations[k,1]<=y_max_current[k] and pred_locations[k,1]>=y_min_current[k]:
                    correct+=1
        
            loss               = mse_loss(pred_locations, true_locations)
            test_loss         += loss.item()
            
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