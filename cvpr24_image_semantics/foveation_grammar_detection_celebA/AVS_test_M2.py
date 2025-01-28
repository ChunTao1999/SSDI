
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
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

torch.set_printoptions(linewidth = 160)
np.set_printoptions(linewidth = 160)
np.set_printoptions(precision=4)
np.set_printoptions(suppress='True')

from utils_custom_tvision_functions import get_dataloaders, plot_curve
from AVS_config_M2 import AVS_config
from AVS_model_M2 import context_network


SEED            = AVS_config.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
#random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def plotregioncenterspots(list_of_regions, glimpse_size = None, color='g'):
    #print(list_of_regions)
    if glimpse_size is None:
        for region in list_of_regions:
            #print(region)
            xmin = region[0].item()
            ymin = region[1].item()
            width = 10
            height = 10
            x_center = xmin + (width / 2.0)  - 0.5
            y_center = ymin + (height / 2.0) - 0.5
            plt.gca().add_patch(Circle((x_center, y_center), radius=2,edgecolor=color,facecolor=color))
    elif glimpse_size is not None:
        if isinstance(glimpse_size, tuple):
            width, height = glimpse_size
        else:
            width = height = glimpse_size
        for region in list_of_regions:
            xmin = region[0].item()
            ymin = region[1].item()
            x_center = xmin + (width / 2.0)  - 0.5
            y_center = ymin + (height / 2.0) - 0.5
            plt.gca().add_patch(Circle((x_center, y_center), radius=2,edgecolor=color,facecolor=color))


config = AVS_config
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataloaders
train_loader, loss_weights    = get_dataloaders(config, loader_type=config.train_loader_type)
valid_loader                  = get_dataloaders(config, loader_type=config.valid_loader_type)

# Loss(es)
mse_loss     = nn.MSELoss()

# Model
model = context_network(in_channels=config.in_num_channels, res_size = config.low_res, avg_size = config.avg_size).to(device)
print(model)
state_dict = torch.load(config.ckpt_dir)['model']
model.load_state_dict(state_dict)


# Optimizer

if not os.path.exists(config.save_dir): os.makedirs(config.save_dir)
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

        x_min_current   = (bbox_targets[:, 0]/float(config.full_res_img_size[0]))*2.0 - 1.0
        x_max_current   = ((bbox_targets[:, 0]+bbox_targets[:, 2])/float(config.full_res_img_size[0]))*2.0 - 1.0
        y_min_current   = (bbox_targets[:, 1]/float(config.full_res_img_size[0]))*2.0 - 1.0
        y_max_current   = ((bbox_targets[:, 1]+bbox_targets[:, 3])/float(config.full_res_img_size[0]))*2.0 - 1.0

        pred_locations      = model(images)
        #print(x_min_current, pred_locations)
        total              += targets.size(0)
        for k in range(0, targets.size(0)):
            #print(pred_locations[k,:], true_locations[k,:])
            if pred_locations[k,0]<=x_max_current[k] and pred_locations[k,0]>=x_min_current[k] and pred_locations[k,1]<=y_max_current[k] and pred_locations[k,1]>=y_min_current[k]:
                correct+=1
    
        loss               = mse_loss(pred_locations, true_locations)
        test_loss         += loss.item()
        #break
        
print("Validation Loss: {:.3f}, Acc: {:.4f} [{}/{}]\n".format((test_loss/(i+1)), (100.*correct/total), correct, total))

for j in range(0, images.size(0)):
    image   = images[j, :]
    pred_pixs = ((pred_locations[j,:]+1.0)/2.0)*float(config.full_res_img_size[0])
    print(pred_pixs, init_locs[j,:], image.size())
    input_img = torch.permute(image, (1,2,0)).cpu().numpy()
    plt.figure()
    plt.imshow(input_img)
    plotregioncenterspots([pred_pixs], color='r')
    plotregioncenterspots([init_locs[j,:]], color='g')
    plt.savefig(config.save_dir + 'cn'+str(j)+'i.png')
    