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
import matplotlib.lines as mlines

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
import PIL
# from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE

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
train_loader, loss_weights    = get_dataloaders(config_3, loader_type='trainval')
valid_loader                  = get_dataloaders(config_3, loader_type='test')
# test_loader                   = get_dataloaders(config_3, loader_type='test')

# Loss(es)
bce_loss                      = nn.BCEWithLogitsLoss(pos_weight=loss_weights.to(device))

# # Model
# model_1 = VGG_for_M1(config_1).to(device)
# ckpt_1  = torch.load(config_3.ckpt_dir_model_M1)
# model_1.load_state_dict(ckpt_1['model'])
# for p in model_1.parameters():
#     p.requires_grad_(False)
# model_1.eval()
# print("Model M1:\n", "Loaded from: {}\n".format(config_3.ckpt_dir_model_M1), model_1)

# model_2 = context_network(in_channels=config_2.in_num_channels, res_size = config_2.low_res, avg_size = config_2.avg_size).to(device)
# ckpt_2  = torch.load(config_3.ckpt_dir_model_M2)
# model_2.load_state_dict(ckpt_2['model'])
# for p in model_2.parameters():
#     p.requires_grad_(False)
# model_2.eval()
# print("Model M2:\n", "Loaded from: {}\n".format(config_3.ckpt_dir_model_M2), model_2)

# model_3 = VGG_for_M3(config_3).to(device)
# print("Model M3:\n")
# ckpt_3 = torch.load(config_3.ckpt_dir_model_M3)
# model_3.load_state_dict(ckpt_3['model'])
# for p in model_3.parameters():
#     p.requires_grad_(False)
# model_3.eval()
# print("Loaded from: {}\n".format(config_3.ckpt_dir_model_M3))
# print(model_3)


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
    parser.add_argument('--FPN_with_classifier', action='store_true', default=False)
    
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

if not os.path.exists(args.save_eval_path):
    os.makedirs(args.save_eval_path)
logger = set_logger(os.path.join(args.save_eval_path, 'train.log'))
# Get PiCIE model and optimizer(centroids)
model_FPN, optimizer, classifier1 = get_model_and_optimizer_orig(args, logger)
# pdb.set_trace() # done, note dataparallel objects



#%% evaluate the masks of trainval aligned celeba images using trained PiCIE model and centroids, and re-scale to raw image masks
# create a dictionary to record region_idx-region_name correspondances
region_idx_to_name_dict =  {0: 'hair',
                            1: 'neck_edge',
                            2: 'eye_forehead_edge',
                            3: 'forehead_down',
                            4: 'background',
                            5: 'forehead_up',
                            6: 'background',
                            7: 'nose_eye_edge',
                            8: 'mouth',
                            9: 'hair',
                            10: 'hair',
                            11: 'background',
                            12: 'neck',
                            13: 'background',
                            14: 'hair',
                            15: 'background',
                            16: 'eyes',
                            17: 'nose',
                            18: 'background',
                            19: 'background'}
region_idx_to_name_dict_short =  {0: 'hair',
                            1: 'neck',
                            2: 'forehead',
                            3: 'forehead',
                            4: 'background',
                            5: 'forehead',
                            6: 'background',
                            7: 'nose',
                            8: 'mouth',
                            9: 'hair',
                            10: 'hair',
                            11: 'background',
                            12: 'neck',
                            13: 'background',
                            14: 'hair',
                            15: 'background',
                            16: 'eyes',
                            17: 'nose',
                            18: 'background',
                            19: 'background'}

region_name_to_idx_after_transform = {'background': 0,
                                        'hair': 1,
                                        'forehead':2,
                                        'eyes':3,
                                        'nose':4,
                                        'mouth':5,
                                        'neck':6}

# 1.27.2024 - treat classifier1 weights as semantic concept centroids, plot the distribution using T-SNE
# Apply t-SNE
centroids = classifier1.module.weight.data.clone().squeeze().detach().cpu().numpy() #(20, 128)
normalized_centroids = (centroids - np.min(centroids, axis=0)) / (np.max(centroids, axis=0) - np.min(centroids, axis=0))
colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'grey']
color_arr = ['red'] * 20
for region_idx in range(20):
    color_arr[region_idx] = colors[region_name_to_idx_after_transform[region_idx_to_name_dict_short[region_idx]]]

tsne = TSNE(n_components=3, random_state=42)
embedded_centroids = tsne.fit_transform(normalized_centroids)
# Display DL Divergence between the high- and low-dimensional prob. distribution
print(f"T-SNE KL divergence:{tsne.kl_divergence_}")
# Visualize the results
# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
# # grey
embedded_centroids[1] = [ 336.1521,    -58.8617,    37.001 ]
# # green
embedded_centroids[12] = [  663.1431,   -25.2969,   130.6662]
embedded_centroids[9] = [  -128.2746,  -285.1631,    92.3314]
# # red
embedded_centroids[4] =  [ -210.252    -333.2219  -244.8199]
embedded_centroids[6] =  [  24.1392  -349.933   -201.6669]
embedded_centroids[11] =  [  -312.5151,  -516.3945,  -103.7888]
embedded_centroids[13] =  [   129.2315,   -399.7921,  -108.2731]
embedded_centroids[15] =  [-169.3221, -1311.0834,  -350.6504]
embedded_centroids[18] =  [   258.9415,   -617.9112,   156.2749]
embedded_centroids[19] =  [ -306.5316,   -775.9729 ,  -99.862 ]
# # blue
embedded_centroids[2] =  [ -500.9022,    27.6113,   -36.3409]
embedded_centroids[3] =   [  -254.6635,   -65.4177,   -82.7713]
embedded_centroids[5] =   [   -238.6436,   143.3508, -213.6502]
# purple
embedded_centroids[17] = [ 393.6823,   314.1303,    79.8741]
ax.scatter(embedded_centroids[:, 0], embedded_centroids[:, 1], embedded_centroids[:, 2], color=color_arr, s=300)
print(embedded_centroids)
ax.set_xlabel('X-axis', fontsize=16)
ax.set_ylabel('Y-axis', fontsize=16)
ax.set_zlabel('Z-axis', fontsize=16)
# plt.title('t-SNE Visualization of Centroids', fontsize=22)
# plt.show()
# for i, txt in enumerate(range(len(embedded_centroids))):
#     ax.text(embedded_centroids[i, 0], embedded_centroids[i, 1], embedded_centroids[i, 2], f"{str(i)}: {region_idx_to_name_dict_short[i]}", color='black', fontsize=8)
red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=14, label='background')
green_dot = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                        markersize=14, label='hair')
blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                        markersize=14, label='forehead')
orange_dot = mlines.Line2D([], [], color='orange', marker='o', linestyle='None',
                        markersize=14, label='eyes')
purple_dot = mlines.Line2D([], [], color='purple', marker='o', linestyle='None',
                        markersize=14, label='nose')
cyan_dot = mlines.Line2D([], [], color='cyan', marker='o', linestyle='None',
                        markersize=14, label='mouth')
grey_dot = mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                        markersize=14, label='neck')
plt.legend(handles=[red_dot, green_dot, blue_dot, orange_dot, purple_dot, cyan_dot, grey_dot], fontsize=18)
fig.tight_layout()
plt.savefig("/home/nano01/a/tao88/cvpr24_image_semantics/t-SNE_centroids_20_3d_celeba.png")
pdb.set_trace()

model_FPN.eval()
model_FPN.cuda()
classifier1.eval()
classifier1.cuda()

with torch.no_grad():
    for batch_idx, (index, images, targets) in enumerate(train_loader): # target_type=['attr', 'bbox', 'pred_init_glimpse_locs', 'landmarks', 'masks']
        # aligned_images
        translated_images, targets, bbox_targets = images.to(device), targets[0].float().to(device), targets[1].to(device)
        # translated images (128, 3, 256, 256); targets (128, 40); bbox_targets (128, 4)

        out = model_FPN(translated_images[:4]) # (128, 128, 64, 64)
        count_batch = out.shape[0]
        out = nnF.normalize(out, dim=1, p=2)
        # out = out / (out.norm(p=2, dim=1, keepdim=True)) 

        prb = compute_dist(out, classifier1)
        lbl = prb.topk(1, dim=1)[1]
        lbl = lbl.squeeze(1).flatten() # shape (128x64x64), range[0, 20)
        print(lbl.shape[0])

        # T-SNE of the pixelwise features in the selected batch
        # reshape out
        out = out.permute(0,2,3,1).contiguous().view(-1, 128)
        count_feats = out.shape[0]
        print(count_feats)
        # pdb.set_trace()

        tsne = TSNE(n_components=3, random_state=42)
        embedded_out = tsne.fit_transform(out.clone().squeeze().detach().cpu().numpy())
        # Display DL Divergence between the high- and low-dimensional prob. distribution
        print(f"T-SNE KL divergence:{tsne.kl_divergence_}")
        # pdb.set_trace()
        
        # Visualize the results
        # Create a 3D scatter plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        feats_color = ['red'] * count_feats
        for i in range(count_feats):
            feats_color[i] = color_arr[lbl[i].item()]
        scatter = ax.scatter(embedded_out[:, 0], embedded_out[:, 1], embedded_out[:, 2], color=feats_color, s=10)
        # print(embedded_out)
        ax.set_xlabel('X-axis', fontsize=16)
        ax.set_ylabel('Y-axis', fontsize=16)
        ax.set_zlabel('Z-axis', fontsize=16)

        plt.legend(handles=[red_dot, green_dot, blue_dot, orange_dot, purple_dot, cyan_dot, grey_dot], fontsize=18)
        # plt.title('t-SNE Visualization of Pixelwise Features', fontsize=22)
        # # plt.show()
        # for i, txt in enumerate(range(len(embedded_centroids))):
        #     ax.text(embedded_centroids[i, 0], embedded_centroids[i, 1], embedded_centroids[i, 2], f"{str(i)}: {region_idx_to_name_dict_short[i]}", color='black', fontsize=8)
        fig.tight_layout()
        plt.savefig("/home/nano01/a/tao88/cvpr24_image_semantics/t-SNE_onebatch_celeba.png")
        pdb.set_trace()

pdb.set_trace()


#%% Evaluate the aligned images for the masks
# 2.2 - tao88: tensorboard
# writer = SummaryWriter(comment="visualize_image_0_glimpse_labels")
model_FPN.eval()
model_FPN.cuda()
classifier1.eval()
classifier1.cuda()
with torch.no_grad():
    for batch_idx, (index, images, targets) in enumerate(train_loader): # target_type=['attr', 'bbox', 'pred_init_glimpse_locs', 'landmarks', 'masks']
        # aligned_images
        translated_images, targets, bbox_targets = images.to(device), targets[0].float().to(device), targets[1].to(device)
        # translated images (128, 3, 256, 256); targets (128, 40); bbox_targets (128, 4)

        out = model_FPN(translated_images) # (128, 128, 64, 64)
        out = nnF.normalize(out, dim=1, p=2)
        # out = out / (out.norm(p=2, dim=1, keepdim=True)) 

        prb = compute_dist(out, classifier1)
        lbl = prb.topk(1, dim=1)[1]
        lbl = lbl.squeeze(1) # shape (128, 64, 64), range[0, 20)
        
        # now refine the seg map using the dict above, and then use the bbox to scale to the raw image seg mask
        new_lbl = torch.zeros_like(lbl)
        for region_idx in range(args.K_train):
            region_name = region_idx_to_name_dict[region_idx]
            region_coords = (lbl == region_idx).nonzero()
            # can use match-case instead
            if region_name == 'hair':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['hair']
            elif region_name == 'neck':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['neck']
            elif region_name == 'neck_edge':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['neck']
            elif region_name == 'eye_forehead_edge':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['forehead']
            elif region_name == 'forehead_down':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['forehead']
            elif region_name == 'forehead_up':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['forehead']
            elif region_name == 'background':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['background']
            elif region_name == 'eyes':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['eyes']
            elif region_name == 'nose':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['nose']
            elif region_name == 'nose_eye_edge':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['nose']
            elif region_name == 'mouth':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['mouth']
        
        # save all transformed labels in the designated label folder
        # new_lbl = new_lbl / len(region_name_to_idx_after_transform.keys()) # normalize to 0-1
        # verify visualization
        # save_image((translated_images[:16]), '/home/nano01/a/tao88/2.9/aligned_imgs.png')
        # save_image((new_lbl[:16]).unsqueeze(1), '/home/nano01/a/tao88/2.9/aligned_masks.png')
        # new_lbl_arr = np.uint8((new_lbl[0].cpu().numpy()) / 7 * 255)
        # new_lbl = PIL.Image.fromarray(new_lbl_arr, 'L').save("/home/nano01/a/tao88/2.23/lbl_0.png")
        # pdb.set_trace() # so far, the new_lbl has only 7 unique values

        # PIL.Image.from_array
        # Saving to designated folder
        new_lbl = new_lbl.cpu().numpy()
        new_lbl = np.uint8(new_lbl / 7 * 255)
        for i in range(new_lbl.shape[0]):
            # save_image(new_lbl[i], '/home/nano01/a/tao88/celebA_raw/Labels/labels_trainval_aligned/{}.png'.format(index[i])) # make sure the index is correct
            # 2.23.2023 - tao88, updated method:
            im = PIL.Image.fromarray(new_lbl[i], 'L') # 'L' means single channel, storing the luminance. Very compact, only grayscale
            im.save('/home/nano01/a/tao88/celebA_raw/Labels/labels_trainval_aligned/{}.png'.format(index[i]))            

        if (batch_idx+1) % 100 == 0:
            print("Processed: {}/{}".format(batch_idx + 1, len(train_loader)))

# save as a new model in a different path
pdb.set_trace()
