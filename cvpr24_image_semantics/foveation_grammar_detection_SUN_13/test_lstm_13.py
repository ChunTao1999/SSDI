#%% Imports
from scipy.io import loadmat
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torchvision
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
from operator import itemgetter
from dataloader_SUNrgbd_13 import SUN_RGBD
import os
import shutil
import random
from utils import ShufflePatches
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import pyplot as plt
from utils import unfold
from config_lstm import LSTM_config
from model_lstm import customizable_LSTM as LSTM
# Debug imports
from torchinfo import summary
import pdb


#%% SEED instantiation
SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#%% Define color map for plotting segmentation mask
num_seg_classes = 14
viridis = mpl.colormaps['viridis'].resampled(num_seg_classes) # 14 colors in the sequential colormap
newcolors = np.empty_like(viridis(np.linspace(0, 1, 14)))
newcolors[0, :] = np.array([0, 0, 0, 1]) # background
newcolors[1, :] = np.array([0, 0, 1, 1]) # bed
newcolors[2, :] = np.array([0.9137, 0.3490, 0.1882, 1]) # books
newcolors[3, :] = np.array([0, 0.8549, 0, 1]) # ceiling
newcolors[4, :] = np.array([0.5843, 0, 0.9412, 1]) # chair
newcolors[5, :] = np.array([0.8706, 0.9451, 0.0941, 1]) # floor
newcolors[6, :] = np.array([1.0000, 0.8078, 0.8078, 1]) # furniture
newcolors[7, :] = np.array([0, 0.8784, 0.8980, 1]) # objects
newcolors[8, :] = np.array([0.4157, 0.5333, 0.8000, 1]) # pictures
newcolors[9, :] = np.array([0.4588, 0.1137, 0.1608, 1]) # sofa
newcolors[10, :] = np.array([0.9412, 0.1373, 0.9216, 1]) # table
newcolors[11, :] = np.array([0, 0.6549, 0.6118, 1]) # tv
newcolors[12, :] = np.array([0.9765, 0.5451, 0, 1]) # wall
newcolors[13, :] = np.array([0.8824, 0.8980, 0.7608, 1]) # window
newcmp = ListedColormap(newcolors)


#%% Dataset and Dataloader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
root_path = "/home/nano01/a/tao88/SUN-RGBD_13_classes"
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])

add_corruption = True
patch_size = 64 # try 64
num_distortion = 6
corruption_type = "patch_shuffle" # patch_shuffle or patch_permute

test_data = SUN_RGBD(root=root_path,
                     seed=SEED,
                     split="test",
                     target_type=["class", "seg_map", "corrupt_label"],
                     transform=transform,
                     add_corruption=add_corruption,
                     corruption_type=corruption_type,
                     num_distortion=num_distortion,
                     patch_size=patch_size)
test_loader = torch.utils.data.DataLoader(test_data, 
                                          batch_size=LSTM_config.batch_size_eval, 
                                          shuffle=False)


#%% LSTM model
model_lstm = LSTM(LSTM_config)
model_lstm.cuda()
# if load from pretrained model
if LSTM_config.from_pretrained:
    ckpt_lstm = torch.load(LSTM_config.ckpt_dir_model_M4)
    model_lstm.load_state_dict(ckpt_lstm['state_dict'])
    for p in model_lstm.parameters():
        p.requires_grad_(False)
    print("Model M4:\n", "Loaded from: {}".format(LSTM_config.ckpt_dir_model_M4))
    model_lstm.eval()

print(model_lstm)


#%% LSTM Optimizer
# optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=LSTM_config.lr_start, weight_decay=LSTM_config.weight_decay)
# lr_scheduler_lstm = torch.optim.lr_scheduler.MultiStepLR(optimizer_lstm, gamma=LSTM_config.gamma, milestones=LSTM_config.milestones, verbose=True)


#%% LSTM losses
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


#%% Patch semantics detector model


#%% Testing log_dict and debug path (for visualization)
log_dict = {"test_loss_lstm": [],
            "test_threshold": [],
            "test_acc_lstm": []}
debug_path = "/home/nano01/a/tao88/4.13_SUN_debug_{}_{}_{}".format(corruption_type, patch_size, num_distortion)
if not os.path.exists(debug_path): os.makedirs(debug_path)


#%% Testing
with torch.no_grad():
    test_loss_lstm = 0.0
    dist_correct_imgs, dist_corrupted_imgs = [], []
    model_lstm.eval()
    for i, (index, images, targets) in enumerate(test_loader):
        translated_images = images.to(device) # (batch_size, num_channels, H, W)
        masks, corrupt_lbls = targets[0].to(device), targets[1].to(device)
        # targets[0]: seg maps, range [0, 13] floattensor
        # targets[1]: corrupt labels, 0 or 1, longtensor

        # visualize images in 1st batch and their gt seg maps
        if i == 0:
            for img_id in range(5):
                save_image(translated_images[img_id], os.path.join(debug_path, 'img_{}.jpg'.format(img_id)))
                # gt image masks
                # fig = plt.figure(figsize=(1,1))
                fig = plt.figure()
                ax = plt.Axes(fig, [0., 0., 1., 1.]) # no borders
                ax.set_axis_off()
                fig.add_axes(ax)
                psm = ax.imshow(X=masks[img_id].squeeze(0).cpu().numpy(),
                                interpolation='nearest',
                                cmap=viridis,
                                vmin=0,
                                vmax=14)
                fig.colorbar(psm, ax=ax)
                plt.savefig(os.path.join(debug_path, 'img_{}_mask.png'.format(img_id)))
                plt.close(fig)
        
        # crop each image and mask into patches, and then start training LSTM for all 38 classes
        patches = unfold(X=translated_images,
                         patch_size=LSTM_config.patch_size) # (bs, num_patch, num_ch, ps, ps)
        patch_masks = unfold(X=masks,
                             patch_size=LSTM_config.patch_size) # (bs, num_patch, num_ch, ps, ps)

        # visualize patches and patch masks
        if i == 0:
            for img_id in range(5):
                for patch_id in range(4):
                    save_image(patches[img_id][patch_id], os.path.join(debug_path, 'img_{}_patch_{}.jpg'.format(img_id, patch_id)))
                    # gt patch masks
                    # fig = plt.figure(figsize=(1,1))
                    fig = plt.figure()
                    ax = plt.Axes(fig, [0., 0., 1., 1.]) # no borders
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    psm = ax.imshow(X=patch_masks[img_id][patch_id].squeeze(0).cpu().numpy(),
                                    interpolation='nearest',
                                    cmap=viridis,
                                    vmin=0,
                                    vmax=14)
                    fig.colorbar(psm, ax=ax)
                    plt.savefig(os.path.join(debug_path, 'img_{}_patch_{}_mask.png'.format(img_id, patch_id)))
                    plt.close(fig)

        # transform from cropped masks to their semantics
        num_pixels = patch_masks.shape[3] * patch_masks.shape[4] # 128*128=16384
        patch_masks = patch_masks.flatten(start_dim=2, end_dim=4) # (128, 4, 16384)
        mask_semantics = torch.nn.functional.one_hot(patch_masks.long()).sum(dim=2) # (128, 4, 14)
        mask_semantics_ratio = (mask_semantics.float()) / num_pixels # (128, 4, 14)

        # feed the sequence of cancatted mask and semantics to LSTM
        # initialize hidden and cell states
        h0, c0 = model_lstm._init_hidden(translated_images.shape[0])
        # (1, 128, 128) (D*num_layers, N=bs, H_out), (1, 128, 128) (D*num_layers, N, H_cell)
        loss_lstm = torch.tensor([0.0]).cuda()

        # Compute MSE loss between output_lstm and next semantics embeddings (128-d)
        output_lstm, (hn, cn) = model_lstm([patch_masks, mask_semantics_ratio], (h0, c0))
        # Forward loss
        loss_lstm = mse_loss(output_lstm[:, :-1, :LSTM_config.semantics_dim], mask_semantics_ratio[:, 1:, :]) #  both (128, 4, 7)
        # Backward loss
        loss_lstm += mse_loss(output_lstm[:, 1:, LSTM_config.semantics_dim:], mask_semantics_ratio[:, :-1, :]) # both (128, 4, 7)
        test_loss_lstm += loss_lstm.item()
        # Print every 10 batches
        if (i+1) % 10 == 0:
            print("Current process: batch: {}/{}, current batch loss: {}".format(i+1, len(test_loader), loss_lstm.item()))

        # record dist curves for each image in dist_batch_imgs
        dist_batch_imgs = torch.norm((output_lstm[:, :-1, :LSTM_config.semantics_dim] - mask_semantics_ratio[:, 1:, :]), dim=2, keepdim=False) # (128, 4)
        dist_batch_imgs += torch.norm((output_lstm[:, 1:, LSTM_config.semantics_dim:] - mask_semantics_ratio[:, :-1, :]), dim=2, keepdim=False) # (128, 4)
        # concat to the dist array for all test images
        if i == 0:
            dist_correct_imgs = dist_batch_imgs[corrupt_lbls==0]
            dist_corrupted_imgs = dist_batch_imgs[corrupt_lbls==1]
        else:
            dist_correct_imgs = torch.cat((dist_correct_imgs, dist_batch_imgs[corrupt_lbls==0]), dim=0)
            dist_corrupted_imgs = torch.cat((dist_corrupted_imgs, dist_batch_imgs[corrupt_lbls==1]), dim=0)
    
    # Print and tally average epoch loss
    print("Test Loss (LSTM) over epoch: {:.5f}\n".format(test_loss_lstm / (i+1)))
    log_dict['test_loss_lstm'].append(test_loss_lstm / (i+1))

    # Use torch.hist for picking threshold and computing test accuracy
    dist_correct_imgs, dist_corrupted_imgs = np.array(dist_correct_imgs.cpu()), np.array(dist_corrupted_imgs.cpu()) # both (#, 4)
    dist_correct_mean, dist_corrupted_mean = dist_correct_imgs.mean(1), dist_corrupted_imgs.mean(1)
    hist_correct, bin_edges_correct = np.histogram(dist_correct_mean, bins=np.arange(0, 2.55, 0.05))
    hist_corrupted, bin_edges_corrupted = np.histogram(dist_corrupted_mean, bins=np.arange(0, 2.55, 0.05))

    cutoff_index = np.where(hist_corrupted>hist_correct)[0][0]
    threshold = bin_edges_correct[cutoff_index] # first occurence
    print("Number of test samples taken into account in hist: correct: {}, corrupted: {}".format(hist_correct.sum(), hist_corrupted.sum()))
    num_fn, num_fp = hist_correct[cutoff_index:].sum(), hist_corrupted[:cutoff_index].sum()
    num_tn, num_tp = hist_corrupted[cutoff_index:].sum(), hist_correct[:cutoff_index].sum()
    test_acc = (num_tn + num_tp) / (num_fn + num_fp + num_tn + num_tp) * 100
    print("To distinguish correct from corrupted samples, the threshold of mean distance between patch semantics and its prediction is %.2f" % (threshold))
    print("By using the previous threshold, the test accuracy is %.3f%%" % (test_acc))

    # Use plt for visualization
    plt.figure()
    # plt.hist(dist_correct_imgs.mean(1), range=(0, 0.8), bins=16, alpha=0.5, color='b', label='correct')
    # plt.hist(dist_corrupted_imgs.mean(1), range=(0, 0.8), bins=16, alpha=0.5, color='r', label='corrupted')
    plt.hist(dist_correct_mean, bins=np.arange(0, 1.55, 0.05), alpha=0.7, color='b', label='correct')
    plt.hist(dist_corrupted_mean, bins=np.arange(0, 1.55, 0.05), alpha=0.7, color='r', label='corrupted')
    plt.xticks(np.arange(0, 1.6, 0.1))
    plt.yticks(np.arange(0, 1100, 100))
    plt.title('LSTM Prediction Results')
    plt.xlabel('Average dist. over 5 patches')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(debug_path, 'hist_distavg_on_test.png'))
    plt.close()
    print("Histogram on testset plotted and saved")

print('\nFinished Testing!\n')


#%% backup code
# \usepackage[table]{xcolor}
# \definecolor{bedColor}{rgb}{0, 0, 1}
# \definecolor{booksColor}{rgb}{0.9137,0.3490,0.1882}
# \definecolor{ceilColor}{rgb}{0, 0.8549, 0}
# \definecolor{chairColor}{rgb}{0.5843,0,0.9412}
# \definecolor{floorColor}{rgb}{0.8706,0.9451,0.0941}
# \definecolor{furnColor}{rgb}{1.0000,0.8078,0.8078}
# \definecolor{objsColor}{rgb}{0,0.8784,0.8980}
# \definecolor{paintColor}{rgb}{0.4157,0.5333,0.8000}
# \definecolor{sofaColor}{rgb}{0.4588,0.1137,0.1608}
# \definecolor{tableColor}{rgb}{0.9412,0.1373,0.9216}
# \definecolor{tvColor}{rgb}{0,0.6549,0.6118}
# \definecolor{wallColor}{rgb}{0.9765,0.5451,0}
# \definecolor{windColor}{rgb}{0.8824,0.8980,0.7608}