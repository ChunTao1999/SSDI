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
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

train_data = SUN_RGBD(root=root_path,
                      seed=SEED,
                      split="train",
                      target_type=["class", "seg_map"],
                      transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, 
                                           batch_size=LSTM_config.batch_size_train, 
                                           shuffle=False)


#%% LSTM model
model_lstm = LSTM(LSTM_config)
model_lstm.cuda()
model_lstm.train()
print("Model M4:\n", "Train from scratch!")

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
optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=LSTM_config.lr_start, weight_decay=LSTM_config.weight_decay)

lr_scheduler_lstm = torch.optim.lr_scheduler.MultiStepLR(optimizer_lstm, gamma=LSTM_config.gamma, milestones=LSTM_config.milestones, verbose=True)


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



#%% Training log_dict, save_path, and debug path (for visualization)
log_dict = {"train_loss_lstm": [],
            "val_loss_lstm": [],
            "val_threshold": [],
            "val_acc_lstm": []}
debug_path = "/home/nano01/a/tao88/4.13_SUN_train_debug"
if not os.path.exists(debug_path): os.makedirs(debug_path)
save_path = './ps={}_bi-lstm_numlayers={}_startlr={}_epoch={}'.format(LSTM_config.patch_size,
                                                                      LSTM_config.num_layers,
                                                                      LSTM_config.lr_start, 
                                                                      LSTM_config.epochs)
if not os.path.exists(save_path): os.makedirs(save_path)


#%% Training
for epoch in range(0, LSTM_config.epochs):
    print("\nEpoch: {}/{}".format(epoch+1, LSTM_config.epochs))
    train_loss_lstm = 0.0
    model_lstm.train()
    for i, (index, images, targets) in enumerate(train_loader):
        translated_images = images.to(device) # (batch_size, num_channels, H, W)
        masks = targets[0].to(device)
        # targets[0]: seg maps, range [0, 13] floattensor

        # visualize images in 1st batch and their gt seg maps
        # if i == 0:
        #     for img_id in range(16):
        #         save_image(translated_images[img_id], os.path.join(debug_path, 'img_{}.jpg'.format(img_id)))
        #         # gt image masks
        #         # fig = plt.figure(figsize=(1,1))
        #         fig = plt.figure()
        #         ax = plt.Axes(fig, [0., 0., 1., 1.]) # no borders
        #         ax.set_axis_off()
        #         fig.add_axes(ax)
        #         psm = ax.imshow(X=masks[img_id].squeeze(0).cpu().numpy(),
        #                         interpolation='nearest',
        #                         cmap=viridis,
        #                         vmin=0,
        #                         vmax=14)
        #         fig.colorbar(psm, ax=ax)
        #         plt.savefig(os.path.join(debug_path, 'img_{}_gtmask.png'.format(img_id)))
        #         plt.close(fig)
        
        # crop each image and mask into patches, and then start training LSTM for all 38 classes
        patches = unfold(X=translated_images,
                         patch_size=LSTM_config.patch_size) # (128, 4, 3, 128, 128)
        patch_masks = unfold(X=masks,
                             patch_size=LSTM_config.patch_size) # (128, 4, 1, 128, 128)

        # visualize patches and patch masks
        # if i == 0:
        #     for img_id in range(16):
        #         for patch_id in range(4):
        #             save_image(patches[img_id][patch_id], os.path.join(debug_path, 'img_{}_patch_{}.jpg'.format(img_id, patch_id)))
        #             # gt patch masks
        #             # fig = plt.figure(figsize=(1,1))
        #             fig = plt.figure()
        #             ax = plt.Axes(fig, [0., 0., 1., 1.]) # no borders
        #             ax.set_axis_off()
        #             fig.add_axes(ax)
        #             psm = ax.imshow(X=patch_masks[img_id][patch_id].squeeze(0).cpu().numpy(),
        #                             interpolation='nearest',
        #                             cmap=viridis,
        #                             vmin=0,
        #                             vmax=14)
        #             fig.colorbar(psm, ax=ax)
        #             plt.savefig(os.path.join(debug_path, 'img_{}_patch_{}_mask.png'.format(img_id, patch_id)))
        #             plt.close(fig)

        # transform from cropped masks to their semantics
        num_pixels = patch_masks.shape[3] * patch_masks.shape[4] # 128*128=16384
        patch_masks = patch_masks.flatten(start_dim=2, end_dim=4) # (128, 4, 16384)
        mask_semantics = torch.nn.functional.one_hot(patch_masks.long()).sum(dim=2) # (128, 4, 14)
        mask_semantics_ratio = (mask_semantics.float()) / num_pixels # (128, 4, 14)

        # feed the sequence of cancatted mask and semantics to LSTM
        # initialize hidden and cell states
        optimizer_lstm.zero_grad()
        h0, c0 = model_lstm._init_hidden(translated_images.shape[0])
        # (1, 128, 128) (D*num_layers, N=bs, H_out), (1, 128, 128) (D*num_layers, N, H_cell)
        loss_lstm = torch.tensor([0.0]).cuda()

        # Compute MSE loss between output_lstm and next semantics embeddings (128-d)
        output_lstm, (hn, cn) = model_lstm([patch_masks, mask_semantics_ratio], (h0, c0))
        # Forward loss
        loss_lstm = mse_loss(output_lstm[:, :-1, :LSTM_config.semantics_dim], mask_semantics_ratio[:, 1:, :]) #  both (128, 4, 7)
        # Backward loss
        loss_lstm += mse_loss(output_lstm[:, 1:, LSTM_config.semantics_dim:], mask_semantics_ratio[:, :-1, :]) # both (128, 4, 7)

        # M4 backpropagate after all patch iterations
        loss_lstm.backward()
        optimizer_lstm.step()
        train_loss_lstm += loss_lstm.item()

        # Print every 10 batches
        if (i+1) % 10 == 0:
            print("Current process: batch: {}/{}, current batch loss: {}".format(i+1, len(train_loader), loss_lstm.item()))

    # End of Epoch
    # Print and tally average epoch loss
    print("Train Loss (LSTM) over epoch: {:.5f}\n".format(train_loss_lstm / (i+1)))
    log_dict['train_loss_lstm'].append(train_loss_lstm / (i+1))

    # Adjust lr every epoch
    if optimizer_lstm.param_groups[0]['lr'] > LSTM_config.lr_min:
        lr_scheduler_lstm.step()

    # Save model params and losses every epoch
    if (epoch+1) % 10 == 0: 
        print("Saving model: epoch: {}/{}\n".format(epoch+1, LSTM_config.epochs))
        torch.save({'epoch': epoch+1, 
                    'state_dict': model_lstm.state_dict(),
                    'optimizer' : optimizer_lstm.state_dict(),
                    'log_dict': log_dict
                    },
                    os.path.join(save_path, 'checkpoint_{}.pth.tar'.format(epoch+1)))
    torch.save({'epoch': epoch+1, 
                'state_dict': model_lstm.state_dict(),
                'optimizer' : optimizer_lstm.state_dict(),
                'log_dict': log_dict
                },
                os.path.join(save_path, 'checkpoint_latest.pth.tar'))

print('\nFinished Training\n')


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