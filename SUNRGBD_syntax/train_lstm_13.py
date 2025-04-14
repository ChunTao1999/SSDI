import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import imageio.v2 as imageio # 4.19.2023 - tao88
import skimage.transform
import torchvision
import torchvision.transforms as transforms

import torch.optim
import RedNet_model
import RedNet_data
from RedNet_data import image_h, image_w
from utils import utils
from utils.utils import load_ckpt, unfold

# 4.19.2023 - tao88
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pdb
import os
import random
from torchvision.utils import save_image
from config_lstm import LSTM_config
from model_lstm import customizable_LSTM as LSTM
from seg_37_to_13 import seg_mask_convertion


#%% SEED instantiation
SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#%% Arguments, device, configs
parser = argparse.ArgumentParser(description='RedNet Indoor Sementic Segmentation Train')
parser.add_argument('--data-dir', default=None, metavar='DIR', help='path to SUNRGB-D')
parser.add_argument('-b', '--batch-size', default=5, type=int, metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('-o', '--output-dir', default=None, metavar='DIR', help='path to output')
parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--rednet-ckpt', default='', type=str, metavar='PATH', help='path to latest checkpoint for RedNet(default: none)')
myargs = parser.parse_args()

device = torch.device("cuda:0" if myargs.cuda and torch.cuda.is_available() else "cpu")
# Define color map for plotting segmentation mask
num_seg_classes = 13
viridis = mpl.colormaps['viridis'].resampled(num_seg_classes) # 37+1 colors in the sequential colormap
# Import seg_mask_convertion that converts from 38 classes to 14 classes


#%% Train Dataset and dataloader
train_data = RedNet_data.SUNRGBD(transform=transforms.Compose([RedNet_data.scaleNorm(),
                                                               RedNet_data.ToTensor(),
                                                               RedNet_data.Normalize()]),
                                                               phase_train=True, # set this to True to avoid corruptions
                                                               data_dir=myargs.data_dir,
                                                               num_distortion=3,
                                                               ps=128)
train_loader = DataLoader(train_data, 
                          batch_size=myargs.batch_size, 
                          shuffle=True,
                          num_workers=myargs.workers, 
                          pin_memory=False)
num_train = len(train_data)
print("Train Dataset size: {}".format(num_train))


#%% RedNet model and checkpoints (only load for gt vs. pred mask comparison)
model = RedNet_model.RedNet(pretrained=False)
print("Model RedNet:")
load_ckpt(model, None, myargs.rednet_ckpt, device)
model.eval()
model.to(device)
print("Pretrained RedNet model loaded from:{}".format(myargs.rednet_ckpt))


#%% LSTM model, checkpoints, optimizer, and lr_scheduler
model_lstm = LSTM(LSTM_config)
model_lstm.cuda()
model_lstm.train()
print("Model M4:\n", "Train from scratch")
print(model_lstm)

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


#%% Train
# Log and paths
log_dict = {"train_loss_lstm": [],
            "val_loss_lstm": [],
            "val_threshold": [],
            "val_acc_lstm": []}
if not os.path.exists(myargs.output_dir): os.makedirs(myargs.output_dir)
save_path = './LSTM_models_trained_on_RedNet_generated_labels/num_classes=13/input_size=640x480/ps={}_bi-lstm_numlayers={}_startlr={}_epoch={}'.format(LSTM_config.patch_size, LSTM_config.num_layers, LSTM_config.lr_start, LSTM_config.epochs)
if not os.path.exists(save_path): os.makedirs(save_path)

# Enumerate trainloader, pass gt-masks and semantics to LSTM
for epoch in range(0, LSTM_config.epochs):
    print("\nEpoch: {}/{}".format(epoch+1, LSTM_config.epochs))
    train_loss_lstm = 0.0
    model_lstm.train()
    model.eval()
    for batch_idx, sample in enumerate(train_loader):
        image = sample['image'].to(device) # (bs, 3, image_w, image_h)
        depth = sample['depth'].to(device)# (bs, 1, image_w, image_h)
        # label = sample['label'].to(device) # (bs, image_w, image_h), can use 'label1' to 'label4' if smaller sizes needed
        
        # Map the label from 38 classes to 14 classes
        index_map_tensor = torch.tensor([seg_mask_convertion[i] for i in range(38)])
        # label = index_map_tensor[sample['label'].long()].float().to(device) # use label tensor as index
        # Now label has range [0, 13], 14 classes

        # Visualize input image, depth, and gt-mask
        # for img_id in range(image.shape[0]):
        #     save_image(image[img_id], os.path.join(myargs.output_dir, "img_{}.png".format(img_id)))
        #     save_image(depth[img_id], os.path.join(myargs.output_dir, "depth_{}.png".format(img_id)))
        #     fig = plt.figure()
        #     ax = plt.Axes(fig, [0., 0., 1., 1.]) # no borders
        #     ax.set_axis_off()
        #     fig.add_axes(ax)
        #     psm = ax.imshow(X=label[img_id].squeeze().cpu().numpy(),
        #                     interpolation='nearest',
        #                     cmap=viridis,
        #                     vmin=0,
        #                     vmax=14)
        #     fig.colorbar(psm, ax=ax)
        #     plt.savefig(os.path.join(myargs.output_dir, "img_{}_gtmask.png".format(img_id)))
        #     plt.close(fig)

        # Visualize, compare gt-mask with RedNet model predictions
        pred = model(image, depth) 
        pred_labels = torch.max(pred, dim=1, keepdim=True)[1] + 1 # plus one to match gt labels
        pred_labels = (index_map_tensor[pred_labels].float() - 1).to(device) # 13 classes, [0, 12]
        # pdb.set_trace()
        # for img_id in range(image.shape[0]):
        #     fig = plt.figure()
        #     ax = plt.Axes(fig, [0., 0., 1., 1.]) # no borders
        #     ax.set_axis_off()
        #     fig.add_axes(ax)
        #     psm = ax.imshow(X=pred_labels[img_id].squeeze().cpu().numpy(),
        #                     interpolation='nearest',
        #                     cmap=viridis,
        #                     vmin=0,
        #                     vmax=14)
        #     fig.colorbar(psm, ax=ax)
        #     plt.savefig(os.path.join(myargs.output_dir, "img_{}_predmask.png".format(img_id)))
        #     plt.close(fig)

        # Crop the gt-masks into patches, and train the LSTM
        # patch_masks = unfold(X=label.unsqueeze(1), patch_size=LSTM_config.patch_size) # (bs, num_patch, num_ch=1, ps, ps)
        patch_masks = unfold(X=pred_labels, patch_size=LSTM_config.patch_size)
        
        # Visualize the patch gtmasks
        # for img_id in range(image.shape[0]):
        #     for patch_id in range(12):
        #         fig = plt.figure()
        #         ax = plt.Axes(fig, [0., 0., 1., 1.]) # no borders
        #         ax.set_axis_off()
        #         fig.add_axes(ax)
        #         psm = ax.imshow(X=patch_masks[img_id][patch_id].squeeze().cpu().numpy(),
        #                         interpolation='nearest',
        #                         cmap=viridis,
        #                         vmin=0,
        #                         vmax=14)
        #         fig.colorbar(psm, ax=ax)
        #         plt.savefig(os.path.join(myargs.output_dir, 'img_{}_patch_{}_gtmask.png'.format(img_id, patch_id)))
        #         plt.close(fig)
        # pdb.set_trace() # Verified

        # transform from cropped masks to their semantics
        num_pixels = patch_masks.shape[3] * patch_masks.shape[4] # ps*ps
        patch_masks = patch_masks.flatten(start_dim=2, end_dim=4) # (bs, num_patches, num_pixels)
        mask_semantics = torch.nn.functional.one_hot(patch_masks.long(), num_classes=LSTM_config.semantics_dim).sum(dim=2) # (bs, num_patches, num_semantics)
        mask_semantics_ratio = (mask_semantics.float()) / num_pixels # (bs, num_patches, num_semantics)
        
        # feed the sequence of cancatted mask and semantics to LSTM
        # initialize hidden and cell states
        optimizer_lstm.zero_grad()
        h0, c0 = model_lstm._init_hidden(image.shape[0])
        # (2, 16, 14) (D*num_layers, N=bs, H_out), (2, 16, 14) (D*num_layers, N, H_cell), D=2 if bidirectional otherwise 1
        loss_lstm = torch.tensor([0.0]).cuda()

        # Compute MSE loss between output_lstm and next semantics embeddings (128-d)
        output_lstm, (hn, cn) = model_lstm([patch_masks, mask_semantics_ratio], (h0, c0))
        # Forward loss
        loss_lstm = mse_loss(output_lstm[:, :-1, :LSTM_config.semantics_dim], mask_semantics_ratio[:, 1:, :]) # both (bs, num_patches, num_semantics)
        # Backward loss
        loss_lstm += mse_loss(output_lstm[:, 1:, LSTM_config.semantics_dim:], mask_semantics_ratio[:, :-1, :]) # both (bs, num_patches, num_semantics)

        # M4 backpropagate after all patch iterations
        loss_lstm.backward()
        optimizer_lstm.step()
        train_loss_lstm += loss_lstm.item()

        # Print every 20 batches
        if (batch_idx+1) % 500 == 0:
            print("Current process: batch: {}/{}, current batch loss: {}".format(batch_idx+1, len(train_loader), loss_lstm.item()))

    # End of Epoch
    # Print and tally average epoch loss
    print("Train Loss (LSTM) over epoch: {:.5f}\n".format(train_loss_lstm / (batch_idx+1)))
    log_dict['train_loss_lstm'].append(train_loss_lstm / (batch_idx+1))

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
    # Plot loss curve
    if epoch+1 == LSTM_config.epochs:
        plt.figure()
        plt.plot(np.arange(epoch+1), log_dict['train_loss_lstm'])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss curve over epochs")
        plt.savefig(os.path.join(save_path, "loss_curve.png"))
        plt.close()

pdb.set_trace()


