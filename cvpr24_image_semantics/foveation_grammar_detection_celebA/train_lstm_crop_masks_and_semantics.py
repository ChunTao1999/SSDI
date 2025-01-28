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
# 3.24.2023 - tao88: different config_3 for creation of valid set
from AVS_config_M3_celeba_valid import AVS_config as AVS_config_for_M3_valid
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
# 3.24.2023 - tao88
config_3_valid = AVS_config_for_M3_valid
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
valid_loader                  = get_dataloaders(config_3_valid, loader_type=config_3.valid_loader_type) # test split

# Loss(es)
bce_loss                    = nn.BCEWithLogitsLoss(pos_weight=loss_weights.to(device))  
ce_loss                     = nn.CrossEntropyLoss()

# Foveation models (no need foveation now)

#%% LSTM-related
# LSTM configs
config_4 = config_for_LSTM

# LSTM model (M4)
model_4 = LSTM_masks(config_4)
if config_3.data_parallel: # for now, don't use DataParallel with LSTM model
    model_4 = nn.DataParallel(model_4)
model_4.cuda()
# model_4.apply(weight_init)
# print("Model M4: \n", "Training M4 from scratch!\n")
# print(model_4)
# model_4.train()

# 3.28.2023 - if load from pretrained model
ckpt_4 = torch.load(config_4.ckpt_dir_model_M4)
model_4.load_state_dict(ckpt_4['state_dict'])
for p in model_4.parameters():
    p.requires_grad_(False)
print("Model M4:\n", "Loaded from: {}\n".format(config_4.ckpt_dir_model_M4))
print(model_4)
model_4.eval()

# check number of parameters and number of parameters that require gradient
print("Number of params: {}".format(sum(p.numel() for p in model_4.parameters())))
print("Number of trainable params: {}".format(sum(p.numel() for p in model_4.parameters() if p.requires_grad)))


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
print("\nModel FPN:\n", "Loaded from: {}\n".format(args.eval_path))
# logger.info('Loaded checkpoint. [epoch {}]'.format(checkpoint['epoch']))
for p in model_FPN.parameters():
    p.requires_grad_(False)
model_FPN.eval()


#%% AVS-specific functions, methods, and parameters
from AVS_functions import location_bounds, extract_and_resize_glimpses_for_batch, crop_five, extract_and_resize_masks

lower, upper = location_bounds(glimpse_w=config_3.glimpse_size_init[0], input_w=config_3.full_res_img_size[0]) #TODO: make sure this works on both x and y coordinates


#%% Training log_dict and save_path
log_dict = {"train_loss_lstm": [],
            "val_loss_lstm": [],
            "val_threshold": [],
            "val_acc_lstm": []} # 2.20.2023 - tao88
save_path = './bi-lstm_models_crop_masks_and_semantics_with_validation_lr={}'.format(config_4.lr_start)
if config_4.use_contr_loss:
    save_path = os.path.join(save_path, 'contr_loss')
else:
    save_path = os.path.join(save_path, 'mse_loss') 
if not os.path.exists(save_path):
    os.makedirs(save_path)


#%% Start training LSTM
for epoch in range(0, config_3.epochs):
    print("\nEpoch: {}/{}".format(epoch+1, config_3.epochs))
    train_loss_lstm = 0.0
    model_4.train()

    for i, (indices, images, targets) in enumerate(train_loader):
        translated_images, masks = images.to(device), targets[4].to(device)
        # indices: indices of images in the dataset (128, 1);
        # images: translated images (128, 3, 256, 256); 
        # targets[4]: masks (128, 1, 256, 256); # can be set to wild or aligned masks

        # some error in compute_rl_loss for last batch (only when doing foveation)
        # if i == len(train_loader) - 1:
        #     continue

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

        # Compute MSE loss between output_lstm and next semantics embeddings (128-d)
        output_lstm, (hn, cn) = model_4([masks_extracted_resized, mask_semantics_ratio], (h0, c0))

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
    # Print losses
    print("Train Loss (LSTM) over epoch: {:.5f}\n".format(train_loss_lstm / (i+1)))
    log_dict['train_loss_lstm'].append(train_loss_lstm / (i+1))


    # End of one epoch, should validate on the half corrupted validation set.
    print("Epoch of training done, start validation.")
    with torch.no_grad():
        model_4.eval()
        dist_correct_imgs, dist_corrupted_imgs = [], []
        val_loss_lstm = 0.0

        for i_val, (index, images, targets) in enumerate(valid_loader):
            translated_images, targets, bbox_targets, corrupt_labels = images.to(device), targets[0].float().to(device), targets[1].to(device), targets[-1].to(device)
            # pdb.set_trace() # done

            # crop the five patches from the batch of images
            batch_patches = crop_five(translated_images, 
                                        left_coords=[64,128,64,128,96], 
                                        top_coords=[64,64,128,128,96], 
                                        widths=[64,64,64,64,64], 
                                        heights=[64,64,64,64,64],              
                                        resized_height=256, 
                                        resized_width=256) # (128, 5, 3, 256, 256)

            # pass batch of each patch to the model_FPN to get predicted masks
            for patch_id in range(batch_patches.shape[1]):
                patches = batch_patches[:, patch_id, :, :, :] # (128, 3, 256, 256)
                masks_predicted = model_FPN(patches) # (128, 7, 64, 64)
                lbl_predicted = masks_predicted.topk(1, dim=1)[1] # (128, 1, 64, 64), dtype=long
                if patch_id == 0:
                    batch_masks = lbl_predicted
                else:
                    batch_masks = torch.cat((batch_masks, lbl_predicted), dim=1)

            # 3.23.2023 - tao88: batch_masks now has shape (128, 5, 64, 64), process the masks into semantics
            num_pixels = batch_masks.shape[2] * lbl_predicted.shape[3]
            batch_masks_flattened = batch_masks.flatten(start_dim=2, end_dim=3) # (128, 5, 4096)
            mask_semantics = torch.nn.functional.one_hot(batch_masks_flattened, num_classes=7).sum(dim=2) # (128, 5, 7)
            mask_semantics_ratio = (mask_semantics.float()) / num_pixels # (128, 5, 7), normalize to ratio by dividing by 4096

            # pass batch_masks_flattened to LSTM, sequence length is 5
            h0, c0 = model_4._init_hidden(translated_images.shape[0])
            loss_lstm_valid = torch.tensor([0.0]).cuda()
            output_lstm, (hn, cn) = model_4([batch_masks_flattened.float(), mask_semantics_ratio], (h0, c0)) # (128, 5, 7)
            loss_lstm_valid = mse_loss(output_lstm[:, :-1, :7], mask_semantics_ratio[:, 1:, :]) + mse_loss(output_lstm[:, :-1, :7], mask_semantics_ratio[:, 1:, :])
            val_loss_lstm += loss_lstm_valid

            # record dist curves for each image in dist_batch_imgs
            dist_batch_imgs = torch.norm((output_lstm[:, :-1, :] - mask_semantics_ratio[:, 1:, :]), dim=2, keepdim=False) # (128, 4)
            # concat to the dist array for all val images
            if i_val == 0:
                dist_correct_imgs = dist_batch_imgs[corrupt_labels==0]
                dist_corrupted_imgs = dist_batch_imgs[corrupt_labels==1]
            else:
                dist_correct_imgs = torch.cat((dist_correct_imgs, dist_batch_imgs[corrupt_labels==0]), dim=0)
                dist_corrupted_imgs = torch.cat((dist_corrupted_imgs, dist_batch_imgs[corrupt_labels==1]), dim=0)

        print("Validation Loss (LSTM): {:.5f}".format(val_loss_lstm / (i_val+1)))
        log_dict['val_loss_lstm'].append(val_loss_lstm / (i_val+1))
        # 3.28.2023 - We now have dist curves of all validation images, we can plot the histogram, and pick a threshold for best validation accuracy. Determine the threshold as the first bin edge where correct samples are greater than corrupted samples.
        dist_correct_imgs, dist_corrupted_imgs = np.array(dist_correct_imgs.cpu()), np.array(dist_corrupted_imgs.cpu()) # both (#, 4)

        # Use torch.hist for picking threshold and computing validation accuracy
        dist_correct_mean, dist_corrupted_mean = dist_correct_imgs.mean(1), dist_corrupted_imgs.mean(1)
        hist_correct, bin_edges_correct = np.histogram(dist_correct_mean, bins=np.arange(0, 1.55, 0.05))
        hist_corrupted, bin_edges_corrupted = np.histogram(dist_corrupted_mean, bins=np.arange(0, 1.55, 0.05))

        cutoff_index = np.where(hist_corrupted>hist_correct)[0][0]
        threshold = bin_edges_correct[cutoff_index] # first occurence
        num_fn, num_fp = hist_correct[cutoff_index:].sum(), hist_corrupted[:cutoff_index].sum()
        num_tn, num_tp = hist_corrupted[cutoff_index:].sum(), hist_correct[:cutoff_index].sum()
        val_acc = (num_tn + num_tp) / (num_fn + num_fp + num_tn + num_tp) * 100

        print("To distinguish correct from corrupted samples, the threshold of mean distance between patch semantics and its prediction is %.2f" % (threshold))
        print("By using the previous threshold, the validation accuracy is %.3f%%" % (val_acc))
        log_dict['val_threshold'].append(threshold)
        log_dict['val_acc_lstm'].append(val_acc)

        # Use plt for visualization
        plt.figure()
        # plt.hist(dist_correct_imgs.mean(1), range=(0, 0.8), bins=16, alpha=0.5, color='b', label='correct')
        # plt.hist(dist_corrupted_imgs.mean(1), range=(0, 0.8), bins=16, alpha=0.5, color='r', label='corrupted')
        plt.hist(dist_correct_mean, bins=np.arange(0, 0.85, 0.05), alpha=0.7, color='b', label='correct')
        plt.hist(dist_corrupted_mean, bins=np.arange(0, 0.85, 0.05), alpha=0.7, color='r', label='corrupted')
        plt.xticks(np.arange(0, 0.9, 0.1))
        plt.yticks(np.arange(0, 3100, 100))
        plt.title('LSTM Prediction Results')
        plt.xlabel('Average dist. over 5 patches')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig('/home/nano01/a/tao88/3.29_fivepatch_on_val_FPN_20_epochs/hist_distavg_on_val_epoch_{}.png'.format(19+1))
        plt.close()
        print("Histogram plotted and saved, for epoch {}".format(epoch+1))


    # Adjust lr every epoch
    # if optimizer_M4.param_groups[0]['lr'] > config_4.lr_min:
    lr_scheduler_M4.step()

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