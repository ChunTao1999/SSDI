#%% Imports
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torchvision.utils import save_image
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth = 160)
np.set_printoptions(linewidth = 160)
np.set_printoptions(precision=4)
np.set_printoptions(suppress='True')
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12 

from utils_custom_tvision_functions import get_dataloaders, plot_curve
from configs.config_test_LSTM import config_test
from AVS_functions import crop_five, extract_and_resize_masks
# LSTM
from configs.config_lstm import AVS_config as config_lstm
from modules.model_lstm import customizable_LSTM as LSTM
from modules.model_lstm_masks import customizable_LSTM as LSTM_masks
from weight_init import weight_init
# PiCIE
from commons import *
from modules import fpn
# Plot results (important)
from utils import set_logger, plot_and_hist, compute_average_mask

# Colormap
num_seg_classes = 7
viridis = mpl.colormaps['viridis'].resampled(num_seg_classes)


#%% Instantiate parameters, dataloaders, and model
# Parameters
device = torch.device("cuda")

# SEED instantiation
SEED = config_test.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Dataloaders
train_loader, loss_weights, count_samples_train = get_dataloaders(config_test, loader_type=config_test.train_loader_type)
print(f"Trainset size: {count_samples_train}")
# valid_loader                = get_dataloaders(config_test, loader_type=config_test.valid_loader_type) # val split
test_loader, count_samples_test  = get_dataloaders(config_test, loader_type=config_test.test_loader_type) # test split
print(f"Testset size: {count_samples_test}")

# Loss(es)
# bce_loss                    = nn.BCEWithLogitsLoss(pos_weight=loss_weights.to(device))  
ce_loss                     = nn.CrossEntropyLoss()

#%% LSTM-related
# LSTM model (M4)
model_4 = LSTM_masks(config_lstm)
if config_test.data_parallel: # for now, don't use DataParallel with LSTM model
    model_4 = nn.DataParallel(model_4)
# from pretrained
ckpt_4 = torch.load(config_lstm.ckpt_dir_model_M4)
model_4.load_state_dict(ckpt_4['state_dict'])
for p in model_4.parameters():
    p.requires_grad_(False)
print("Model M4:\n", "Loaded from: {}".format(config_lstm.ckpt_dir_model_M4))
print(model_4)
model_4.cuda()
model_4.eval()

# check number of parameters and number of parameters that require gradient
print("Number of params: {}".format(sum(p.numel() for p in model_4.parameters())))
print("Number of trainable params: {}".format(sum(p.numel() for p in model_4.parameters() if p.requires_grad)))


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


#%% LSTM Optimizer
optimizer_M4 = torch.optim.Adam(model_4.parameters(), lr=config_lstm.lr_start, weight_decay=config_lstm.weight_decay)
lr_scheduler_M4 = torch.optim.lr_scheduler.MultiStepLR(optimizer_M4, gamma=config_lstm.gamma, milestones=config_lstm.milestones, verbose=True)


#%% Resnet+FPN model for glimpse context (not needed in training, needed in test)
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

model_FPN = fpn.PanopticFPN(args)
if config_test.data_parallel:
    model_FPN = nn.DataParallel(model_FPN) # comment for using one device
model_FPN = model_FPN.cuda()
checkpoint  = torch.load(args.eval_path)
if config_test.data_parallel:
    model_FPN.load_state_dict(checkpoint['state_dict'])
else:
    # If model_FPN is not data-parallel
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    model_FPN.load_state_dict(new_state_dict)
print("\nModel FPN:\n", "Loaded from: {}\n".format(args.eval_path))
for p in model_FPN.parameters():
    p.requires_grad_(False)
model_FPN.eval()


#%% Print experiment name
if config_test.landmark_shuffle:
    exp_name = 'landmark_shuffle'
    print("Current experiment: landmark_shuffle, num_distortions:{}, box_size:{}\n".format(config_test.num_distortion, config_test.box_size))
    save_path = './LSTM_test_results_avg_semantics_similarity/{}/num_distortion_{}_box_size_{}'.format(exp_name, config_test.num_distortion, config_test.box_size)
elif config_test.black_box:
    exp_name = 'black_box'
    print("Current experiment: black_box, num_boxes:{}, box_size:{}\n".format(config_test.num_box, config_test.box_size))
    save_path = './LSTM_test_results_avg_semantics_similarity/{}/num_box_{}_box_size_{}'.format(exp_name, config_test.num_box, config_test.box_size)
elif config_test.gaussian_blur:
    exp_name = 'gaussian_blur'
    print("Current experiment: gaussian_blur, num_boxes:{}, box_size:{}\n".format(config_test.num_box, config_test.box_size))
    save_path = './LSTM_test_results_avg_semantics_similarity/{}/_num_box_{}_box_size_{}'.format(exp_name, config_test.num_box, config_test.box_size)
elif config_test.puzzle_solving:
    exp_name = 'puzzle_solving'
    print("Current experiment: puzzle_solving, num_permute:{}, box_size:{}\n".format(config_test.num_permute, config_test.box_size))
    save_path = './LSTM_test_results_avg_semantics_similarity/{}/num_permute_{}_box_size_{}'.format(exp_name, config_test.num_permute, config_test.box_size)
else:
    exp_name = 'correct'
    print("Current experiment: correct images")
    save_path = './LSTM_test_results_avg_semantics_similarity/{}'.format(exp_name)
if not os.path.exists(save_path): os.makedirs(save_path)
print("Saving results to: {}".format(save_path))
# pdb.set_trace()


#%% Testing

with torch.no_grad():
    model_4.eval()

    # initialize empty torch tensors
    # sum_masks = torch.zeros(5, 64, 64, 7)
    # sum_semantics = torch.zeros(5, 7)
    # from trainval set, determine the most represented class for each pixel location
    # 11.6: imitate tarin_lstm_crop_masks_and_semantics_bilstm.py
    # print("Begin testing on trainset to get average masks and semantics.")
    # for i, (index, images, targets) in enumerate(train_loader):
    #     translated_iamges, masks = images.to(device), targets[4].to(device)
    #     # over the trainval set, collect the masks and compute average mask and semantics for each patch
    #     masks = torch.round(masks * 7)

    #     masks_extracted_resized = extract_and_resize_masks(masks, resized_height=64, resized_width=64) #(128, 5, 1, 64, 64)
    #     mask_semantics = torch.nn.functional.one_hot(masks_extracted_resized.long()).squeeze() # (128, 5, 64, 64, 7)
    #     if i == 0:
    #         sum_semantics = torch.sum(mask_semantics, dim=0, keepdim=False)
    #     else:
    #         sum_semantics += torch.sum(mask_semantics, dim=0, keepdim=False)
    #     if (i+1) % 50 == 0:
    #         print("{}/{}".format(i+1, len(train_loader)))
    # # can get average masks and semantics from sum_semantics
    # torch.save(sum_semantics, './LSTM_test_results_avg_semantics_similarity/sum_patch_semantics_trainset.pt')

    # Loading average masks and semantics obtained from train dataset
    sum_semantics = torch.load('./LSTM_test_results_avg_semantics_similarity/sum_patch_semantics_trainset.pt') # (5, 64, 64, 7)
    sum_semantics_dom = torch.argmax(sum_semantics, dim=-1) # (5, 64, 64), here we use argmax, can switch to mean
    sum_semantics_ratio_dom = nnF.one_hot(sum_semantics_dom.flatten(start_dim=1, end_dim=2), num_classes=7).sum(dim=1)
    sum_semantics_ratio_dom =  sum_semantics_ratio_dom.float() / (sum_semantics.shape[1] * sum_semantics.shape[2]) # (5,7)
    pdb.set_trace()
    for patch_id in range(5):
        fig = plt.figure(figsize=(1, 1))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        fig.add_axes(ax)
        psm = ax.imshow(sum_semantics_dom[patch_id].squeeze().cpu().numpy(),
                        interpolation='nearest',
                        cmap=viridis,
                        vmin=0,
                        vmax=7)
        cbar = fig.colorbar(psm, ticks=[0, 1, 2, 3, 4, 5, 6])
        # cbar.set_ticks([0, 1, 2, 3, 4, 5, 6])
        plt.savefig(os.path.join('./LSTM_test_results_avg_semantics_similarity', 'patch_{}_dom_mask.png'.format(patch_id)))
        plt.close(fig)


    # initialize empty torch tensors
    dist_correct_imgs, dist_corrupted_imgs = [], []
    if config_test.puzzle_solving:
        dist_batch_allperms_mean = []
    sum_masks_all_batches = torch.randn(5, 64, 64)
    sum_semantics_all_batches = torch.randn(5, 7)
    sum_pred_semantics_forward_all_batches = torch.randn(4, 7)
    sum_pred_semantics_backward_all_batches = torch.randn(4, 7)

    print("Begin testing.")
    for i, (index, images, targets) in enumerate(test_loader): 
        translated_images, targets, bbox_targets, corrupt_labels = images.to(device), targets[0].float().to(device), targets[1].to(device), targets[-1].to(device) 

        # visualize correct and corrupted test images
        if i == 0:
            for img_id in range(16):
                if config_test.puzzle_solving:
                    for perm_id in range(config_test.num_permute+1):
                        save_image(translated_images[img_id][perm_id], os.path.join(save_path, 'img_{}_perm_{}.png'.format(img_id, perm_id)))
                else:
                    save_image(translated_images[img_id], os.path.join(save_path, 'img_{}.png'.format(img_id)))

        batch_patches = crop_five(translated_images, 
                                  left_coords=[64,128,64,128,96], 
                                  top_coords=[64,64,128,128,96], 
                                  widths=[64,64,64,64,64], 
                                  heights=[64,64,64,64,64],              
                                  resized_height=256, 
                                  resized_width=256) # (128, 5, 3, 256, 256) or (128, 5, 4, 3, 256, 256)

        if config_test.puzzle_solving:
            batch_patches = torch.permute(batch_patches, (0, 2, 1, 3, 4, 5)) # (128, num_permute, 5, 3, 256, 256)

        # pass batch of each patch to the model_FPN to get predicted masks
        if not config_test.puzzle_solving:
            for patch_id in range(batch_patches.shape[1]):
                patches = batch_patches[:, patch_id, :, :, :] # (128, 3, 256, 256)
                masks_predicted = model_FPN(patches) # (128, 7, 64, 64)
                lbl_predicted = masks_predicted.topk(1, dim=1)[1] # (128, 1, 64, 64), dtype=long
                if patch_id == 0:
                    batch_masks = lbl_predicted
                else:
                    batch_masks = torch.cat((batch_masks, lbl_predicted), dim=1) # (128, 5, 64, 64)
            if i == 0:
                sum_masks_all_batches = batch_masks.sum(dim=0) # (5, 64, 64)
            else:
                sum_masks_all_batches += batch_masks.sum(dim=0)            
        else:
            for perm_id in range(batch_patches.shape[1]): # 4 perms
                for patch_id in range(batch_patches.shape[2]): # 5 patches
                    patches = batch_patches[:, perm_id, patch_id, :, :, :] # (128, 3, 256, 256)
                    masks_predicted = model_FPN(patches) # (128, 7, 64, 64)
                    lbl_predicted = masks_predicted.topk(1, dim=1)[1] # (128, 1, 64, 64), dtype=long
                    if patch_id == 0:
                        batch_patch_masks = lbl_predicted
                    else:
                        batch_patch_masks = torch.cat((batch_patch_masks, lbl_predicted), dim=1) # (128, 5, 64, 64)
                if perm_id == 0:
                    batch_masks = torch.unsqueeze(batch_patch_masks, dim=1) # (128, 1, 5, 64, 64)
                else:
                    batch_masks = torch.cat((batch_masks, torch.unsqueeze(batch_patch_masks, dim=1)), dim=1) # (128, 4, 5, 64, 64)

        if not config_test.puzzle_solving:
            # batch masks now has shape (128, #_perms, 64, 64), process the masks into semantics
            num_pixels = batch_masks.shape[2] * batch_masks.shape[3]
            batch_masks_flattened = batch_masks.flatten(start_dim=2, end_dim=3) # (128, 5, 4096)
            mask_semantics = nnF.one_hot(batch_masks_flattened, num_classes=7).sum(dim=2) # (128, 5, 7)
            mask_semantics_ratio = (mask_semantics.float()) / num_pixels # (128, 5, 7), normalized
            # collect sum of gt semantics
            if i == 0:
                sum_semantics_all_batches = mask_semantics_ratio.sum(dim=0)
            else:
                sum_semantics_all_batches += mask_semantics_ratio.sum(dim=0)

            # pass batch_masks_flattened to LSTM, sequence length is 5
            h0, c0 = model_4._init_hidden(translated_images.shape[0])
            output_lstm, (hn, cn) = model_4([batch_masks_flattened.float(), mask_semantics_ratio], (h0, c0)) # (128, 5, 7)

            # collect pred semantics
            if i == 0:
                sum_pred_semantics_forward_all_batches = output_lstm[:, :-1, :7].sum(dim=0)
                sum_pred_semantics_backward_all_batches = output_lstm[:, 1:, 7:].sum(dim=0)
            else:
                sum_pred_semantics_forward_all_batches += output_lstm[:, :-1, :7].sum(dim=0)
                sum_pred_semantics_backward_all_batches += output_lstm[:, 1:, 7:].sum(dim=0)

            # record dist curves for each image in dist_batch_imgs
            # 11.6, transform the tensor "sum_semantics_ratio_dom" and compare with the predictions
            stacked_semantics_ratio = torch.stack([sum_semantics_ratio_dom]*translated_images.shape[0], dim=0) # (128, 5, 7)
            dist_batch_imgs_for = torch.norm((output_lstm[:, :-1, :7] - stacked_semantics_ratio[:, 1:, :]), dim=2, keepdim=False) # (128, 4)
            dist_batch_imgs_back = torch.norm((output_lstm[:, 1:, 7:] - stacked_semantics_ratio[:, :-1, :]), dim=2, keepdim=False) # (128, 4)

            dist_batch_imgs = dist_batch_imgs_for + dist_batch_imgs_back
            dist_batch_imgs_both = torch.zeros(output_lstm.shape[0], output_lstm.shape[1]).to(device)
            dist_batch_imgs_both[:, 1:] = dist_batch_imgs_for
            dist_batch_imgs_both[:, :-1] += dist_batch_imgs_back # (128, 5)

            # 5.4.2023 - below for demo
            if i == 0:
                dist_correct_imgs = dist_batch_imgs[corrupt_labels==0]
                dist_corrupted_imgs = dist_batch_imgs[corrupt_labels==1]

                dist_correct_imgs_for = dist_batch_imgs_for[corrupt_labels==0]
                dist_corrupted_imgs_for = dist_batch_imgs_for[corrupt_labels==1]
                dist_correct_imgs_back = dist_batch_imgs_back[corrupt_labels==0]
                dist_corrupted_imgs_back = dist_batch_imgs_back[corrupt_labels==1]
                dist_correct_imgs_both = dist_batch_imgs_both[corrupt_labels==0]
                dist_corrupted_imgs_both = dist_batch_imgs_both[corrupt_labels==1]
            else:
                dist_correct_imgs = torch.cat((dist_correct_imgs, dist_batch_imgs[corrupt_labels==0]), dim=0)
                dist_corrupted_imgs = torch.cat((dist_corrupted_imgs, dist_batch_imgs[corrupt_labels==1]), dim=0)

                dist_correct_imgs_for = torch.cat((dist_correct_imgs_for, dist_batch_imgs_for[corrupt_labels==0]), dim=0) # (#, 4)
                dist_corrupted_imgs_for = torch.cat((dist_corrupted_imgs_for, dist_batch_imgs_for[corrupt_labels==1]), dim=0) # (#, 4)
                dist_correct_imgs_back = torch.cat((dist_correct_imgs_back, dist_batch_imgs_back[corrupt_labels==0]), dim=0) # (#, 4)
                dist_corrupted_imgs_back = torch.cat((dist_corrupted_imgs_back, dist_batch_imgs_back[corrupt_labels==1]), dim=0) # (#, 4)
                dist_correct_imgs_both = torch.cat((dist_correct_imgs_both, dist_batch_imgs_both[corrupt_labels==0]), dim=0)
                dist_corrupted_imgs_both = torch.cat((dist_corrupted_imgs_both, dist_batch_imgs_both[corrupt_labels==1]), dim=0)
        else:
            num_pixels = batch_masks.shape[-2] * batch_masks.shape[-1]
            batch_masks_flattened = batch_masks.flatten(start_dim=-2, end_dim=-1) # (128, 4, 5, 4096)
            mask_semantics = nnF.one_hot(batch_masks_flattened, num_classes=7).sum(dim=3) # (128, 4, 5, 7)
            mask_semantics_ratio = (mask_semantics.float()) / num_pixels # (128, 4, 5, 7), normalized
            stacked_semantics_ratio = torch.stack([sum_semantics_ratio_dom]*translated_images.shape[0], dim=0) # (128, 5, 7)

            for perm_id in range(config_test.num_permute + 1):
                h0, c0 = model_4._init_hidden(translated_images.shape[0])
                output_lstm, (hn, cn) = model_4([batch_masks_flattened[:, perm_id, :, :].float(), mask_semantics_ratio[:, perm_id, :, :]], (h0, c0))
                dist_batch_imgs = torch.norm((output_lstm[:, :-1, :7] - stacked_semantics_ratio[:, 1:, :]), dim=2, keepdim=False)
                dist_batch_imgs += torch.norm((output_lstm[:, 1:, 7:] - stacked_semantics_ratio[:, :-1, :]), dim=2, keepdim=False) # (128, num_patch-1)
                if perm_id == 0: # fist permutation is correct
                    dist_allperms_mean = dist_batch_imgs.mean(1).unsqueeze(1)
                else:
                    dist_allperms_mean = torch.cat((dist_allperms_mean, dist_batch_imgs.mean(1).unsqueeze(1)), dim=1) #(bs, num_permute+1)
            if i == 0:
                dist_batch_allperms_mean = dist_allperms_mean
            else:
                dist_batch_allperms_mean = torch.cat((dist_batch_allperms_mean, dist_allperms_mean), dim=0) # (num_samples, num_permute+1)

        
        if (i+1) % 50 == 0:
            print("{}/{}".format(i+1, len(test_loader)))

    # End of Test Epoch
    if not config_test.puzzle_solving:
        plot_and_hist(config_test,
                      save_path,
                      dist_correct_imgs=dist_correct_imgs, dist_corrupted_imgs=dist_corrupted_imgs,
                      dist_correct_imgs_for=dist_correct_imgs_for, dist_corrupted_imgs_for=dist_corrupted_imgs_for,
                      dist_correct_imgs_back=dist_correct_imgs_back, dist_corrupted_imgs_back=dist_corrupted_imgs_back,
                      dist_correct_imgs_both=dist_correct_imgs_both, dist_corrupted_imgs_both=dist_corrupted_imgs_both)
    else:
        plot_and_hist(config_test,
                      save_path,
                      dist_batch_allperms_mean=dist_batch_allperms_mean)
    compute_average_mask(save_path, 
                         sum_masks_all_batches,
                         sum_semantics_all_batches,
                         sum_pred_semantics_forward_all_batches,
                         sum_pred_semantics_backward_all_batches,
                         count_samples=count_samples_test,
                         cmap=viridis)
    pdb.set_trace()
    
