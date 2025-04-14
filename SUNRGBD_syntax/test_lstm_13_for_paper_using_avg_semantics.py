#%% Imports
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as nnF
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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
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
parser = argparse.ArgumentParser(description='RedNet Indoor Sementic Segmentation Test')
parser.add_argument('--data-dir', default=None, metavar='DIR', help='path to SUNRGB-D')
parser.add_argument('-b', '--batch-size', default=5, type=int, metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('-o', '--output-dir', default=None, metavar='DIR', help='path to output')
parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--rednet-ckpt', default='', type=str, metavar='PATH', help='path to latest checkpoint for RedNet(default: none)')
myargs = parser.parse_args()
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12 

device = torch.device("cuda:0" if myargs.cuda and torch.cuda.is_available() else "cpu")

# Define color map for plotting segmentation mask
num_seg_classes = 13+1
viridis_14 = mpl.colormaps['viridis'].resampled(num_seg_classes) # 13+1 colors in the sequential colormap
viridis_13 = ListedColormap(viridis_14(np.linspace(0, 1, 14))[1:])


#%% Test Dataset and dataloader
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


transform=transforms.Compose([RedNet_data.scaleNorm(),
                              RedNet_data.ToTensor(),
                              RedNet_data.Normalize()])
test_data = RedNet_data.SUNRGBD(transform=transform,
                                phase_train=False,
                                data_dir=myargs.data_dir,
                                all_corrupt=LSTM_config.all_corrupt,
                                corruption_type=LSTM_config.corruption_type,
                                num_permute=LSTM_config.num_permute,
                                num_distortion=LSTM_config.num_distortion,
                                num_box=LSTM_config.num_box,
                                ps=LSTM_config.patch_size)
test_loader = DataLoader(test_data, 
                         batch_size=myargs.batch_size, 
                         shuffle=True,
                         num_workers=myargs.workers, 
                         pin_memory=False)
num_test = len(test_data)
print("Test Dataset size: {}".format(num_test))


#%% RedNet model and checkpoints
model = RedNet_model.RedNet(pretrained=False)
print("Model RedNet:")
load_ckpt(model, None, myargs.rednet_ckpt, device)
model.eval()
model.to(device)
print("Pretrained RedNet model loaded from:{}".format(myargs.rednet_ckpt))


#%% LSTM model, checkpoints, and optimizer
model_lstm = LSTM(LSTM_config)
model_lstm.to(device)
ckpt_lstm = torch.load(LSTM_config.ckpt_dir_model_M4)
model_lstm.load_state_dict(ckpt_lstm['state_dict'])
for p in model_lstm.parameters():
    p.requires_grad_(False)
print("Model M4:\n", "Loaded from: {}".format(LSTM_config.ckpt_dir_model_M4))
model_lstm.eval()
print(model_lstm)



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


#%% Test
# Logs, output dir (for visualization), and save path (for test results)
log_dict = {"test_loss_lstm": [],
            "test_threshold": [],
            "test_acc_lstm": []}
if not os.path.exists(myargs.output_dir): os.makedirs(myargs.output_dir)
save_path = './LSTM_test_results_model_on_generated_masks/num_classes={}/input_size=640x480/{}/ps={}_num_distortion_{}_num_permute_{}_num_box_{}_bi-lstm_numlayers={}_startlr={}_epoch={}'.format(LSTM_config.semantics_dim, LSTM_config.corruption_type, LSTM_config.patch_size, LSTM_config.num_distortion, LSTM_config.num_permute, LSTM_config.num_box, LSTM_config.num_layers, LSTM_config.lr_start, LSTM_config.epochs)
if not os.path.exists(save_path): os.makedirs(save_path)
print("Saving results to: {}".format(save_path))

# Enumerate testloader
with torch.no_grad():
    test_loss_lstm = 0.0
    dist_correct_imgs, dist_corrupted_imgs = [], []
    model.eval()
    model_lstm.eval()
    index_map_tensor = torch.tensor([seg_mask_convertion[i] for i in range(38)])

    # train loop
    # for batch_idx, sample in enumerate(train_loader):
    #     image = sample['image'].to(device) # (bs, 3, image_w, image_h)
    #     depth = sample['depth'].to(device)# (bs, 1, image_w, image_h)
    #     pred = model(image, depth)
    #     pred_labels = torch.max(pred, dim=1, keepdim=True)[1] + 1 # plus one to match gt label values
    #     pred_labels = (index_map_tensor[pred_labels].float() - 1)
    #     patch_masks = unfold(X=pred_labels,
    #                          patch_size=LSTM_config.patch_size)
    #     patch_masks = nnF.one_hot(patch_masks.long()).squeeze()
    #     if batch_idx == 0:
    #         sum_semantics = torch.sum(patch_masks, dim=0, keepdim=False) # (12, 160, 160, 13)
    #     else:
    #         sum_semantics += torch.sum(patch_masks, dim=0, keepdim=False)
    #     if (batch_idx + 1) % 50 == 0:
    #         print(f"{batch_idx + 1}/{len(train_loader)}")
    # # can get average masks and semantics from sum_semantics
    # torch.save(sum_semantics, f'./LSTM_test_results_model_on_generated_masks/num_classes=13/input_size=640x480/sum_patch_semantics_trainset_{LSTM_config.patch_size}.pt')
    # pdb.set_trace()
    
    sum_semantics = torch.load(f'./LSTM_test_results_model_on_generated_masks/num_classes=13/input_size=640x480/sum_patch_semantics_trainset_{LSTM_config.patch_size}.pt') # (12 or 48, 160, 160, 13)
    sum_semantics_dom = torch.argmax(sum_semantics, dim=-1) # (12, 160, 160), here we use argmax, can switch to mean
    sum_semantics_ratio_dom = nnF.one_hot(sum_semantics_dom.flatten(start_dim=1, end_dim=2), num_classes=13).sum(dim=1) # (12, 13)
    sum_semantics_ratio_dom =  sum_semantics_ratio_dom.float() / (sum_semantics.shape[1] * sum_semantics.shape[2]) # (12, 13)
    # for patch_id in range(48):
    #     fig = plt.figure(figsize=(1, 1))
    #     ax = plt.Axes(fig, [0., 0., 1., 1.])
    #     # ax.set_axis_off()
    #     fig.add_axes(ax)
    #     psm = ax.imshow(sum_semantics_dom[patch_id].squeeze().cpu().numpy(),
    #                     interpolation='nearest',
    #                     cmap=viridis_13,
    #                     vmin=1,
    #                     vmax=14)
    #     cbar = fig.colorbar(psm, ax=ax)
    #     # cbar.set_ticks([0, 1, 2, 3, 4, 5, 6])
    #     plt.savefig(os.path.join('./LSTM_test_results_model_on_generated_masks/num_classes=13/input_size=640x480', f'avg_masks_ps={LSTM_config.patch_size}', 'patch_{}_dom_mask.png'.format(patch_id)))
    #     plt.close(fig)
    # pdb.set_trace()

        
    # test loop
    for batch_idx, sample in enumerate(test_loader):
        image = sample['image'].to(device) # half of the images should be corrupted (bs, 3, 640, 480)
        depth = sample['depth'].unsqueeze(-3).to(device) # depth also half corrupted (bs, 1, 640, 480)
        corrupt_labels = sample['corrupt_lbl'].to(device) # (bs)
        indices = sample['index'].to(device)

        if not LSTM_config.corruption_type == "puzzle_solving":
            pred = model(image, depth) 
            pred_labels = torch.max(pred, dim=1, keepdim=False)[1] + 1 # (bs, image_w, image_h), range [1, 37]
            # pred_labels = torch.max(pred, dim=1, keepdim=False)[1]
            # transform both gt and predicted labels to from 37-class to 13-class seg map
            index_map_tensor = torch.tensor([seg_mask_convertion[i] for i in range(38)])
            labels = index_map_tensor[sample['label'].long()].float().to(device) # gt labels also half corrupted (bs, 640, 480), range [0, 37]
            pred_labels = (index_map_tensor[pred_labels] - 1).float().to(device) # (bs, 640, 480)

            # Visualize input image and depth, and gt vs. predicted seg maps
            # for img_id in range(1):
            #     save_image(image[img_id], os.path.join(myargs.output_dir, "imgorig_{}_batch_{}.png".format(img_id, batch_idx)))
                # save_image(depth[img_id], os.path.join(myargs.output_dir, "imgorig_{}_depth.png".format(img_id)))

                # fig = plt.figure()
                # ax = plt.Axes(fig, [0., 0., 1., 1.]) # no borders
                # ax.set_axis_off()
                # fig.add_axes(ax)
                # psm = ax.imshow(X=labels[img_id].squeeze().cpu().numpy(),
                #                 interpolation='nearest',
                #                 cmap=viridis_14,
                #                 vmin=0,
                #                 vmax=14)
                # fig.colorbar(psm, ax=ax)
                # plt.savefig(os.path.join(myargs.output_dir, "img_{}_gtmask.png".format(img_id)))
                # plt.close(fig)
                # fig = plt.figure()
                # ax = plt.Axes(fig, [0., 0., 1., 1.]) # no borders
                # ax.set_axis_off()
                # fig.add_axes(ax)
                # psm = ax.imshow(X=pred_labels[img_id].squeeze().cpu().numpy(),
                #                 interpolation='nearest',
                #                 cmap=viridis_13,
                #                 vmin=1,
                #                 vmax=14)
                # fig.colorbar(psm, ax=ax)
                # plt.savefig(os.path.join(myargs.output_dir, "img_{}_predmask.png".format(img_id)))
                # plt.close(fig)
            

            # Crop the predicted seg maps into patches, and test with LSTM
            pred_patch_masks = unfold(X=pred_labels.unsqueeze(1), patch_size=LSTM_config.patch_size) # (bs, num_patch, num_ch=1, ps, ps)
            
            
            # 5.20 - for demo
            if batch_idx == 0:
                sum_masks_all_batches = pred_patch_masks.sum(dim=0)
            else:
                sum_masks_all_batches += pred_patch_masks.sum(dim=0)

            # Visualize patch masks
            # for img_id in range(1):
            #     # save_image(image[img_id], os.path.join(myargs.output_dir, "img_{}.png".format(img_id)))
            #     # save_image(depth[img_id], os.path.join(myargs.output_dir, "img_{}_depth.png".format(img_id)))
            #     for patch_id in range(16):
            #         fig = plt.figure()
            #         ax = plt.Axes(fig, [0., 0., 1., 1.]) # no borders
            #         ax.set_axis_off()
            #         fig.add_axes(ax)
            #         psm = ax.imshow(X=pred_patch_masks[img_id][patch_id].squeeze().cpu().numpy(),
            #                         interpolation='nearest',
            #                         cmap=viridis_13,
            #                         vmin=1,
            #                         vmax=14)
            #         # fig.colorbar(psm, ax=ax)
            #         plt.savefig(os.path.join(myargs.output_dir, 'img_{}_pred_patch_{}_mask.png'.format(img_id, patch_id)))
            #         plt.close(fig)
            

            # Transform from cropped masks to their semantics
            num_pixels = pred_patch_masks.shape[3] * pred_patch_masks.shape[4] # ps*ps
            pred_patch_masks = pred_patch_masks.flatten(start_dim=2, end_dim=4) # (bs, num_patch, ps*ps)
            mask_semantics = torch.nn.functional.one_hot(pred_patch_masks.long(), num_classes=13).sum(dim=2) # (bs, num_patch, num_semantics)
            mask_semantics_ratio = (mask_semantics.float()) / num_pixels # (bs, num_patch, num_semantics)
            
            # 5.20 - for demo
            if batch_idx == 0:
                sum_semantics_all_batches = mask_semantics_ratio.sum(dim=0)
            else:
                sum_semantics_all_batches += mask_semantics_ratio.sum(dim=0)

            h0, c0 = model_lstm._init_hidden(image.shape[0])
            # (2, 16, 14) (D*num_layers, N=bs, H_out), (2, 16, 14) (D*num_layers, N, H_cell), D=2 if bidirectional otherwise 1
            loss_lstm = torch.tensor([0.0]).cuda()

            # Compute MSE loss between output_lstm and next semantics embeddings (128-d)
            output_lstm, (hn, cn) = model_lstm([pred_patch_masks, mask_semantics_ratio], (h0, c0))
            # Forward loss
            loss_lstm = mse_loss(output_lstm[:, :-1, :LSTM_config.semantics_dim], mask_semantics_ratio[:, 1:, :]) #  both (bs, num_patch-1, num_semantics)
            # Backward loss
            loss_lstm += mse_loss(output_lstm[:, 1:, LSTM_config.semantics_dim:], mask_semantics_ratio[:, :-1, :]) # both (bs, num_patch-1, num_semantics)
            test_loss_lstm += loss_lstm.item()

            # 5.20 - for demo
            if batch_idx == 0:
                sum_pred_semantics_forward_all_batches = output_lstm[:, :-1, :LSTM_config.semantics_dim].sum(dim=0)
                sum_pred_semantics_backward_all_batches = output_lstm[:, 1:, LSTM_config.semantics_dim:].sum(dim=0)
            else:
                sum_pred_semantics_forward_all_batches += output_lstm[:, :-1, :LSTM_config.semantics_dim].sum(dim=0)
                sum_pred_semantics_backward_all_batches += output_lstm[:, 1:, LSTM_config.semantics_dim:].sum(dim=0)

            # 5.6.2023 - for paper:
            stacked_semantics_ratio = torch.stack([sum_semantics_ratio_dom]*image.shape[0], dim=0).to(device) #(bs, 12, 13)
            dist_batch_imgs_for = torch.norm((output_lstm[:, :-1, :LSTM_config.semantics_dim] - stacked_semantics_ratio[:, 1:, :]), dim=2, keepdim=False) # (bs, num_patch-1)
            dist_batch_imgs_back = torch.norm((output_lstm[:, 1:, LSTM_config.semantics_dim:] - stacked_semantics_ratio[:, :-1, :]), dim=2, keepdim=False) # (bs, num_patch-1)
            dist_batch_imgs = dist_batch_imgs_for + dist_batch_imgs_back # (bs, num_patch-1)

            # concat to the dist array for all test images
            if batch_idx == 0:
                dist_correct_imgs = dist_batch_imgs[corrupt_labels==0]
                dist_corrupted_imgs = dist_batch_imgs[corrupt_labels==1]
            else:
                dist_correct_imgs = torch.cat((dist_correct_imgs, dist_batch_imgs[corrupt_labels==0]), dim=0)
                dist_corrupted_imgs = torch.cat((dist_corrupted_imgs, dist_batch_imgs[corrupt_labels==1]), dim=0)
       
        else:
            stacked_semantics_ratio = torch.stack([sum_semantics_ratio_dom]*image.shape[0], dim=0).to(device) #(bs, 12, 13)
            # for perm_id in range(image.shape[1]):
            #     save_image(image[0, perm_id, :, :, :], os.path.join(myargs.output_dir, "perm_{}.png".format(perm_id)))
            for perm_id in range(image.shape[1]):
                pred = model(image[:, perm_id, :, :, :], depth[:, perm_id, :, :, :])
                pred_labels = torch.max(pred, dim=1, keepdim=False)[1] + 1
                index_map_tensor = torch.tensor([seg_mask_convertion[i] for i in range(38)])
                pred_labels = (index_map_tensor[pred_labels] -1).float().to(device)

                pred_patch_masks = unfold(X=pred_labels.unsqueeze(1), patch_size=LSTM_config.patch_size) # (bs, num_patch, num_ch=1, ps, ps)
                if perm_id == 0:
                    batch_patch_masks = pred_patch_masks
                else:
                    batch_patch_masks = torch.cat((batch_patch_masks, pred_patch_masks), dim=-3) # (bs, num_patch, num_permute+1, ps, ps)
            batch_patch_masks = torch.permute(batch_patch_masks, (0, 2, 1, 3, 4))
            # prepared batch_patch_semantics
            num_pixels = batch_patch_masks.shape[-2] * batch_patch_masks.shape[-1] # ps*ps
            batch_patch_masks_flattened = batch_patch_masks.flatten(start_dim=-2, end_dim=-1) # (bs, num_permute+1, num_patch, ps*ps)
            mask_semantics = nnF.one_hot(batch_patch_masks_flattened.long(), num_classes=13).sum(dim=3) # (bs, num_permute+1, num_patch, semantics_dim)
            mask_semantics_ratio = (mask_semantics.float()) / num_pixels # (bs, num_permute+1, num_patch, semantics_dim), normalized

            # run each perm sequence through trained LSTM, and collect the average residences for each perm sequence
            for perm_id in range(image.shape[1]):
                h0, c0 = model_lstm._init_hidden(image.shape[0])
                output_lstm, (hn, cn) = model_lstm([batch_patch_masks_flattened[:, perm_id, :, :], mask_semantics_ratio[:, perm_id, :, :]], (h0, c0))
                dist_batch_imgs = torch.norm((output_lstm[:, :-1, :LSTM_config.semantics_dim] - stacked_semantics_ratio[:, 1:, :]), dim=2, keepdim=False)
                dist_batch_imgs += torch.norm((output_lstm[:, 1:, LSTM_config.semantics_dim:] - stacked_semantics_ratio[:, :-1, :]), dim=2, keepdim=False) # (bs, num_patch-1)
                if perm_id == 0: # fist permutation is correct
                    dist_batch_allperms_mean = dist_batch_imgs.mean(1).unsqueeze(1)
                else:
                    dist_batch_allperms_mean = torch.cat((dist_batch_allperms_mean, dist_batch_imgs.mean(1).unsqueeze(1)), dim=1) #(bs, num_permute+1)
            # collect to dist_batch_allperms_mean over the entire testset
            if batch_idx == 0:
                dist_allperms_mean = dist_batch_allperms_mean
            else:
                dist_allperms_mean = torch.cat((dist_allperms_mean, dist_batch_allperms_mean), dim=0) # (num_samples, num_permute+1)

        # Print every 50 batches
        if (batch_idx+1) % 50 == 0:
            if not LSTM_config.corruption_type == "puzzle_solving":
                print("Current process: batch: {}/{}, current batch loss: {}".format(batch_idx+1, len(test_loader), loss_lstm.item()))
            else:
                print("Current process: batch: {}/{}".format(batch_idx+1, len(test_loader)))

    # End of test epoch
    if not LSTM_config.corruption_type == "puzzle_solving":
        avg_masks_all_batches = sum_masks_all_batches / 5050
        avg_semantics_all_batches = sum_semantics_all_batches / 5050
        avg_pred_semantics_forward_all_batches = sum_pred_semantics_forward_all_batches / 5050
        avg_pred_semantics_backward_all_batches = sum_pred_semantics_backward_all_batches / 5050
        
        for patch_id in range(12):
            fig = plt.figure(figsize=(1, 1))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            # ax.set_axis_off()
            fig.add_axes(ax)
            psm = ax.imshow(avg_masks_all_batches[patch_id].squeeze().cpu().numpy(),
                        interpolation='nearest',
                        cmap=viridis_13,
                        vmin=1,
                        vmax=14)
            cbar = fig.colorbar(psm, ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
            # cbar.set_ticks([0, 1, 2, 3, 4, 5, 6])
            plt.savefig(os.path.join(myargs.output_dir, 'avgpatch_{}_mask_noround.png'.format(patch_id)))
            plt.close(fig)
        print(avg_semantics_all_batches)
        print(avg_pred_semantics_forward_all_batches)
        print(avg_pred_semantics_backward_all_batches)
        print(torch.norm((avg_pred_semantics_forward_all_batches - avg_semantics_all_batches[1:]), dim=1, keepdim=False))
        print(torch.norm((avg_pred_semantics_backward_all_batches - avg_semantics_all_batches[:-1]), dim=1, keepdim=False))
    
    if not LSTM_config.corruption_type == "puzzle_solving":
        print("Test Loss (LSTM) over epoch: {:.5f}\n".format(test_loss_lstm / (batch_idx+1)))
        log_dict['test_loss_lstm'].append(test_loss_lstm / (batch_idx+1))

        # Use torch.hist for picking threshold and computing test accuracy
        dist_correct_imgs, dist_corrupted_imgs = np.array(dist_correct_imgs.cpu()), np.array(dist_corrupted_imgs.cpu()) # both (len//2, num_patch-1)
        dist_correct_mean, dist_corrupted_mean = dist_correct_imgs.sum(1), dist_corrupted_imgs.sum(1)

        hist_correct, bin_edges_correct = np.histogram(dist_correct_mean, bins=np.arange(0, 120.05, 0.2))
        hist_corrupted, bin_edges_corrupted = np.histogram(dist_corrupted_mean, bins=np.arange(0, 120.05, 0.2))
        # Stats
        print("Number of test samples taken into account in hist: correct: {}, corrupted: {}".format(hist_correct.sum(), hist_corrupted.sum()))
        print("Correct sample mean distance range: {} to {}".format(dist_correct_mean.min(), dist_correct_mean.max()))
        print("Corrupted sample mean distance range: {} to {}".format(dist_corrupted_mean.min(), dist_corrupted_mean.max()))

        # Determine threshold
        cutoff_index = np.where(hist_corrupted>hist_correct)[0][0]
        first_occ_threshold = bin_edges_correct[cutoff_index] # first occurence
        print("First occurence where there're more corrupted samples than correct samples: %.2f" % (first_occ_threshold))
        for cutoff_index in range(0, bin_edges_correct.shape[0]):
            threshold = bin_edges_correct[cutoff_index] # first occurence
            num_fp, num_fn = hist_correct[cutoff_index:].sum(), hist_corrupted[:cutoff_index].sum()
            num_tp, num_tn = hist_corrupted[cutoff_index:].sum(), hist_correct[:cutoff_index].sum()
            test_acc = (num_tn + num_tp) / (num_fn + num_fp + num_tn + num_tp) * 100
            det_acc = (num_tp) / (num_fn + num_tp) * 100
            print("Threshold: %.2f" %(threshold), "test accuracy %.3f%%, det accuracy %.3f%%, fn %d, fp %d, tn %d, tp %d" % (test_acc, det_acc, num_fn, num_fp, num_tn, num_tp))
        # 48, 80
        plt.figure()
        plt.hist(dist_correct_mean, bins=np.arange(0, 85.05, 0.2), alpha=0.7, color='b', label='correct')
        plt.hist(dist_corrupted_mean, bins=np.arange(0, 85.05, 0.2), alpha=0.7, color='r', label='corrupted')
        plt.xticks(np.arange(0, 85, 10))
        plt.yticks(np.arange(0, 47, 5))
        plt.xlabel('Total residual', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.legend(["correct", "corrupted"], fontsize="18", loc ="upper right")        
        plt.savefig(os.path.join(save_path, 'hist_distavg_on_test.png'))
        plt.close()
        print("Histogram on testset plotted and saved")

    else:
        dist_allperms_mean = np.array(dist_allperms_mean.cpu()) # (num_samples, num_permute+1)
        # if entry 0 < all other entries, correct prediction!
        num_correct = (dist_allperms_mean[:,0]<dist_allperms_mean[:,1:].min(axis=1)).sum()
        num_samples = dist_allperms_mean.shape[0]
        test_acc = (num_correct / num_samples) *100
        print("Number of samples when the correct puzzle is picked: %d" % (num_correct))
        print("Total count of samples in the testset: %d" % (num_samples))
        print("The test accuracy is %.3f%%" % (test_acc))

        # Histogram for visualizing distribution of correct vs. fake puzzles
        plt.figure()
        plt.hist(dist_allperms_mean[:,0], bins=np.arange(0, 2.55, 0.05), alpha=0.7, color='b', label='correct')
        plt.hist(dist_allperms_mean[:,1:].flatten(), bins=np.arange(0, 2.55, 0.05), alpha=0.7, color='r', label='corrupted')
        plt.xticks(np.arange(0, 2.55, 0.5))
        plt.yticks(np.arange(0, 2100, 100))
        plt.title('LSTM Prediction Results')
        plt.xlabel('Average dist. over sequence of patches')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'hist_puzzle_solving.png'))
        plt.close()
        print("Histogram on test plotted and saved")

print('\nFinished Testing!\n')
pdb.set_trace()
