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

device = torch.device("cuda:0" if myargs.cuda and torch.cuda.is_available() else "cpu")

# Define color map for plotting segmentation mask
num_seg_classes = 13+1
viridis_14 = mpl.colormaps['viridis'].resampled(num_seg_classes) # 13+1 colors in the sequential colormap
viridis_13 = ListedColormap(viridis_14(np.linspace(0, 1, 14))[1:])


#%% Test Dataset and dataloader
test_data = RedNet_data.SUNRGBD(transform=transforms.Compose([RedNet_data.scaleNorm(),
                                                              RedNet_data.ToTensor(),
                                                              RedNet_data.Normalize()]),
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
    for batch_idx, sample in enumerate(test_loader):
        image = sample['image'].to(device) # half of the images should be corrupted (bs, (num_permute), 3, 640, 480)
        depth = sample['depth'].unsqueeze(-3).to(device) # depth also half corrupted (bs, (num_permute), 1, 640, 480)
        corrupt_labels = sample['corrupt_lbl'].to(device) # (bs)

        if not LSTM_config.corruption_type == "puzzle_solving":
            pred = model(image, depth) 
            # pred_labels = torch.max(pred, dim=1, keepdim=False)[1] + 1 # (bs, image_w, image_h), range [1, 37]
            pred_labels = torch.max(pred, dim=1, keepdim=False)[1].float().to(device)

            # Visualize input image and depth, and gt vs. predicted seg maps
            for img_id in range(8, 16):
            #     save_image(image[img_id], os.path.join(myargs.output_dir, "img_{}.png".format(img_id)))
            #     save_image(depth[img_id], os.path.join(myargs.output_dir, "img_{}_depth.png".format(img_id)))
            #     fig = plt.figure()
            #     ax = plt.Axes(fig, [0., 0., 1., 1.]) # no borders
            #     ax.set_axis_off()
            #     fig.add_axes(ax)
            #     psm = ax.imshow(X=labels[img_id].squeeze().cpu().numpy(),
            #                     interpolation='nearest',
            #                     cmap=viridis_14,
            #                     vmin=0,
            #                     vmax=14)
            #     fig.colorbar(psm, ax=ax)
                # plt.savefig(os.path.join(myargs.output_dir, "img_{}_gtmask.png".format(img_id)))
                # plt.close(fig)

                fig = plt.figure()
                ax = plt.Axes(fig, [0., 0., 1., 1.]) # no borders
                ax.set_axis_off()
                fig.add_axes(ax)
                psm = ax.imshow(X=pred_labels[img_id].squeeze().cpu().numpy(),
                                interpolation='nearest',
                                cmap=viridis_13,
                                vmin=1,
                                vmax=14)
                fig.colorbar(psm, ax=ax)
                plt.savefig(os.path.join(myargs.output_dir, "img_{}_predmask.png".format(img_id)))
                plt.close(fig)

            pdb.set_trace()
            # Crop the predicted seg maps into patches, and test with LSTM
            pred_patch_masks = unfold(X=pred_labels.unsqueeze(1), patch_size=LSTM_config.patch_size) # (bs, num_patch, num_ch=1, ps, ps)

            # Visualize patch masks
            # for img_id in range(image.shape[0]):
            #     for patch_id in range(12):
            #         fig = plt.figure()
            #         ax = plt.Axes(fig, [0., 0., 1., 1.]) # no borders
            #         ax.set_axis_off()
            #         fig.add_axes(ax)
            #         psm = ax.imshow(X=pred_patch_masks[img_id][patch_id].squeeze().cpu().numpy(),
            #                         interpolation='nearest',
            #                         cmap=viridis_13,
            #                         vmin=1,
            #                         vmax=14)
            #         fig.colorbar(psm, ax=ax)
            #         plt.savefig(os.path.join(myargs.output_dir, 'img_{}_pred_patch_{}_mask.png'.format(img_id, patch_id)))
            #         plt.close(fig)
        
            # Transform from cropped masks to their semantics
            num_pixels = pred_patch_masks.shape[3] * pred_patch_masks.shape[4] # ps*ps
            pred_patch_masks = pred_patch_masks.flatten(start_dim=2, end_dim=4) # (bs, num_patch, ps*ps)
            mask_semantics = torch.nn.functional.one_hot(pred_patch_masks.long(), num_classes=37).sum(dim=2) # (bs, num_patch, num_semantics)
            mask_semantics_ratio = (mask_semantics.float()) / num_pixels # (bs, num_patch, num_semantics)
            
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

            # record dist curves for each image in dist_batch_imgs
            dist_batch_imgs = torch.norm((output_lstm[:, :-1, :LSTM_config.semantics_dim] - mask_semantics_ratio[:, 1:, :]), dim=2, keepdim=False) # (bs, num_patch-1)
            dist_batch_imgs += torch.norm((output_lstm[:, 1:, LSTM_config.semantics_dim:] - mask_semantics_ratio[:, :-1, :]), dim=2, keepdim=False) # (bs, num_patch-1)
            # concat to the dist array for all test images
            if batch_idx == 0:
                dist_correct_imgs = dist_batch_imgs[corrupt_labels==0]
                dist_corrupted_imgs = dist_batch_imgs[corrupt_labels==1]
            else:
                dist_correct_imgs = torch.cat((dist_correct_imgs, dist_batch_imgs[corrupt_labels==0]), dim=0)
                dist_corrupted_imgs = torch.cat((dist_corrupted_imgs, dist_batch_imgs[corrupt_labels==1]), dim=0)
        else: # 4.28.2023-tao88: in case of puzzle solving, dimension is different
            for perm_id in range(image.shape[1]):
                pred = model(image[:, perm_id, :, :, :], depth[:, perm_id, :, :, :])
                pred_labels = torch.max(pred, dim=1, keepdim=False)[1].float().to(device)
                pred_patch_masks = unfold(X=pred_labels.unsqueeze(1), patch_size=LSTM_config.patch_size) # (bs, num_patch, num_ch=1, ps, ps)
                if perm_id == 0:
                    batch_patch_masks = pred_patch_masks
                else:
                    batch_patch_masks = torch.cat((batch_patch_masks, pred_patch_masks), dim=-3) # (bs, num_patch, num_permute+1, ps, ps)

            # batch_patch_masks is prepared
            batch_patch_masks = torch.permute(batch_patch_masks, (0, 2, 1, 3, 4)) # (bs, num_permute+1, num_patch, ps, ps)
            # prepared batch_patch_semantics
            num_pixels = batch_patch_masks.shape[-2] * batch_patch_masks.shape[-1] # ps*ps
            batch_patch_masks_flattened = batch_patch_masks.flatten(start_dim=-2, end_dim=-1) # (bs, num_permute+1, num_patch, ps*ps)
            mask_semantics = nnF.one_hot(batch_patch_masks_flattened.long(), num_classes=37).sum(dim=3) # (bs, num_permute+1, num_patch, semantics_dim)
            mask_semantics_ratio = (mask_semantics.float()) / num_pixels # (bs, num_permute+1, num_patch, semantics_dim), normalized

            # run each perm sequence through trained LSTM, and collect the average residences for each perm sequence
            for perm_id in range(image.shape[1]):
                h0, c0 = model_lstm._init_hidden(image.shape[0])
                output_lstm, (hn, cn) = model_lstm([batch_patch_masks_flattened[:, perm_id, :, :], mask_semantics_ratio[:, perm_id, :, :]], (h0, c0))
                dist_batch_imgs = torch.norm((output_lstm[:, :-1, :LSTM_config.semantics_dim] - mask_semantics_ratio[:, perm_id, 1:, :]), dim=2, keepdim=False)
                dist_batch_imgs += torch.norm((output_lstm[:, 1:, LSTM_config.semantics_dim:] - mask_semantics_ratio[:, perm_id, :-1, :]), dim=2, keepdim=False) # (bs, num_patch-1)
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
    # Print and tally average epoch loss
    if not LSTM_config.corruption_type == "puzzle_solving":
        print("Test Loss (LSTM) over epoch: {:.5f}\n".format(test_loss_lstm / (batch_idx+1)))
        log_dict['test_loss_lstm'].append(test_loss_lstm / (batch_idx+1))

        # Use torch.hist for picking threshold and computing test accuracy
        dist_correct_imgs, dist_corrupted_imgs = np.array(dist_correct_imgs.cpu()), np.array(dist_corrupted_imgs.cpu()) # both (len//2, num_patch-1)
        dist_correct_mean, dist_corrupted_mean = dist_correct_imgs.mean(1), dist_corrupted_imgs.mean(1)
        hist_correct, bin_edges_correct = np.histogram(dist_correct_mean, bins=np.arange(0, 2.51, 0.01))
        hist_corrupted, bin_edges_corrupted = np.histogram(dist_corrupted_mean, bins=np.arange(0, 2.51, 0.01))
        # Stats
        print("Number of test samples taken into account in hist: correct: {}, corrupted: {}".format(hist_correct.sum(), hist_corrupted.sum()))
        print("Correct sample mean distance range: {} to {}".format(dist_correct_mean.min(), dist_correct_mean.max()))
        print("Corrupted sample mean distance range: {} to {}".format(dist_corrupted_mean.min(), dist_corrupted_mean.max()))

        # Determine threshold
        cutoff_index = np.where(hist_corrupted>hist_correct)[0][0]
        first_occ_threshold = bin_edges_correct[cutoff_index] # first occurence
        print("First occurence where there're more corrupted samples than correct samples: %.2f" % (first_occ_threshold))
        # num_fp, num_fn = hist_correct[cutoff_index:].sum(), hist_corrupted[:cutoff_index].sum()
        # num_tp, num_tn = hist_corrupted[cutoff_index:].sum(), hist_correct[:cutoff_index].sum()
        # test_acc = (num_tn + num_tp) / (num_fn + num_fp + num_tn + num_tp) * 100
        # print("To distinguish correct from corrupted samples, the threshold of mean distance between patch semantics and its prediction is %.2f" % (threshold))
        # print("By using the previous threshold, the test accuracy is %.3f%%" % (test_acc))

        for cutoff_index in range(0, bin_edges_correct.shape[0]):
            threshold = bin_edges_correct[cutoff_index] # first occurence
            num_fp, num_fn = hist_correct[cutoff_index:].sum(), hist_corrupted[:cutoff_index].sum()
            num_tp, num_tn = hist_corrupted[cutoff_index:].sum(), hist_correct[:cutoff_index].sum()
            test_acc = (num_tn + num_tp) / (num_fn + num_fp + num_tn + num_tp) * 100
            print("Threshold: %.2f" %(threshold), "By using that threshold, the test accuracy is %.3f%%, fn %d, fp %d, tn %d, tp %d" % (test_acc, num_fn, num_fp, num_tn, num_tp))
        # pdb.set_trace()
        
        # Use plt for visualization
        plt.figure()
        # plt.hist(dist_correct_imgs.mean(1), range=(0, 0.8), bins=16, alpha=0.5, color='b', label='correct')
        # plt.hist(dist_corrupted_imgs.mean(1), range=(0, 0.8), bins=16, alpha=0.5, color='r', label='corrupted')
        plt.hist(dist_correct_mean, bins=np.arange(0, 2.5, 0.01), alpha=0.7, color='b', label='correct')
        plt.hist(dist_corrupted_mean, bins=np.arange(0, 2.5, 0.01), alpha=0.7, color='r', label='corrupted')
        plt.xticks(np.arange(0, 2.5, 0.5))
        plt.yticks(np.arange(0, 110, 10))
        plt.title('LSTM Prediction Results')
        plt.xlabel('Average dist. over 5 patches')
        plt.ylabel('Frequency')
        plt.legend()
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
