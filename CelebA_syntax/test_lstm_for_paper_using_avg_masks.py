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
from utils import set_logger, plot_and_hist, compute_average_mask, calculate_iou

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
    save_path = './LSTM_test_results_avg_masks/{}/num_distortion_{}_box_size_{}'.format(exp_name, config_test.num_distortion, config_test.box_size)
elif config_test.black_box:
    exp_name = 'black_box'
    print("Current experiment: black_box, num_boxes:{}, box_size:{}\n".format(config_test.num_box, config_test.box_size))
    save_path = './LSTM_test_results_avg_masks/{}/num_box_{}_box_size_{}'.format(exp_name, config_test.num_box, config_test.box_size)
elif config_test.gaussian_blur:
    exp_name = 'gaussian_blur'
    print("Current experiment: gaussian_blur, num_boxes:{}, box_size:{}\n".format(config_test.num_box, config_test.box_size))
    save_path = './LSTM_test_results_avg_masks/{}/_num_box_{}_box_size_{}'.format(exp_name, config_test.num_box, config_test.box_size)
elif config_test.puzzle_solving:
    exp_name = 'puzzle_solving'
    print("Current experiment: puzzle_solving, num_permute:{}, box_size:{}\n".format(config_test.num_permute, config_test.box_size))
    save_path = './LSTM_test_results_avg_masks/{}/num_permute_{}_box_size_{}'.format(exp_name, config_test.num_permute, config_test.box_size)
else:
    exp_name = 'correct'
    print("Current experiment: correct images")
    save_path = './LSTM_test_results_avg_masks/{}'.format(exp_name)
if not os.path.exists(save_path): os.makedirs(save_path)
print("Saving results to: {}".format(save_path))
# pdb.set_trace()


#%% Testing

with torch.no_grad():
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
    #     masks_extracted_resized = masks_extracted_resized # (128, 5, 64, 64)
    #     mask_semantics = torch.nn.functional.one_hot(masks_extracted_resized.long()).squeeze() # (128, 5, 64, 64, 7)
    #     if i == 0:
    #         sum_semantics = torch.sum(mask_semantics, dim=0, keepdim=False)
    #     else:
    #         sum_semantics += torch.sum(mask_semantics, dim=0, keepdim=False)
    #     if (i+1) % 50 == 0:
    #         print("{}/{}".format(i+1, len(train_loader)))
    # # can get average masks and semantics from sum_semantics
    # torch.save(sum_semantics, './LSTM_test_results_avg_masks/sum_patch_semantics.pt')

    # Loading average masks and semantics obtained from train dataset
    sum_semantics = torch.load('./LSTM_test_results_avg_masks/sum_patch_semantics_trainset.pt') # (5, 64, 64, 7)
    sum_semantics_dom = torch.argmax(sum_semantics, dim=-1) # (5, 64, 64), here we use argmax, can switch to mean
    sum_semantics_ratio_dom = nnF.one_hot(sum_semantics_dom.flatten(start_dim=1, end_dim=2), num_classes=7).sum(dim=1)
    sum_semantics_ratio_dom =  sum_semantics_ratio_dom.float() / (sum_semantics.shape[1] * sum_semantics.shape[2]) # (5,7)
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
        plt.savefig(os.path.join('./LSTM_test_results_avg_masks', 'patch_{}_dom_mask.png'.format(patch_id)))
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
        
        # compute the distance between batch_masks and avg batch masks sum_semantics_dom
        if not config_test.puzzle_solving:
            batch_iou = torch.zeros(translated_images.shape[0], 7) # (128, 7)
            for k in range(translated_images.shape[0]):
                batch_iou[k] = calculate_iou(batch_masks[k], sum_semantics_dom, 7)
            batch_mean_iou = torch.mean(batch_iou, dim=1) # (128,)
            if i == 0:
                iou_correct_imgs = batch_mean_iou[corrupt_labels==0]
                iou_corrupted_imgs = batch_mean_iou[corrupt_labels==1]
            else:
                iou_correct_imgs = torch.cat((iou_correct_imgs, batch_mean_iou[corrupt_labels==0]), dim=0)
                iou_corrupted_imgs = torch.cat((iou_corrupted_imgs, batch_mean_iou[corrupt_labels==1]), dim=0)
        else:
            batch_iou = torch.zeros(translated_images.shape[0], config_test.num_permute+1, 7)
            for k in range(translated_images.shape[0]):
                for perm_id in range(config_test.num_permute+1):
                    batch_iou[k][perm_id] = calculate_iou(batch_masks[k][perm_id], sum_semantics_dom, 7)
            batch_mean_iou = torch.mean(batch_iou, dim=2) # (128, 4)
            if i == 0:
                all_batch_mean_iou = batch_mean_iou
            else:
                all_batch_mean_iou = torch.cat((all_batch_mean_iou, batch_mean_iou), dim=0)

        if (i+1) % 50 == 0:
            print("{}/{}".format(i+1, len(test_loader)))

    # End of Test Epoch
    if not config_test.puzzle_solving:
        # Determine threshold and make histogram
        iou_correct_imgs, iou_corrupted_imgs = np.array(iou_correct_imgs.cpu()), np.array(iou_corrupted_imgs.cpu())
        iou_correct_mean, iou_corrupted_mean = iou_correct_imgs, iou_corrupted_imgs
        hist_correct, bin_edges_correct = np.histogram(iou_correct_mean, bins=np.arange(0, 6.51, 0.05))
        hist_corrupted, bin_edges_corrupted = np.histogram(iou_corrupted_mean, bins=np.arange(0, 6.51, 0.05))
        print("Number of test samples taken into account in hist: correct: {}, corrupted: {}".format(hist_correct.sum(), hist_corrupted.sum()))
        print("Correct sample mean distance range: {} to {}".format(iou_correct_mean.min(), iou_correct_mean.max()))
        print("Corrupted sample mean distance range: {} to {}".format(iou_corrupted_mean.min(), iou_corrupted_mean.max()))
        # Determine threshold
        cutoff_index = np.where(hist_corrupted<hist_correct)[0][0]
        first_occ_threshold = bin_edges_correct[cutoff_index] # first occurence
        print("First occurence where there're more corrupted samples than correct samples: %.2f" % (first_occ_threshold))
        for cutoff_index in range(0, bin_edges_correct.shape[0]):
            threshold = bin_edges_correct[cutoff_index] # first occurence
            num_fp, num_fn = hist_correct[:cutoff_index].sum(), hist_corrupted[cutoff_index:].sum()
            num_tp, num_tn = hist_corrupted[:cutoff_index].sum(), hist_correct[cutoff_index:].sum()
            test_acc = (num_tn + num_tp) / (num_fn + num_fp + num_tn + num_tp) * 100
            det_acc = (num_tp) / (num_fn + num_tp) * 100
            print("Threshold: %.2f" %(threshold), "test accuracy is %.3f%%, det accuracy is %.3f%%, fn %d, fp %d, tn %d, tp %d" % (test_acc, det_acc, num_fn, num_fp, num_tn, num_tp))
    else:
        all_batch_mean_iou = np.array(all_batch_mean_iou.cpu()) # (num_samples, num_permute+1)
        # if entry 0 < all other entries, correct prediction!
        num_correct = (all_batch_mean_iou[:,0]>all_batch_mean_iou[:,1:].min(axis=1)).sum()
        num_samples = all_batch_mean_iou.shape[0]
        test_acc = (num_correct / num_samples) * 100
        print("Number of samples when the correct puzzle is picked: %d" % (num_correct))
        print("Total count of samples in the testset: %d" % (num_samples))
        print("The test accuracy is %.3f%%" % (test_acc))
    pdb.set_trace()
    
