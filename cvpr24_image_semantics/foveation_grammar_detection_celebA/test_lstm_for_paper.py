#%% Imports
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torchviz import make_dot
import matplotlib as mpl
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth = 160)
np.set_printoptions(linewidth = 160)
np.set_printoptions(precision=4)
np.set_printoptions(suppress='True')

mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12 

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

# Colormap
num_seg_classes = 7
viridis = mpl.colormaps['viridis'].resampled(num_seg_classes)

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
# train_loader, loss_weights   = get_dataloaders(config_3, loader_type=config_3.train_loader_type) # train split
valid_loader                = get_dataloaders(config_3_valid, loader_type=config_3_valid.valid_loader_type) # val split
test_loader                 = get_dataloaders(config_3_valid, loader_type=config_3_valid.test_loader_type) # test split

# Loss(es)
# bce_loss                    = nn.BCEWithLogitsLoss(pos_weight=loss_weights.to(device))  
ce_loss                     = nn.CrossEntropyLoss()

#%% LSTM-related
# LSTM configs
config_4 = config_for_LSTM

# LSTM model (M4)
model_4 = LSTM_masks(config_4)
if config_3.data_parallel: # for now, don't use DataParallel with LSTM model
    model_4 = nn.DataParallel(model_4)
model_4.cuda()
# 3.28.2023 - if load from pretrained model
ckpt_4 = torch.load(config_4.ckpt_dir_model_M4)
model_4.load_state_dict(ckpt_4['state_dict'])
for p in model_4.parameters():
    p.requires_grad_(False)
print("Model M4:\n", "Loaded from: {}".format(config_4.ckpt_dir_model_M4))
print(model_4)
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
optimizer_M4 = torch.optim.Adam(model_4.parameters(), lr=config_4.lr_start, weight_decay=config_4.weight_decay)
lr_scheduler_M4 = torch.optim.lr_scheduler.MultiStepLR(optimizer_M4, gamma=config_4.gamma, milestones=config_4.milestones, verbose=True)


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
# 5.22 - FPN model
# pdb.set_trace()


#%% AVS-specific functions, methods, and parameters
from AVS_functions import location_bounds, extract_and_resize_glimpses_for_batch, crop_five

lower, upper = location_bounds(glimpse_w=config_3.glimpse_size_init[0], input_w=config_3.full_res_img_size[0]) #TODO: make sure this works on both x and y coordinates


#%% Print experiment name
if config_3_valid.landmark_shuffle:
    exp_name = 'landmark_shuffle'
    print("Current experiment: landmark_shuffle, num_distortions:{}, box_size:{}\n".format(config_3_valid.num_distortion, config_3_valid.box_size))
elif config_3_valid.black_box:
    exp_name = 'black_box'
    print("Current experiment: black_box, num_boxes:{}, box_size:{}\n".format(config_3_valid.num_box, config_3_valid.box_size))
elif config_3_valid.gaussian_blur:
    exp_name = 'gaussian_blur'
    print("Current experiment: gaussian_blur, num_boxes:{}, box_size:{}\n".format(config_3_valid.num_box, config_3_valid.box_size))
elif config_3_valid.puzzle_solving:
    exp_name = 'puzzle_solving'
    print("Current experiment: puzzle_solving, num_permute:{}, box_size:{}\n".format(config_3_valid.num_permute, config_3_valid.box_size))
else:
    exp_name = 'correct'
    print("Current experiment: correct images")


#%% Define result save path
save_path = './LSTM_test_results/{}/num_distortion_{}_num_box_{}_box_size_{}_num_permute_{}'.format(exp_name, config_3_valid.num_distortion, config_3_valid.num_box, config_3_valid.box_size, config_3_valid.num_permute)
if not os.path.exists(save_path): os.makedirs(save_path)
print("Saving results to: {}".format(save_path))


#%% Validation
# with torch.no_grad():
#     model_4.eval()
#     dist_correct_imgs, dist_corrupted_imgs = [], []
#     # val_loss_lstm = 0.0
#     print("Begin validation.")
#     for i, (index, images, targets) in enumerate(valid_loader): # target_type=['attr', 'bbox', 'pred_init_glimpse_locs', 'landmarks', 'masks']
#         # 2.15.2023 - tao88, in order to account for error in compute_rl_loss
#         # if i == len(valid_loader) - 1:
#         #     continue 

#         translated_images, targets, bbox_targets, corrupt_labels = images.to(device), targets[0].float().to(device), targets[1].to(device), targets[-1].to(device) # translated images (128, 3, 256, 256); targets (128, 40); bbox_targets (128, 4); corrupt_labels (128), 0 for correct, 1 for corrnano01upted
#         # masks = torch.round(masks * 7).long() # preprocess the aligned masks

#         # 3.12.2023 - tao88: visualize batch of images
#         # save_path = "/home/nano01/a/tao88/4.17_puzzle_solving/corrupted_val"
#         # if not os.path.exists(save_path): os.makedirs(save_path)
#         # if i == 0:
#         #     for img_id in range(16):
#         #         if config_3_valid.puzzle_solving:
#         #             for perm_id in range(config_3_valid.num_permute+1):
#         #                 save_image(translated_images[img_id][perm_id], os.path.join(save_path, 'img_{}_perm_{}.png'.format(img_id, perm_id)))
#         #         else:
#         #             save_image(translated_images[img_id], os.path.join(save_path, 'img_{}.png'.format(img_id)))

#         batch_patches = crop_five(translated_images, 
#                                   left_coords=[64,128,64,128,96], 
#                                   top_coords=[64,64,128,128,96], 
#                                   widths=[64,64,64,64,64], 
#                                   heights=[64,64,64,64,64],              
#                                   resized_height=256, 
#                                   resized_width=256) # (128, 5, 3, 256, 256)

#         # pass batch of each patch to the model_FPN to get predicted masks
#         for patch_id in range(batch_patches.shape[1]):
#             patches = batch_patches[:, patch_id, :, :, :] # (128, 3, 256, 256)
#             masks_predicted = model_FPN(patches) # (128, 7, 64, 64)
#             lbl_predicted = masks_predicted.topk(1, dim=1)[1] # (128, 1, 64, 64), dtype=long
#             if patch_id == 0:
#                 batch_masks = lbl_predicted
#             else:
#                 batch_masks = torch.cat((batch_masks, lbl_predicted), dim=1)

#         # 3.13.2023 - visualize the patches and patch predictions (done)
#             # if i == 0:
#             #     # patches
#             #     for img_id in range(16):
#             #         save_image(patches[img_id], os.path.join(save_path, 'img_{}_patch_{}.png'.format(img_id, patch_id)))
#             #         # predicted patch masks from model_FPN
#             #         fig = plt.figure(figsize=(1,1))
#             #         ax = plt.Axes(fig, [0., 0., 1., 1.])
#             #         ax.set_axis_off()
#             #         fig.add_axes(ax)
#             #         ax.imshow(lbl_predicted[img_id].squeeze(0).cpu().numpy(),
#             #                     interpolation='nearest',
#             #                     cmap='Paired',
#             #                     vmin=0,
#             #                     vmax=6)
#             #         plt.savefig(os.path.join(save_path, 'img_{}_patch_{}_mask.png'.format(img_id, patch_id)))
#             #         plt.close(fig)
        
#         # 3.23.2023 - tao88: batch masks now has shape (128, 5, 64, 64), process the masks into semantics
#         num_pixels = batch_masks.shape[2] * lbl_predicted.shape[3]
#         batch_masks_flattened = batch_masks.flatten(start_dim=2, end_dim=3) # (128, 5, 4096)
#         mask_semantics = torch.nn.functional.one_hot(batch_masks_flattened, num_classes=7).sum(dim=2) # (128, 5, 7)
#         mask_semantics_ratio = (mask_semantics.float()) / num_pixels # (128, 5, 7), normalized

#         # pass batch_masks_flattened to LSTM, sequence length is 5
#         h0, c0 = model_4._init_hidden(translated_images.shape[0])
#         output_lstm, (hn, cn) = model_4([batch_masks_flattened.float(), mask_semantics_ratio], (h0, c0)) # (128, 5, 7)

#         # record dist curves for each image in dist_batch_imgs
#         dist_batch_imgs = torch.norm((output_lstm[:, :-1, :7] - mask_semantics_ratio[:, 1:, :]), dim=2, keepdim=False) # (128, 4)
#         dist_batch_imgs += torch.norm((output_lstm[:, 1:, 7:] - mask_semantics_ratio[:, :-1, :]), dim=2, keepdim=False) # (128, 4)
#         # concat to the dist array for all val images
#         if i == 0:
#             dist_correct_imgs = dist_batch_imgs[corrupt_labels==0]
#             dist_corrupted_imgs = dist_batch_imgs[corrupt_labels==1]
#         else:
#             dist_correct_imgs = torch.cat((dist_correct_imgs, dist_batch_imgs[corrupt_labels==0]), dim=0)
#             dist_corrupted_imgs = torch.cat((dist_corrupted_imgs, dist_batch_imgs[corrupt_labels==1]), dim=0)

#     # Use torch.hist for picking threshold and computing validation accuracy
#     dist_correct_imgs, dist_corrupted_imgs = np.array(dist_correct_imgs.cpu()), np.array(dist_corrupted_imgs.cpu()) # both (#, 4)
#     dist_correct_mean, dist_corrupted_mean = dist_correct_imgs.mean(1), dist_corrupted_imgs.mean(1)
#     hist_correct, bin_edges_correct = np.histogram(dist_correct_mean, bins=np.arange(0, 2.55, 0.05))
#     hist_corrupted, bin_edges_corrupted = np.histogram(dist_corrupted_mean, bins=np.arange(0, 2.55, 0.05))

#     cutoff_index = np.where(hist_corrupted>hist_correct)[0][0]
#     threshold = bin_edges_correct[cutoff_index] # first occurence
#     print("Number of val samples taken into account in hist: correct: {}, corrupted: {}".format(hist_correct.sum(), hist_corrupted.sum()))
#     num_fn, num_fp = hist_correct[cutoff_index:].sum(), hist_corrupted[:cutoff_index].sum()
#     num_tn, num_tp = hist_corrupted[cutoff_index:].sum(), hist_correct[:cutoff_index].sum()
#     val_acc = (num_tn + num_tp) / (num_fn + num_fp + num_tn + num_tp) * 100
#     print("To distinguish correct from corrupted samples, the threshold of mean distance between patch semantics and its prediction is %.2f" % (threshold))
#     print("By using the previous threshold, the validation accuracy is %.3f%%" % (val_acc))

#     # Use plt for visualization
#     plt.figure()
#     # plt.hist(dist_correct_imgs.mean(1), range=(0, 0.8), bins=16, alpha=0.5, color='b', label='correct')
#     # plt.hist(dist_corrupted_imgs.mean(1), range=(0, 0.8), bins=16, alpha=0.5, color='r', label='corrupted')
#     plt.hist(dist_correct_mean, bins=np.arange(0, 1.55, 0.05), alpha=0.7, color='b', label='correct')
#     plt.hist(dist_corrupted_mean, bins=np.arange(0, 1.55, 0.05), alpha=0.7, color='r', label='corrupted')
#     plt.xticks(np.arange(0, 1.6, 0.1))
#     plt.yticks(np.arange(0, 2100, 100))
#     plt.title('LSTM Prediction Results')
#     plt.xlabel('Average dist. over 5 patches')
#     plt.ylabel('Frequency')
#     plt.legend()
#     plt.savefig(os.path.join(save_path, 'hist_distavg_on_val.png'))
#     plt.close()
#     print("Histogram on validation plotted and saved")
#     print("Finished validation.\n")


#%% Testing
sum_masks_all_batches = torch.randn(5, 64, 64)
sum_semantics_all_batches = torch.randn(5, 7)
sum_pred_semantics_forward_all_batches = torch.randn(4, 7)
sum_pred_semantics_backward_all_batches = torch.randn(4, 7)

with torch.no_grad():
    model_4.eval()
    dist_correct_imgs, dist_corrupted_imgs = [], []
    if config_3_valid.puzzle_solving:
        dist_batch_allperms_mean = []
    # test_loss_lstm = 0.0
    print("Begin testing.")
    # save_path = "/home/nano01/a/tao88/4.26_puzzle_solving_ps_{}_num_permute_{}/corrupted_test".format(config_3_valid.box_size, config_3_valid.num_permute)
    save_path = "/home/nano01/a/tao88/5.20/celeba/correct"
    if not os.path.exists(save_path): os.makedirs(save_path)

    for i, (index, images, targets) in enumerate(test_loader): 
        # if i > 10:
        #     pdb.set_trace()
        #     continue
        translated_images, targets, bbox_targets, corrupt_labels = images.to(device), targets[0].float().to(device), targets[1].to(device), targets[-1].to(device) 

        # visualize correct and corrupted test images
        if i == 0:
            for img_id in range(16):
                if config_3_valid.puzzle_solving:
                    for perm_id in range(config_3_valid.num_permute+1):
                        save_image(translated_images[img_id][perm_id], os.path.join(save_path, 'img_{}_perm_{}.png'.format(img_id, perm_id)))
                else:
                    save_image(translated_images[img_id], os.path.join(save_path, 'img_{}.png'.format(img_id)))
        # pdb.set_trace() # 4.25.2023 - tao88: verify puzzle solving

        batch_patches = crop_five(translated_images, 
                                  left_coords=[64,128,64,128,96], 
                                  top_coords=[64,64,128,128,96], 
                                  widths=[64,64,64,64,64], 
                                  heights=[64,64,64,64,64],              
                                  resized_height=256, 
                                  resized_width=256) # (128, 5, 3, 256, 256) or (128, 5, 4, 3, 256, 256)

        if config_3_valid.puzzle_solving:
            batch_patches = torch.permute(batch_patches, (0, 2, 1, 3, 4, 5)) # (128, num_permute, 5, 3, 256, 256)

        # pass batch of each patch to the model_FPN to get predicted masks
        if not config_3_valid.puzzle_solving:
            for patch_id in range(batch_patches.shape[1]):
                patches = batch_patches[:, patch_id, :, :, :] # (128, 3, 256, 256)
                masks_predicted = model_FPN(patches) # (128, 7, 64, 64)
                lbl_predicted = masks_predicted.topk(1, dim=1)[1] # (128, 1, 64, 64), dtype=long
                if patch_id == 0:
                    batch_masks = lbl_predicted
                else:
                    batch_masks = torch.cat((batch_masks, lbl_predicted), dim=1) # (128, 5, 64, 64)
                # 3.13.2023 - visualize the patches and patch predictions (done)
                # if i == 0:
                # # patches
                #     for img_id in range(16):
                #         save_image(patches[img_id], os.path.join(save_path, 'img_{}_patch_{}.png'.format(img_id, patch_id)))
                #         # predicted patch masks from model_FPN
                #         fig = plt.figure(figsize=(1,1))
                #         ax = plt.Axes(fig, [0., 0., 1., 1.])
                #         ax.set_axis_off()
                #         fig.add_axes(ax)
                #         ax.imshow(lbl_predicted[img_id].squeeze(0).cpu().numpy(),
                #                 interpolation='nearest',
                #                 cmap=viridis,
                #                 vmin=0,
                #                 vmax=7)
                #         plt.savefig(os.path.join(save_path, 'img_{}_patch_{}_mask.png'.format(img_id, patch_id)))
                #         plt.close(fig)

            if i == 0:
                sum_masks_all_batches = batch_masks.sum(dim=0) # (5, 64, 64)
            else:
                sum_masks_all_batches += batch_masks.sum(dim=0)
            # 5.20 - collect average predicted masks over 10 batches
            # pdb.set_trace()
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
            # visualize
            # for img_id in range(2):
            #     for perm_id in range(config_3_valid.num_permute + 1):
            #         for patch_id in range(5):
            #             save_image(batch_patches[img_id][perm_id][patch_id], os.path.join(save_path, 'img_{}_perm_{}_patch_{}.png'.format(img_id, perm_id, patch_id)))
            #             fig = plt.figure()
            #             ax = plt.Axes(fig, [0., 0., 1., 1.])
            #             ax.set_axis_off()
            #             fig.add_axes(ax)
            #             ax.imshow(batch_masks[img_id][perm_id][patch_id].squeeze(0).cpu().numpy(),
            #                         interpolation='nearest',
            #                         cmap='Paired',
            #                         vmin=0,
            #                         vmax=7)
            #             plt.savefig(os.path.join(save_path, 'img_{}_perm_{}_patch_{}_mask.png'.format(img_id, perm_id, patch_id)))
            #             plt.close(fig)

        if not config_3_valid.puzzle_solving:
            # 3.23.2023 - tao88: batch masks now has shape (128, #_perms, 64, 64), process the masks into semantics
            num_pixels = batch_masks.shape[2] * batch_masks.shape[3]
            batch_masks_flattened = batch_masks.flatten(start_dim=2, end_dim=3) # (128, 5, 4096)
            mask_semantics = nnF.one_hot(batch_masks_flattened, num_classes=7).sum(dim=2) # (128, 5, 7)
            mask_semantics_ratio = (mask_semantics.float()) / num_pixels # (128, 5, 7), normalized
            # 5.20 - collect sum of gt semantics
            if i == 0:
                sum_semantics_all_batches = mask_semantics_ratio.sum(dim=0)
            else:
                sum_semantics_all_batches += mask_semantics_ratio.sum(dim=0)


            # pass batch_masks_flattened to LSTM, sequence length is 5
            h0, c0 = model_4._init_hidden(translated_images.shape[0])
            output_lstm, (hn, cn) = model_4([batch_masks_flattened.float(), mask_semantics_ratio], (h0, c0)) # (128, 5, 7)

            # 5.20 - collect pred semantics
            if i == 0:
                sum_pred_semantics_forward_all_batches = output_lstm[:, :-1, :7].sum(dim=0)
                sum_pred_semantics_backward_all_batches = output_lstm[:, 1:, 7:].sum(dim=0)
            else:
                sum_pred_semantics_forward_all_batches += output_lstm[:, :-1, :7].sum(dim=0)
                sum_pred_semantics_backward_all_batches += output_lstm[:, 1:, 7:].sum(dim=0)

            # record dist curves for each image in dist_batch_imgs
            dist_batch_imgs_for = torch.norm((output_lstm[:, :-1, :7] - mask_semantics_ratio[:, 1:, :]), dim=2, keepdim=False) # (128, 4)
            dist_batch_imgs_back = torch.norm((output_lstm[:, 1:, 7:] - mask_semantics_ratio[:, :-1, :]), dim=2, keepdim=False) # (128, 4)

            dist_batch_imgs = dist_batch_imgs_for + dist_batch_imgs_back
            dist_batch_imgs_both = torch.zeros(output_lstm.shape[0], output_lstm.shape[1]).to(device)
            dist_batch_imgs_both[:, 1:] = dist_batch_imgs_for
            dist_batch_imgs_both[:, :-1] += dist_batch_imgs_back # (128, 5)
            # concat to the dist array for all val images

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

            for perm_id in range(config_3_valid.num_permute + 1):
                h0, c0 = model_4._init_hidden(translated_images.shape[0])
                output_lstm, (hn, cn) = model_4([batch_masks_flattened[:, perm_id, :, :].float(), mask_semantics_ratio[:, perm_id, :, :]], (h0, c0))
                dist_batch_imgs = torch.norm((output_lstm[:, :-1, :7] - mask_semantics_ratio[:, perm_id, 1:, :]), dim=2, keepdim=False)
                dist_batch_imgs += torch.norm((output_lstm[:, 1:, 7:] - mask_semantics_ratio[:, perm_id, :-1, :]), dim=2, keepdim=False) # (128, num_patch-1)
                if perm_id == 0: # fist permutation is correct
                    dist_allperms_mean = dist_batch_imgs.mean(1).unsqueeze(1)
                else:
                    dist_allperms_mean = torch.cat((dist_allperms_mean, dist_batch_imgs.mean(1).unsqueeze(1)), dim=1) #(bs, num_permute+1)
            if i == 0:
                dist_batch_allperms_mean = dist_allperms_mean
            else:
                dist_batch_allperms_mean = torch.cat((dist_batch_allperms_mean, dist_allperms_mean), dim=0) # (num_samples, num_permute+1)

        # pdb.set_trace()
        if (i+1) % 50 == 0:
            print("{}/{}".format(i+1, len(test_loader)))

    # End of Test Epoch
    # if not config_3_valid.puzzle_solving:
    #     # for paper:
    #     dist_correct_imgs_for, dist_corrupted_imgs_for = np.array(dist_correct_imgs_for.cpu()), np.array(dist_corrupted_imgs_for.cpu()) # both (#, 4)
    #     dist_correct_imgs_back, dist_corrupted_imgs_back = np.array(dist_correct_imgs_back.cpu()), np.array(dist_corrupted_imgs_back.cpu()) # both (#, 4)
    #     dist_correct_imgs_both, dist_corrupted_imgs_both = np.array(dist_correct_imgs_both.cpu()), np.array(dist_corrupted_imgs_both.cpu()) # both (#, 5)
    #     # print data
    #     print("Printing forward_correct_mean:")
    #     for item in np.mean(dist_correct_imgs_for, axis=0): print(item)
    #     print("Printing forward_correct_25th:")
    #     for item in np.percentile(dist_correct_imgs_for, 25, axis=0): print(item)
    #     print("Printing forward_correct_75th:")
    #     for item in np.percentile(dist_correct_imgs_for, 75, axis=0): print(item)
    #     print("Printing forward_corrupted_mean:")
    #     for item in np.mean(dist_corrupted_imgs_for, axis=0): print(item)
    #     print("Printing forward_corrupted_25th:")
    #     for item in np.percentile(dist_corrupted_imgs_for, 25, axis=0): print(item)
    #     print("Printing forward_corrupted_75th:")
    #     for item in np.percentile(dist_corrupted_imgs_for, 75, axis=0): print(item)
    #     print("Printing backward_correct_mean:")
    #     for item in np.mean(dist_correct_imgs_back, axis=0): print(item)
    #     print("Printing backward_correct_25th:")
    #     for item in np.percentile(dist_correct_imgs_back, 25, axis=0): print(item)
    #     print("Printing backward_correct_75th:")
    #     for item in np.percentile(dist_correct_imgs_back, 75, axis=0): print(item)
    #     print("Printing backward_corrupted_mean:")
    #     for item in np.mean(dist_corrupted_imgs_back, axis=0): print(item)
    #     print("Printing backward_corrupted_25th:")
    #     for item in np.percentile(dist_corrupted_imgs_back, 25, axis=0): print(item)
    #     print("Printing backward_corrupted_75th:")
    #     for item in np.percentile(dist_corrupted_imgs_back, 75, axis=0): print(item)
    #     print("Printing both_correct_mean:")
    #     for item in np.mean(dist_correct_imgs_both, axis=0): print(item)
    #     print("Printing both_correct_25th:")
    #     for item in np.percentile(dist_correct_imgs_both, 25, axis=0): print(item)
    #     print("Printing both_correct_75th:")
    #     for item in np.percentile(dist_correct_imgs_both, 75, axis=0): print(item)
    #     print("Printing both_corrupted_mean:")
    #     for item in np.mean(dist_corrupted_imgs_both, axis=0): print(item)
    #     print("Printing both_corrupted_25th:")
    #     for item in np.percentile(dist_corrupted_imgs_both, 25, axis=0): print(item)
    #     print("Printing both_corrupted_75th:")
    #     for item in np.percentile(dist_corrupted_imgs_both, 75, axis=0): print(item)
    #     # pdb.set_trace()
    #     # 5.4.2023 - tao88: make plots of distribution of distance over glimpes, for correct and corrupt images
    #     plt.figure()
    #     plt.plot(np.arange(2, 6), np.mean(dist_correct_imgs_for, axis=0), 'b')
    #     plt.plot(np.arange(2, 6), np.max(dist_correct_imgs_for, axis=0), 'b')
    #     plt.plot(np.arange(2, 6), np.min(dist_correct_imgs_for, axis=0), 'b')
    #     plt.plot(np.arange(2, 6), np.mean(dist_corrupted_imgs_for, axis=0), 'r')
    #     plt.plot(np.arange(2, 6), np.max(dist_corrupted_imgs_for, axis=0), 'r')
    #     plt.plot(np.arange(2, 6), np.min(dist_corrupted_imgs_for, axis=0), 'r')
    #     plt.ylim([0, 2.5])
    #     plt.savefig(os.path.join(save_path, 'spread_together_forward.png'))
    #     plt.close()

    #     plt.figure()
    #     plt.plot(np.arange(2, 6), np.mean(dist_correct_imgs_back, axis=0), 'b')
    #     plt.plot(np.arange(2, 6), np.max(dist_correct_imgs_back, axis=0), 'b')
    #     plt.plot(np.arange(2, 6), np.min(dist_correct_imgs_back, axis=0), 'b')
    #     plt.plot(np.arange(2, 6), np.mean(dist_corrupted_imgs_back, axis=0), 'r')
    #     plt.plot(np.arange(2, 6), np.max(dist_corrupted_imgs_back, axis=0), 'r')
    #     plt.plot(np.arange(2, 6), np.min(dist_corrupted_imgs_back, axis=0), 'r')
    #     plt.ylim([0, 2.5])
    #     plt.savefig(os.path.join(save_path, 'spread_together_backward.png'))
    #     plt.close()

    #     plt.figure()
    #     plt.plot(np.arange(2, 6), np.mean(dist_correct_imgs_for, axis=0) + np.mean(dist_correct_imgs_for, axis=0), 'b')
    #     plt.plot(np.arange(2, 6), np.mean(dist_corrupted_imgs_for, axis=0) + np.mean(dist_corrupted_imgs_back, axis=0), 'r')
    #     plt.ylim([0, 2.5])
    #     plt.savefig(os.path.join(save_path, 'spread_together_both.png'))
    #     plt.close()
    #     # pdb.set_trace()
    #     # np.percentile(dist_correct_imgs_back, 75, axis=0)


    #     # Determine threshold and make histogram
    #     dist_correct_imgs, dist_corrupted_imgs = np.array(dist_correct_imgs.cpu()), np.array(dist_corrupted_imgs.cpu())
    #     dist_correct_mean, dist_corrupted_mean = dist_correct_imgs.sum(1), dist_corrupted_imgs.sum(1)
    #     hist_correct, bin_edges_correct = np.histogram(dist_correct_mean, bins=np.arange(0, 6.51, 0.05))
    #     hist_corrupted, bin_edges_corrupted = np.histogram(dist_corrupted_mean, bins=np.arange(0, 6.51, 0.05))
    #     print("Number of test samples taken into account in hist: correct: {}, corrupted: {}".format(hist_correct.sum(), hist_corrupted.sum()))
    #     print("Correct sample mean distance range: {} to {}".format(dist_correct_mean.min(), dist_correct_mean.max()))
    #     print("Corrupted sample mean distance range: {} to {}".format(dist_corrupted_mean.min(), dist_corrupted_mean.max()))

    #     # Determine threshold
    #     cutoff_index = np.where(hist_corrupted>hist_correct)[0][0]
    #     first_occ_threshold = bin_edges_correct[cutoff_index] # first occurence
    #     print("First occurence where there're more corrupted samples than correct samples: %.2f" % (first_occ_threshold))
    #     # num_fp, num_fn = hist_correct[cutoff_index:].sum(), hist_corrupted[:cutoff_index].sum()
    #     # num_tp, num_tn = hist_corrupted[cutoff_index:].sum(), hist_correct[:cutoff_index].sum()
    #     # test_acc = (num_tn + num_tp) / (num_fn + num_fp + num_tn + num_tp) * 100
    #     # print("Number of test samples taken into account in hist: correct: {}, corrupted: {}".format(hist_correct.sum(), hist_corrupted.sum()))
    #     # print("To distinguish correct from corrupted samples, the threshold of mean distance between patch semantics and its prediction is %.2f" % (threshold))
    #     # print("By using the previous threshold, the test accuracy is %.3f%%" % (test_acc))

    #     for cutoff_index in range(0, bin_edges_correct.shape[0]):
    #         threshold = bin_edges_correct[cutoff_index] # first occurence
    #         num_fp, num_fn = hist_correct[cutoff_index:].sum(), hist_corrupted[:cutoff_index].sum()
    #         num_tp, num_tn = hist_corrupted[cutoff_index:].sum(), hist_correct[:cutoff_index].sum()
    #         test_acc = (num_tn + num_tp) / (num_fn + num_fp + num_tn + num_tp) * 100
    #         print("Threshold: %.2f" %(threshold), "By using that threshold, the test accuracy is %.3f%%, fn %d, fp %d, tn %d, tp %d" % (test_acc, num_fn, num_fp, num_tn, num_tp))
    #     # pdb.set_trace()
    #     # Use plt for visualization
    #     plt.figure()
    #     plt.hist(dist_correct_mean, bins=np.arange(0, 6.55, 0.05), alpha=0.7, color='b', label='correct')
    #     plt.hist(dist_corrupted_mean, bins=np.arange(0, 6.55, 0.05), alpha=0.7, color='r', label='corrupted')
    #     plt.xticks(np.arange(0, 6.5, 0.5))
    #     plt.yticks(np.arange(0, 700, 100))
    #     # plt.title('LSTM Prediction Results')
    #     plt.xlabel('Total residual', fontsize=16)
    #     plt.ylabel('Count', fontsize=16)
    #     plt.legend(["correct", "corrupted"], fontsize="18", loc ="upper right")
    #     plt.savefig(os.path.join(save_path, 'hist.png'))
    #     plt.close()
    #     print("Histogram on test plotted and saved")
    #     print("Finished testing.\n")
    #     pdb.set_trace()
    # else:
    #     dist_batch_allperms_mean = np.array(dist_batch_allperms_mean.cpu()) # (num_samples, num_permute+1)
    #     # if entry 0 < all other entries, correct prediction!
    #     num_correct = (dist_batch_allperms_mean[:,0]<dist_batch_allperms_mean[:,1:].min(axis=1)).sum()
    #     num_samples = dist_batch_allperms_mean.shape[0]
    #     test_acc = (num_correct / num_samples) *100
    #     print("Number of samples when the correct puzzle is picked: %d" % (num_correct))
    #     print("Total count of samples in the testset: %d" % (num_samples))
    #     print("The test accuracy is %.3f%%" % (test_acc))

    #     # Histogram for visualizing distribution of correct vs. fake puzzles
    #     plt.figure()
    #     plt.hist(dist_batch_allperms_mean[:,0], bins=np.arange(0, 1.6, 0.05), alpha=0.7, color='b', label='correct')
    #     plt.hist(dist_batch_allperms_mean[:,1:].flatten(), bins=np.arange(0, 1.6, 0.05), alpha=0.7, color='r', label='corrupted')
    #     plt.xticks(np.arange(0, 1.6, 0.1))
    #     plt.yticks(np.arange(0, 2100, 100))
    #     plt.title('LSTM Prediction Results')
    #     plt.xlabel('Average dist. over 5 patches')
    #     plt.ylabel('Frequency')
    #     plt.legend()
    #     plt.savefig(os.path.join(save_path, 'hist_puzzle_solving.png'))
    #     plt.close()
    #     print("Histogram on test plotted and saved")
    #     print("Finished testing.\n")

# pdb.set_trace()
# 5.20 - sum_masks_all_batches has shape (5, 64, 64)
avg_masks_all_batches = torch.round(sum_masks_all_batches / 19962)
avg_semantics_all_batches = sum_semantics_all_batches / 19962
avg_pred_semantics_forward_all_batches = sum_pred_semantics_forward_all_batches / 19962
avg_pred_semantics_backward_all_batches = sum_pred_semantics_backward_all_batches / 19962
pdb.set_trace()
for patch_id in range(5):
    fig = plt.figure(figsize=(1, 1))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    fig.add_axes(ax)
    psm = ax.imshow(avg_masks_all_batches[patch_id].squeeze().cpu().numpy(),
                interpolation='nearest',
                cmap=viridis,
                vmin=0,
                vmax=7)
    cbar = fig.colorbar(psm, ticks=[0, 1, 2, 3, 4, 5, 6])
    # cbar.set_ticks([0, 1, 2, 3, 4, 5, 6])
    plt.savefig(os.path.join(save_path, 'avgpatch_{}_mask.png'.format(patch_id)))
    plt.close(fig)
print(avg_semantics_all_batches)
print(avg_pred_semantics_forward_all_batches)
print(avg_pred_semantics_backward_all_batches)
print(torch.norm((avg_pred_semantics_forward_all_batches - avg_semantics_all_batches[1:]), dim=1, keepdim=False))
print(torch.norm((avg_pred_semantics_backward_all_batches - avg_semantics_all_batches[:-1]), dim=1, keepdim=False))

pdb.set_trace()

#%% backup code to plot individual distance curves for every image
 # visualize the distance curves for individual images
        # for sample_id in range(16):
        #     if not config_3.landmark_shuffle:
        #         if not config_3.use_corrupted_testset:
        #             save_path = '/home/nano01/a/tao88/3.24_fivepatch_masks_and_semantics_bilstm_maskpred_20epochs/correct/img_{}'.format(sample_id)
        #         else:
        #             save_path = '/home/nano01/a/tao88/3.24_fivepatch_masks_and_semantics_bilstm_maskpred_20epochs/corrupted_testset/img_{}'.format(sample_id)
        #     else:
        #         save_path = '/home/nano01/a/tao88/3.24_fivepatch_masks_and_semantics_bilstm_maskpred_20epochs/corrupted/img_{}'.format(sample_id)
        #     plot_curve(np.arange(1, 5), dist_batch_imgs[sample_id], 'Dist btw. prediction and glimpse semantics', 'Glimpse iteration', 'L2 distance', os.path.join(save_path, 'plot_dist.png'))

        # 3.8.2023 - tao88: individual dist curves, blue for correct, red for corrupted
        # x, y = [np.arange(1, 5) for _ in range(translated_images.shape[0])], dist_batch_imgs
        # colors = ['b', 'r']
        # corruptness = ['correct', 'corrupted']
        # seen = set()
        # plt.figure()
        # for j in range(len(x)):
        #     corrupt_label = corrupt_labels[j].item()
        #     if corrupt_label not in seen:
        #         seen.add(corrupt_label)
        #         plt.plot(x[j], y[j], c=colors[corrupt_label], label=corruptness[corrupt_label])
        #     else:
        #         plt.plot(x[j], y[j], c=colors[corrupt_label])
        # plt.title('Dist btw. prediction and glimpse semantics')
        # plt.xlabel('Glimpse iteration')
        # plt.ylabel('L2 distance')
        # plt.legend()
        # plt.savefig('/home/nano01/a/tao88/3.24_fivepatch_masks_and_semantics_bilstm_maskpred_20epochs/curves_all.png')
        # plt.close()

        # 3.9.2023 - tao88: plot an average distance curve for all correct images, and another for all corrupted
        # dist_correct_imgs, dist_corrupted_imgs = [], []
        # for j in range(translated_images.shape[0]):
        #     corrupt_label = corrupt_labels[j].item()
        #     if corrupt_label == 0:
        #         dist_correct_imgs.append(dist_batch_imgs[j])
        #     else:
        #         dist_corrupted_imgs.append(dist_batch_imgs[j])
        # dist_correct_avg = np.average(np.array(dist_correct_imgs), axis=0) # compute average curve for all correct images
        # dist_corrupted_avg = np.average(np.array(dist_corrupted_imgs), axis=0) # compute average curve for all corrupted images
        # plt.figure()
        # plt.plot(np.arange(1, 5), dist_correct_avg, c='b', label=corruptness[0], linewidth=4)
        # plt.plot(np.arange(1, 5), dist_corrupted_avg, c='r', label=corruptness[1], linewidth=4)
        # plt.title('Averaged dist btw. prediction and glimpse semantics')
        # plt.xlabel('Glimpse iteration')
        # plt.ylabel('L2 distance')
        # plt.legend()
        # plt.savefig('/home/nano01/a/tao88/3.24_fivepatch_masks_and_semantics_bilstm_maskpred_20epochs/curves_together_avg.png')    
        # plt.close()

        # 3.23.2023 - tao88: from dist_correct_imgs and dist_corrupt_imgs, plot individual running average curves
        # dist_correct_imgs, dist_corrupted_imgs = np.array(dist_correct_imgs), np.array(dist_corrupted_imgs)
        # plt.figure()
        # for j in range(dist_correct_imgs.shape[0]):
        #     correct_runavg = np.divide(np.cumsum(dist_correct_imgs[j]), np.arange(1, 5))
        #     if j == 0: 
        #         plt.plot(np.arange(1, 5), correct_runavg, c=colors[0], label=corruptness[0])
        #     else:
        #         plt.plot(np.arange(1, 5), correct_runavg, c=colors[0])
        # for j in range(dist_corrupted_imgs.shape[0]):
        #     corrupted_runavg = np.divide(np.cumsum(dist_corrupted_imgs[j]), np.arange(1, 5))
        #     if j == 0: 
        #         plt.plot(np.arange(1, 5), corrupted_runavg, c=colors[1], label=corruptness[1])
        #     else:
        #         plt.plot(np.arange(1, 5), corrupted_runavg, c=colors[1])
        # plt.title('Runavg dist btw. prediction and glimpse semantics')
        # plt.xlabel('Glimpse iteration')
        # plt.ylabel('L2 distance')
        # plt.legend()
        # plt.savefig('/home/nano01/a/tao88/3.24_fivepatch_masks_and_semantics_bilstm_maskpred_20epochs/curves_all_runavg.png')    
        # plt.close()
