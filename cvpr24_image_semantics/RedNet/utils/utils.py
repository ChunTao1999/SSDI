import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnF
from torchvision import transforms
from torchvision.utils import save_image
import os
import random
# Debug
import pdb

med_frq = [0.382900, 0.452448, 0.637584, 0.377464, 0.585595,
           0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
           2.589096, 0.980794, 0.920340, 0.667984, 1.172291,
           0.862240, 0.921714, 2.154782, 1.187832, 1.178115,
           1.848545, 1.428922, 2.849658, 0.771605, 1.656668,
           4.483506, 2.209922, 1.120280, 2.790182, 0.706519,
           3.994768, 2.220004, 0.972934, 1.481525, 5.342475,
           0.750738, 4.040773]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

label_colours = [(0, 0, 0),
                 # 0=background
                 (148, 65, 137), (255, 116, 69), (86, 156, 137),
                 (202, 179, 158), (155, 99, 235), (161, 107, 108),
                 (133, 160, 103), (76, 152, 126), (84, 62, 35),
                 (44, 80, 130), (31, 184, 157), (101, 144, 77),
                 (23, 197, 62), (141, 168, 145), (142, 151, 136),
                 (115, 201, 77), (100, 216, 255), (57, 156, 36),
                 (88, 108, 129), (105, 129, 112), (42, 137, 126),
                 (155, 108, 249), (166, 148, 143), (81, 91, 87),
                 (100, 124, 51), (73, 131, 121), (157, 210, 220),
                 (134, 181, 60), (221, 223, 147), (123, 108, 131),
                 (161, 66, 179), (163, 221, 160), (31, 146, 98),
                 (99, 121, 30), (49, 89, 240), (116, 108, 9),
                 (161, 176, 169), (80, 29, 135), (177, 105, 197),
                 (139, 110, 246)]


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=med_frq):
        super(CrossEntropyLoss2d, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(),
                                           size_average=False, reduce=False)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            mask = targets > 0
            targets_m = targets.clone()
            targets_m[mask] -= 1
            loss_all = self.ce_loss(inputs, targets_m.long())
            losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss


def color_label(label):
    label = label.clone().cpu().data.numpy()
    colored_label = np.vectorize(lambda x: label_colours[int(x)])

    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()

    try:
        return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
    except ValueError:
        return torch.from_numpy(colored[np.newaxis, ...])


def print_log(global_step, epoch, local_count, count_inter, dataset_size, loss, time_inter):
    print('Step: {:>5} Train Epoch: {:>3} [{:>4}/{:>4} ({:3.1f}%)]    '
          'Loss: {:.6f} [{:.2f}s every {:>4} data]'.format(
        global_step, epoch, local_count, dataset_size,
        100. * local_count / dataset_size, loss.data, time_inter, count_inter))


def save_ckpt(ckpt_dir, model, optimizer, global_step, epoch, local_count, num_train):
    # usually this happens only on the start of a epoch
    epoch_float = epoch + (local_count / num_train)
    state = {
        'global_step': global_step,
        'epoch': epoch_float,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{:0.2f}.pth".format(epoch_float)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))


def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            if 'optimizer' in checkpoint.keys(): # 4.21.2023 - tao88
                optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        return step, epoch
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        os._exit(0)



#%% Corruption methods during test
# Shuffling with set num_distortion
def corrupt_img_landmark_shuffle(tensors, num_distortion, patch_size):
    result = []
    for it, X in enumerate(tensors):
        if len(X.shape) == 3:
            X = X[None,...]
        elif len(X.shape) == 2:
            X = X[None,None,...]
        patches = nnF.unfold(X, kernel_size=patch_size, stride=patch_size, padding=0) # only 4D inputs supported
        # permute the patches (according to num_distortion)
        num_patches = patches.shape[-1]
        # pdb.set_trace()
        if it == 0: # make sure img and depth/mask get same distortions
            chosen_indices = random.sample(range(num_patches), num_distortion) # sample without replacement, fix it for all inputs
            orig_order = torch.LongTensor(range(num_patches))
            permuted_order = torch.LongTensor(range(num_patches))
            for action in range(num_distortion):
                permuted_order[chosen_indices[action]] = orig_order[chosen_indices[(action+1)%num_distortion]]
        # concat the patches
        patches_concatted = torch.cat([b_[:, permuted_order][None,...] for b_ in patches], dim=0)
        # fold back
        X = nnF.fold(patches_concatted, X.shape[-2:], kernel_size=patch_size, stride=patch_size, padding=0)
        X = X.squeeze()
        result.append(X)
    return result


# Random permute with set num_permute copies
def corrupt_img_permute(tensors, num_permute, patch_size):
    result = []
    for it, X in enumerate(tensors):
        if len(X.shape) == 3:
            X = X[None,...]
        elif len(X.shape) == 2:
            X = X[None,None,...]
        patches = nnF.unfold(X, kernel_size=patch_size, stride=patch_size, padding=0) # only 4D inputs supported
        # permute the patches
        num_patches = patches.shape[-1]
        if it == 0:
            permuted_orders = []
            for perm_id in range(num_permute):
                permuted_order = torch.randperm(num_patches)
                permuted_orders.append(permuted_order)
        # concat the different perms
        res_tensor = X.clone()[None,...] # the correct version, (1, 3, 256, 256)
        for perm_id, order in enumerate(permuted_orders):
            patches_concatted = torch.cat([b_[:, order][None,...] for b_ in patches], dim=0)
            # fold back
            new_X = nnF.fold(patches_concatted, X.shape[-2:], kernel_size=patch_size, stride=patch_size, padding=0)
            # concat to res_tensor
            res_tensor = torch.cat((res_tensor, new_X[None,...]), dim=0)
        result.append(res_tensor.squeeze())
    return result


def corrupt_img_black_box(tensors, num_box, patch_size):
    result = []
    for it, X in enumerate(tensors):
        if len(X.shape) == 3:
            X = X[None,...]
        elif len(X.shape) == 2:
            X = X[None,None,...]
        # break the image and shadow tensors into patches according to ps
        patches = nnF.unfold(X, kernel_size=patch_size, stride=patch_size, padding=0) # only 4D inputs supported
        # make one or more patches into black box(es) according to num_box
        num_patches = patches.shape[-1]
        if it == 0: # fix the indices of corrupted patches for both rgb image and depth
            chosen_indices = random.sample(range(num_patches), num_box)
        for i, b_ in enumerate(patches):
            patches[i][:, chosen_indices] = 0
        # fold back, no need to permute the order
        res_tensor = nnF.fold(patches, X.shape[-2:], kernel_size=patch_size, stride=patch_size, padding=0)
        result.append(res_tensor.squeeze())
    return result


def corrupt_img_gaussian_blurring(tensors, num_box, patch_size, kernel_size=(11, 11), sigma=3):
    result = []
    blurrer = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    for it, X in enumerate(tensors):
        if len(X.shape) == 3:
            X = X[None,...]
        elif len(X.shape) == 2:
            X = X[None,None,...]
        # break the image and shadow tensors into patches according to ps
        patches = nnF.unfold(X, kernel_size=patch_size, stride=patch_size, padding=0) # only 4D inputs supported
        # make one or more patches into black box(es) according to num_box
        num_patches = patches.shape[-1]
        if it == 0: # fix the indices of corrupted patches for both rgb image and depth
            chosen_indices = random.sample(range(num_patches), num_box)
        for i, b_ in enumerate(patches):
            for index in chosen_indices:
                patches[i][:, index] = blurrer(patches[i][:, index].reshape(X.shape[1], patch_size, patch_size)).flatten()
        # fold back, no need to permute the order
        res_tensor = nnF.fold(patches, X.shape[-2:], kernel_size=patch_size, stride=patch_size, padding=0)
        result.append(res_tensor.squeeze())
    return result


#%% Break image into patches 
def unfold(X, patch_size):
    # X has shape (bs, num_ch, 640, 480)
    bs, num_ch = X.shape[0], X.shape[1]
    # divide
    patches = nnF.unfold(X, kernel_size=patch_size, stride=patch_size, padding=0) # (bs, num_ch*ps*ps, num_patches)
    num_patches = patches.shape[-1]
    # permute the last dimension to the right order
    if patch_size == 160: # num_patches = 12
        perm = torch.LongTensor([0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11]) # apply to ps=160
    elif patch_size == 80: # num_patches = 48
        perm = torch.LongTensor([0,1,2,3,4,5,6,7,15,14,13,12,11,10,9,8,16,17,18,19,20,21,22,23,31,30,29,28,27,26,25,24,32,33,34,35,36,37,38,39,47,46,45,44,43,42,41,40])
    patches = patches[:, :, perm]
    # make x the right shape
    patches = torch.permute(patches, (0, 2, 1))
    patches = torch.reshape(patches, (bs, num_patches, num_ch, patch_size, patch_size))
    return patches


#%% test functionality (verified)
# old_img = torch.rand(3, 480, 640)
# old_depth = torch.rand(1, 480, 640)
# new_img, new_depth = corrupt_img_gaussian_blurring(tensors=[old_img, old_depth],
#                                                    num_box=4,
#                                                    patch_size=160)
# save_path = "/home/nano01/a/tao88/4.30_gaussianblur"
# if not os.path.exists(save_path): os.makedirs(save_path)
# save_image(old_img, os.path.join(save_path, "old_img.png"))
# save_image(old_depth, os.path.join(save_path, "old_depth.png"))
# save_image(new_img, os.path.join(save_path, "new_img.png"))
# save_image(new_depth, os.path.join(save_path, "new_depth.png"))
# pdb.set_trace()


def calculate_iou(pred_mask, true_mask, num_classes):
    iou = torch.zeros(num_classes)
    for class_idx in range(num_classes):
        pred_class_mask = (pred_mask == class_idx).float()
        true_class_mask = (true_mask == class_idx).float()
        intersection = torch.sum(pred_class_mask * true_class_mask)
        union = torch.sum(pred_class_mask) + torch.sum(true_class_mask) - intersection
        # avoid division by zero
        iou[class_idx] = (intersection + 1e-6) / (union + 1e-6)
    return iou