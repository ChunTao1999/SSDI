import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnF
from torchvision import transforms
from torchvision.utils import save_image
import os
import random


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