import torch
import torch.nn.functional as nnF
from torchvision import transforms
import torchvision.transforms.functional as transF
import json
import numpy as np
import random
import math
# debug
import pdb

#%% Miscellaneous
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


#%% Transforms
def transform_label(resize, totensor, normalize, centercrop):
    options = []
    if centercrop:
        options.append(transforms.CenterCrop(160))
    if resize:
        options.append(transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)) # 64x64
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0, 0, 0), (0, 0, 0)))
    transform = transforms.Compose(options)
    return transform


#%% Corruptions
# One-time permutation (batchwise)
class ShufflePatches(object):
  def __init__(self, patch_size):
    self.ps = patch_size

  def __call__(self, X):
    # divide the batch of images into non-overlapping patches
    u = nnF.unfold(X, kernel_size=self.ps, stride=self.ps, padding=0)
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = nnF.fold(pu, X.shape[-2:], kernel_size=self.ps, stride=self.ps, padding=0)
    return f
  

# Shuffling with set num_distortion (imagewise)
def corrupt_img_landmark_shuffle(tensors, num_distortion, patch_size):
    result = []
    for it, X in enumerate(tensors):
        if len(X.shape) == 3:
            X = X[None,...]
        patches = nnF.unfold(X, kernel_size=patch_size, stride=patch_size, padding=0) # only 4D inputs supported
        # permute the patches (according to num_distortion)
        num_patches = patches.shape[-1]
        if it == 0:
            chosen_indices = random.sample(range(num_patches), num_distortion) # sample without replacement, fix it for all inputs
            orig_order = torch.LongTensor(range(num_patches))
            permuted_order = torch.LongTensor(range(num_patches))
            for action in range(num_distortion):
                permuted_order[chosen_indices[action]] = orig_order[chosen_indices[(action+1)%num_distortion]]
        # concat the patches
        patches_concatted = torch.cat([b_[:, permuted_order][None,...] for b_ in patches], dim=0)
        # fold back
        X = nnF.fold(patches_concatted, X.shape[-2:], kernel_size=patch_size, stride=patch_size, padding=0)
        if len(X.shape) == 4:
            X = X.squeeze(0)
        result.append(X)
    return result


# Random permute shuffling once
def corrupt_img_permute(tensors, patch_size):
    result = []
    for it, X in enumerate(tensors):
        if len(X.shape) == 3:
            X = X[None,...]
        patches = nnF.unfold(X, kernel_size=patch_size, stride=patch_size, padding=0) # only 4D inputs supported
        # permute the patches
        num_patches = patches.shape[-1]
        if it == 0:
            permuted_order = torch.randperm(num_patches)
        # concat the patches
        patches_concatted = torch.cat([b_[:, permuted_order][None,...] for b_ in patches], dim=0)
        # fold back
        X = nnF.fold(patches_concatted, X.shape[-2:], kernel_size=patch_size, stride=patch_size, padding=0)
        if len(X.shape) == 4:
            X = X.squeeze(0)
        result.append(X)
    return result


#%% Break image into patches 
def unfold(X, patch_size):
    bs, num_ch = X.shape[0], X.shape[1]
    # divide
    patches = nnF.unfold(X, kernel_size=patch_size, stride=patch_size, padding=0) # (bs, num_ch*ps*ps, num_patches)
    num_patches = patches.shape[-1]
    # permute the last dimension to the right order
    if patch_size == 128:
        perm = torch.LongTensor([0, 1, 3, 2]) # apply to ps=128 and 64, respectively
    elif patch_size == 64:
        perm = torch.LongTensor([0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12])
    patches = patches[:, :, perm]
    # make x the right shape
    patches = torch.permute(patches, (0, 2, 1))
    patches = torch.reshape(patches, (bs, num_patches, num_ch, patch_size, patch_size))
    return patches

