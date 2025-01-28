import torch
import torch.nn.functional as nnF
from torchvision import transforms
import random
import pdb

class PermutePatches(object):
  def __init__(self, patch_size):
    self.ps = patch_size

  def __call__(self, x):
    # divide the batch of images into non-overlapping patches
    u = nnF.unfold(x, kernel_size=self.ps, stride=self.ps, padding=0)
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = nnF.fold(pu, x.shape[-2:], kernel_size=self.ps, stride=self.ps, padding=0)
    return f


def shuffle_patches(x, num_distortion, patch_size):
  # x is defaulted to be 4-d (bs, num_ch, h, w)
  patches = nnF.unfold(x, kernel_size=patch_size, stride=patch_size, padding=0)
  num_patches = patches.shape[-1]
  chosen_indices = random.sample(range(num_patches), num_distortion)
  orig_order = torch.LongTensor(range(num_patches))
  permuted_order = torch.LongTensor(range(num_patches))
  # set number of shuffles
  for action_id in range(num_distortion):
    permuted_order[chosen_indices[action_id]] = orig_order[chosen_indices[(action_id+1)%num_distortion]]
  patches_concatted = torch.cat([b_[:, permuted_order][None,...]for b_ in patches], dim=0)
  new_x = nnF.fold(patches_concatted, x.shape[-2:], kernel_size=patch_size, stride=patch_size, padding=0)
  return new_x


def black_patches(x, num_box, patch_size):
  patches = nnF.unfold(x, kernel_size=patch_size, stride=patch_size, padding=0) # only 4D inputs supported
  # make one or more patches into black box(es) according to num_box
  num_patches = patches.shape[-1]
  chosen_indices = random.sample(range(num_patches), num_box)
  for i, b_ in enumerate(patches):
      patches[i][:, chosen_indices] = 0
  new_x = nnF.fold(patches, x.shape[-2:], kernel_size=patch_size, stride=patch_size, padding=0)
  return new_x


def blur_patches(x, num_box, patch_size, kernel_size=(11, 11), sigma=3):
  blurrer = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
  patches = nnF.unfold(x, kernel_size=patch_size, stride=patch_size, padding=0) # only 4D inputs supported
  # gaussian blurring
  num_patches = patches.shape[-1]
  chosen_indices = random.sample(range(num_patches), num_box)
  for i, b_ in enumerate(patches):
    for index in chosen_indices:
      patches[i][:, index] = blurrer(patches[i][:, index].reshape(x.shape[1], patch_size, patch_size)).flatten()
  new_x = nnF.fold(patches, x.shape[-2:], kernel_size=patch_size, stride=patch_size, padding=0)
  return new_x