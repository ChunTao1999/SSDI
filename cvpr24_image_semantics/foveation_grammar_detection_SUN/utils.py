import torch
from torchvision import transforms
import torch.nn.functional as nnF
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

class ShufflePatches(object):
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

def unfold(tensor, patch_size):
   x = nnF.unfold(tensor, kernel_size=patch_size, stride=patch_size, padding=0)
   return x
    

# write a new function, use unfold function to break the image into patches, so that they can be fed to LSTM
