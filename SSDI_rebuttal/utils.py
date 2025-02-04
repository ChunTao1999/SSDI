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