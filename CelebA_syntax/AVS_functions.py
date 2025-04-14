#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 17:54:34 2022
Verified on May 25 2022

@author: tibrayev
"""
import torch
import torchvision.transforms.functional as F_vision
from torchvision import transforms
from torchvision.utils import save_image
import pdb

def location_bounds(glimpse_w, input_w):
    """Given input image width and glimpse width returns (lower,upper) bound in (-1,1) for glimpse centers.
    :param: int  glimpse_w      width of glimpse patch
    :param: int  input_w        width of input image
    :return: int lower          lower bound in (-1,1) for glimpse center locations
    :return: int upper
    """
    offset = float(glimpse_w) / input_w
    lower = (-1 + offset)
    upper = (1 - offset)
    assert lower >= -1 and lower <= 1, 'lower must be in (-1,1), is {}'.format(lower)
    assert upper >= -1 and upper <= 1, 'upper must be in (-1,1), is {}'.format(upper)
    return lower, upper


#%% extract_and_resize_glimpses_for_batch
# def extract_and_resize_glimpses_for_batch(images, glimpses_locs_dims, resized_height, resized_width):
#     """
#     Given the batch of images and the batch of glimpse locations and their sizes, 
#     this function extracts these glimpses from images and resizes them to the same size.

#     Parameters
#     ----------
#     images : Tensor[batch_size, channels, height, width]
#         The batch of tensor images, from which glimpses are to be extracted.
#     glimpses_locs_dims : Tensor[batch_size, 4]
#         The batch of glimpse locations and their sizes, where second dimension is 4-sized tuple,
#         representing (x_TopLeftCorner, y_TopLeftCorner, width, height) of each glimpse in the batch.
#     resized_height: Int
#         The height of glimpses in the output batch of glimpses
#     resized_width: Int
#         The width of glimpses in the output batch of glimpses

#     Returns
#     -------
#     batch_extracted_and_resized_glimpses : Tensor[batch_size, channels, resized_height, resized_width]
#         The output batch of extracted and resized glimpses, extracted from the batch of images.
        
#     Note: It is user's responsibility to make sure that the glimpse dimensions do not exceed image dimensions.
#     """
#     batch_extracted_and_resized_glimpses = []
    
#     left_coords = glimpses_locs_dims[:, 0]
#     top_coords  = glimpses_locs_dims[:, 1]
#     widths      = glimpses_locs_dims[:, 2]
#     heights     = glimpses_locs_dims[:, 3]
#     h_fixed     = resized_height
#     w_fixed     = resized_width
    
#     for image, left, top, width, height in zip(images, left_coords, top_coords, widths, heights):
#         resized_glimpse = F_vision.resized_crop(image, top, left, height, width, (h_fixed, w_fixed))
#         batch_extracted_and_resized_glimpses.append(resized_glimpse)

#     batch_extracted_and_resized_glimpses = torch.stack(batch_extracted_and_resized_glimpses, dim=0)
#     return batch_extracted_and_resized_glimpses


#%% extract_and_resize_glimpses_for_batch, can choose copies
def extract_and_resize_glimpses_for_batch(images, glimpses_locs_dims, resized_height, resized_width, resized_height_FPN=256, resized_width_FPN=256, copies=1, interpolation_mode=transforms.InterpolationMode.BILINEAR):
    """
    Given the batch of images and the batch of glimpse locations and their sizes, 
    this function extracts these glimpses from images and resizes them to the same size.

    Parameters
    ----------
    images : Tensor[batch_size, channels, height, width]
        The batch of tensor images, from which glimpses are to be extracted.
    glimpses_locs_dims : Tensor[batch_size, 4]
        The batch of glimpse locations and their sizes, where second dimension is 4-sized tuple,
        representing (x_TopLeftCorner, y_TopLeftCorner, width, height) of each glimpse in the batch.
    resized_height: Int
        The height of glimpses in the output batch of glimpses
    resized_width: Int
        The width of glimpses in the output batch of glimpses

    Returns
    -------
    batch_extracted_and_resized_glimpses : Tensor[batch_size, channels, resized_height, resized_width]
        The output batch of extracted and resized glimpses, extracted from the batch of images.
        
    Note: It is user's responsibility to make sure that the glimpse dimensions do not exceed image dimensions.
    """
    batch_extracted_and_resized_glimpses = []
    # 2.7.2023 - tao88
    batch_extracted_and_resized_glimpses_FPN = []
    
    left_coords = glimpses_locs_dims[:, 0]
    top_coords  = glimpses_locs_dims[:, 1]
    widths      = glimpses_locs_dims[:, 2]
    heights     = glimpses_locs_dims[:, 3]
    h_fixed     = resized_height
    w_fixed     = resized_width
    # 2.7.2023 - tao88
    h_fixed_FPN = resized_height_FPN
    w_fixed_FPN = resized_width_FPN
    
    for image, left, top, width, height in zip(images, left_coords, top_coords, widths, heights):
        resized_glimpse = F_vision.resized_crop(image, top, left, height, width, (h_fixed, w_fixed), interpolation=interpolation_mode)
        batch_extracted_and_resized_glimpses.append(resized_glimpse)

        # 2.7.2023 - tao88
        resized_glimpse_FPN = F_vision.resized_crop(image, top, left, height, width, (h_fixed_FPN, w_fixed_FPN), interpolation=interpolation_mode)
        batch_extracted_and_resized_glimpses_FPN.append(resized_glimpse_FPN)

    batch_extracted_and_resized_glimpses = torch.stack(batch_extracted_and_resized_glimpses, dim=0)
    # 2.7.2023 - tao88
    batch_extracted_and_resized_glimpses_FPN = torch.stack(batch_extracted_and_resized_glimpses_FPN, dim=0)
    if copies == 1:
        return batch_extracted_and_resized_glimpses
    else:
        return batch_extracted_and_resized_glimpses, batch_extracted_and_resized_glimpses_FPN
    

#%% 3.10.2023 - tao88: crop five patches around center
def crop_five(images, left_coords=[64,128,64,128,96], top_coords=[64,64,128,128,96], widths=[64,64,64,64,64], heights=[64,64,64,64,64], resized_height=256, resized_width=256):
    batch_cropped_five = []
    for img_idx, image in enumerate(images):
        image_cropped_five = []
        for patch_idx, left, top, width, height in zip(range(5), left_coords, top_coords, widths, heights):
            resized_patch = F_vision.resized_crop(image, top, left, height, width, (resized_height, resized_width))
            image_cropped_five.append(resized_patch)
            # save_image(resized_patch, '/home/nano01/a/tao88/five_patches/img_{}_patch_{}.png'.format(img_idx, patch_idx))
        image_cropped_five = torch.stack(image_cropped_five, dim=0)
        batch_cropped_five.append(image_cropped_five)
    batch_cropped_five = torch.stack(batch_cropped_five, dim=0)
    return batch_cropped_five


def extract_and_resize_masks(masks, left_coords=[64,128,64,128,96], top_coords=[64,64,128,128,96], widths=[64,64,64,64,64], heights=[64,64,64,64,64], resized_height=64, resized_width=64):
    batch_cropped_five = []
    for mask_idx, mask in enumerate(masks):
        mask_cropped_five = []
        for patch_idx, left, top, width, height in zip(range(5), left_coords, top_coords, widths, heights):
            resized_patch = F_vision.resized_crop(mask, top, left, height, width, (resized_height, resized_width))
            mask_cropped_five.append(resized_patch)
            # save_image(resized_patch, '/home/nano01/a/tao88/five_patches/img_{}_patch_{}.png'.format(img_idx, patch_idx))
        mask_cropped_five = torch.stack(mask_cropped_five, dim=0)
        batch_cropped_five.append(mask_cropped_five)
    batch_cropped_five = torch.stack(batch_cropped_five, dim=0)
    return batch_cropped_five