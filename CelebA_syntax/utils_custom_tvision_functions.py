#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 22:53:23 2022

@author: tibrayev
"""

import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import os
import random
import pdb
import math

#%% =============================================================================
#   Custom Torchvision functions
# =============================================================================
def resize_boxes(boxes, original_size, new_size):
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    # FYI: here assumption that there is only one box input. Change to unbind(1) if multiple boxes are provided at once.
    xmin, ymin, width, height = boxes.unbind(0)

    xmin = xmin * ratio_width
    width = width * ratio_width
    ymin = ymin * ratio_height
    height = height * ratio_height
    return torch.stack((xmin, ymin, width, height), dim=0)

def hflip_box(box, x_bound):
    # assumption: box is 4-sized tuple of (x, y, width, height)
    reducer = torch.zeros_like(box)
    reducer[0] = x_bound - box[2]
    return torch.abs(box - reducer)

def resize_landmarks(landmarks, original_size, new_size):
    ratios = [torch.tensor(s, dtype=torch.float32, device=landmarks.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=landmarks.device)
        for s, s_orig in zip(new_size, original_size)]
    ratio_height, ratio_width = ratios
    # FYI: here assumption that there is only one box input.
    return torch.mul(torch.reshape(landmarks, (5, 2)), torch.tensor([ratio_width, ratio_height])) # torch.Size([5, 2]), each pair is of (x, y)

def hflip_landmark(landmark, x_bound): # just change x for each landmark
    landmark[:, 0] = x_bound - landmark[:, 0]
    return torch.abs(landmark)
    

class custom_Compose:
    """
    CUSTOMIZATION: modified call to output boolean flags, indicating whether or not
    the image was (a) resized, (b) horizontally flipped.
    
    Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        resized     = False
        hflipped    = False
        for t in self.transforms:
            if isinstance(t, custom_RandomHorizontalFlip):
                img, hflipped = t(img)              
            elif isinstance(t, transforms.Resize):
                resized = True
                img = t(img)
            else:
                img = t(img)
        return img, resized, hflipped

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

    
class custom_RandomHorizontalFlip(torch.nn.Module):
    """
    CUSTOMIZATION: modified forward to output boolean flag, indicating whether or not
    the input image was flipped.
    
    Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return F.hflip(img), True
        return img, False

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

#%% =============================================================================
#   Dataloader functions
# =============================================================================
from dataloader_celebA_inthewild import CelebA_inthewild
from dataloader_celebA_aligned import CelebA_aligned
from dataloader_cubirds import CUBirds_2011

def get_dataloaders(config, loader_type='train'):
    assert config.dataset.lower() in ['celeba', 'birds'], "Received unsupported type of dataset!"
    assert loader_type in ['train', 'valid', 'test', 'all', 'trainval'], "Received unsupported type of dataset split!"

    ### CelebA dataset    
    if config.dataset.lower() == 'celeba':
        selected_attributes             = config.selected_attributes
        correct_imbalance               = config.correct_imbalance
        at_least_true_attributes        = config.at_least_true_attributes
        treat_attributes_as_classes     = config.treat_attributes_as_classes
        # tao88
        add_corruption                  = config.add_corruption
        all_corrupt                     = config.all_corrupt # if True, correct all images, otherwise, according to corrupt indices
        landmark_shuffle                = config.landmark_shuffle
        num_distortion                  = config.num_distortion
        black_box                       = config.black_box
        num_box                         = config.num_box
        box_size                        = config.box_size
        gaussian_blur                   = config.gaussian_blur
        puzzle_solving                  = config.puzzle_solving
        puzzle_solving_all_perms = config.puzzle_solving_all_perms
        num_permute                     = config.num_permute
        # 3.7.2023 - tao88
        use_corrupted_testset           = config.use_corrupted_testset
        use_corrupted_valset            = config.use_corrupted_valset
        with_seg_masks                  = config.with_seg_masks
        aligned_mask                    = config.aligned_mask
        with_glimpses                   = config.with_glimpses
        with_glimpse_masks              = config.with_glimpse_masks
        with_glimpse_actions            = config.with_glimpse_actions

        target_type = ['attr', 'bbox', 'pred_init_glimpse_locs', 'landmarks']
        if with_seg_masks:
            target_type.append('masks')
        if with_glimpses:
            target_type.append('glimpses')
        if with_glimpse_masks:
            target_type.append('glimpse_masks')
        if with_glimpse_actions:
            target_type.append('glimpse_actions')
        if add_corruption:
            target_type.append('corrupt_labels')

        if loader_type == 'train' or loader_type == 'trainval':
            transform = custom_Compose([transforms.Resize(size=config.full_res_img_size),
                        # custom_RandomHorizontalFlip(p=0.5), 
                        transforms.ToTensor()])
            data    = CelebA_inthewild(config.dataset_dir, split = loader_type, target_type=['attr', 'bbox', 'pred_init_glimpse_locs', 'landmarks', 'masks'], 
                                        transform = transform, 
                                        selected_attributes              = selected_attributes,
                                        at_least_true_attributes         = at_least_true_attributes, 
                                        treat_attributes_as_classes      = treat_attributes_as_classes,
                                        landmark_shuffle                 = False,
                                        aligned_mask                     = True)
            # return data
            loader  = torch.utils.data.DataLoader(data, batch_size=config.batch_size_train, shuffle=True,
                                                  num_workers=4, pin_memory=False) # suggested max num_workers=8
            # loader  = torch.utils.data.DataLoader(data, batch_size=config.batch_size_train, shuffle=True)
            if correct_imbalance:
                # Since CelebA is imbalanced dataset, during training we can provide weights to the BCEloss
                # First, count how many positive and negative samples are there for each attribute
                cnt_pos_attr = data.all_attr.sum(dim=0)
                cnt_neg_attr = data.all_attr.shape[0] - cnt_pos_attr
                # Then, divide the number of negative samples by the number of positive samples to have scaling factor
                # for each individual attribute. As a result, you will effectively (from the perspective of loss)
                # have the same number of positive examples as that of negative examples.
                # See documentation of BCELoss for more details.
                pos_weight   = cnt_neg_attr*1.0/cnt_pos_attr

                print("Attempt to correct imbalance in attributes distribution!")
                print("Number of positive samples per attribute:")
                print("{}".format(cnt_pos_attr))
                print("Positive weights are:")
                print("{}".format(pos_weight))
                return data, loader, pos_weight
            else:
                equal_weights = torch.tensor([1.0 for _ in range(config.num_classes)])
                return loader, equal_weights, data.__len__()

            
        elif loader_type == 'valid' or loader_type == 'test' or loader_type == 'all':
            if use_corrupted_testset or use_corrupted_valset:
                transform = transforms.ToTensor()
            else:
                transform = custom_Compose([transforms.Resize(size=config.full_res_img_size),
                                            transforms.ToTensor()])
            data    = CelebA_inthewild(config.dataset_dir, 
                                       split = loader_type, 
                                       target_type=target_type, 
                                        transform = transform, 
                                        selected_attributes              = selected_attributes,
                                        at_least_true_attributes         = at_least_true_attributes, 
                                        treat_attributes_as_classes      = treat_attributes_as_classes,
                                        add_corruption                   = add_corruption,
                                        all_corrupt                      = all_corrupt,
                                        landmark_shuffle                 = landmark_shuffle,
                                        num_distortion                   = num_distortion,
                                        black_box                        = black_box,
                                        num_box                          = num_box,
                                        box_size                         = box_size,
                                        gaussian_blur                    = gaussian_blur,
                                        puzzle_solving                   = puzzle_solving,
                                        puzzle_solving_all_perms  = puzzle_solving_all_perms,
                                        num_permute                      = num_permute,
                                        use_corrupted_testset            = use_corrupted_testset,
                                        use_corrupted_valset             = use_corrupted_valset)
            if add_corruption:          
                loader  = torch.utils.data.DataLoader(data, batch_size=config.batch_size_eval, shuffle=True)
            else:
                loader  = torch.utils.data.DataLoader(data, batch_size=config.batch_size_eval, shuffle=False)
            return loader, data.__len__()


    ### PASCAL VOC 2010 dataset
    ### Birds dataset
    elif config.dataset.lower() == 'birds':
        correct_imbalance               = config.correct_imbalance
        
        if loader_type == 'train' or loader_type == 'trainval':
            transform   = custom_Compose([transforms.Resize(size=config.full_res_img_size),
                                          custom_RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor()])
            data        = CUBirds_2011(config.dataset_dir, split = loader_type, target_type = ['attr', 'bbox', 'class'], transform = transform)
            loader      = torch.utils.data.DataLoader(data, batch_size=config.batch_size_train, shuffle=True)

            if correct_imbalance:
                # Since Birds is imbalanced dataset, during training we can provide weights to the BCEloss
                # First, count how many positive and negative samples are there for each attribute
                cnt_pos_attr = data.attr.sum(dim=0)
                cnt_neg_attr = data.attr.shape[0] - cnt_pos_attr
                # Then, divide the number of negative samples by the number of positive samples to have scaling factor
                # for each individual attribute. As a result, you will effectively (from the perspective of loss)
                # have the same number of positive examples as that of negative examples.
                # See documentation of BCELoss for more details.
                pos_weight   = cnt_neg_attr*1.0/cnt_pos_attr

                print("Attempt to correct imbalance in attributes distribution!")
                print("Number of positive samples per attribute:")
                print("{}".format(cnt_pos_attr))
                print("Positive weights are:")
                print("{}".format(pos_weight))
                return loader, pos_weight
            else:
                equal_weights = torch.tensor([1.0 for _ in range(config.num_classes)])
                return loader, equal_weights
        
        elif loader_type == 'valid' or loader_type == 'test' or loader_type == 'all':
            transform   = custom_Compose([transforms.Resize(size=config.full_res_img_size),
                                          transforms.ToTensor()])
            data        = CUBirds_2011(config.dataset_dir, split = loader_type, target_type = ['attr', 'bbox', 'class'], transform = transform)
            loader      = torch.utils.data.DataLoader(data, batch_size=config.batch_size_eval, shuffle=False)
            return loader

#%% Bird no normlization
# def birds_no_normalization(config, loader_type='train'):
#     if loader_type == 'train' or loader_type == 'trainval':    
#         transform   = custom_Compose([transforms.Resize(size=config.full_res_img_size),
#                                       custom_RandomHorizontalFlip(p=0.5),
#                                       transforms.ToTensor()])
#         data        = CUBirds_2011(config.dataset_dir, split = loader_type, target_type = ['attr', 'bbox', 'pred_init_glimpse_locs'], transform = transform)
#         loader      = torch.utils.data.DataLoader(data, batch_size=config.batch_size_train, shuffle=True)
        
#         return loader
#     elif loader_type == 'valid' or loader_type == 'test' or loader_type == 'all':
#         transform   = custom_Compose([transforms.Resize(size=config.full_res_img_size),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
#         data        = CUBirds_2011(config.dataset_dir, split = loader_type, target_type = ['attr', 'bbox', 'pred_init_glimpse_locs'], transform = transform)
#         loader      = torch.utils.data.DataLoader(data, batch_size=config.batch_size_eval, shuffle=False)
#         return loader    



# 3.6.2023 - tao88
#%%
def get_dataloaders_aligned(config, loader_type='train'):
    assert config.dataset.lower() in ['celeba', 'birds'], "Received unsupported type of dataset!"
    assert loader_type in ['train', 'valid', 'test', 'all', 'trainval'], "Received unsupported type of dataset split!"

    ### CelebA dataset    
    if config.dataset.lower() == 'celeba':
        selected_attributes             = config.selected_attributes
        correct_imbalance               = config.correct_imbalance
        at_least_true_attributes        = config.at_least_true_attributes
        treat_attributes_as_classes     = config.treat_attributes_as_classes
        # tao88
        landmark_shuffle                = config.landmark_shuffle
        with_seg_masks                  = config.with_seg_masks
        aligned_mask                    = config.aligned_mask
        with_glimpses                   = config.with_glimpses
        with_glimpse_masks              = config.with_glimpse_masks
        with_glimpse_actions            = config.with_glimpse_actions

        target_type = ['attr', 'bbox', 'pred_init_glimpse_locs', 'landmarks']
        if with_seg_masks:
            target_type.append('masks')
        if with_glimpses:
            target_type.append('glimpses')
        if with_glimpse_masks:
            target_type.append('glimpse_masks')
        if with_glimpse_actions:
            target_type.append('glimpse_actions')

        if loader_type == 'train' or loader_type == 'trainval':
            transform = custom_Compose([transforms.Resize(size=config.full_res_img_size),
                        # custom_RandomHorizontalFlip(p=0.5), 
                        transforms.ToTensor()])
            data    = CelebA_aligned(config.dataset_dir, split = loader_type, target_type=target_type, 
                                    transform = transform, 
                                    selected_attributes              = selected_attributes,
                                    at_least_true_attributes         = at_least_true_attributes, 
                                    treat_attributes_as_classes      = treat_attributes_as_classes,
                                    landmark_shuffle                 = landmark_shuffle,
                                    aligned_mask                     = aligned_mask)
            print(loader_type+" set, len="+str(data.__len__()))
            loader  = torch.utils.data.DataLoader(data, batch_size=config.batch_size_train, shuffle=True,
                                                  num_workers=4, pin_memory=False) # suggested max num_workers=8
            # loader  = torch.utils.data.DataLoader(data, batch_size=config.batch_size_train, shuffle=True)
            if correct_imbalance:
                # Since CelebA is imbalanced dataset, during training we can provide weights to the BCEloss
                # First, count how many positive and negative samples are there for each attribute
                cnt_pos_attr = data.all_attr.sum(dim=0)
                cnt_neg_attr = data.all_attr.shape[0] - cnt_pos_attr
                # Then, divide the number of negative samples by the number of positive samples to have scaling factor
                # for each individual attribute. As a result, you will effectively (from the perspective of loss)
                # have the same number of positive examples as that of negative examples.
                # See documentation of BCELoss for more details.
                pos_weight   = cnt_neg_attr*1.0/cnt_pos_attr

                print("Attempt to correct imbalance in attributes distribution!")
                print("Number of positive samples per attribute:")
                print("{}".format(cnt_pos_attr))
                print("Positive weights are:")
                print("{}".format(pos_weight))
                return loader, pos_weight
            else:
                equal_weights = torch.tensor([1.0 for _ in range(config.num_classes)])
                return loader, equal_weights

            
        elif loader_type == 'valid' or loader_type == 'test' or loader_type == 'all':
            transform = custom_Compose([transforms.Resize(size=config.full_res_img_size),
                                        transforms.ToTensor()])
            data    = CelebA_aligned(config.dataset_dir, split = loader_type, target_type=target_type, 
                                    transform = transform, 
                                    selected_attributes              = selected_attributes,
                                    at_least_true_attributes         = at_least_true_attributes, 
                                    treat_attributes_as_classes      = treat_attributes_as_classes,
                                    landmark_shuffle                 = landmark_shuffle)
            print(loader_type+" set, len="+str(data.__len__()))
            loader  = torch.utils.data.DataLoader(data, batch_size=config.batch_size_eval, shuffle=False)
            return loader
    


#%% =============================================================================
#   Visualization functions
# =============================================================================
def plot_curve(x, y, title, xlabel, ylabel, fname):
    plt.figure()
    plt.plot(x, y, 'b')
    #plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname)
    #plt.show()

def plot_curve_multiple(x, y, title, xlabel, ylabel, fname, legend=""):
    plt.figure()
    for i in range(len(x)):
        plt.plot(x[i], y[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.savefig(fname)

def imshow(input, normalize=True):
    input_to_show = input.cpu().clone().detach()
    if normalize:
        input_to_show = (input_to_show - input_to_show.min())/(input_to_show.max() - input_to_show.min())
    plt.figure()
    if input_to_show.ndim == 4 and input_to_show.size(1) == 3:
        plt.imshow(input_to_show[0].permute(1,2,0))
    elif input_to_show.ndim == 4 and input_to_show.size(1) == 1:
        plt.imshow(input_to_show[0,0])
    elif input_to_show.ndim == 3 and input_to_show.size(0) == 3:
        plt.imshow(input_to_show.permute(1,2,0))
    elif input_to_show.ndim == 3 and input_to_show.size(0) == 1:
        plt.imshow(input_to_show[0])
    elif input_to_show.ndim == 2:
        plt.imshow(input_to_show)
    else:
        raise ValueError("Input with {} dimensions is not supported by this function!".format(input_to_show.ndim))
    # plt.savefig() # added this line to save the resulted fig

def plotregions(list_of_regions, glimpse_size = None, color='g', **kwargs):
    if glimpse_size is None:
        for region in list_of_regions:
            xmin = region[0].item()
            ymin = region[1].item()
            width = region[2].item()
            height = region[3].item()
            # Add the patch to the Axes
            # FYI: Rectangle doc says the first argument defines bottom left corner. However, in reality it changes based on plt axis. 
            # So, if the origin of plt (0,0) is at top left, then (x,y) specify top left corner. 
            # Essentially, (x,y) needs to point to x min and y min of bbox.
            plt.gca().add_patch(Rectangle((xmin,ymin), width, height, linewidth=2, edgecolor=color, facecolor='none', **kwargs))
    elif glimpse_size is not None:
        if isinstance(glimpse_size, tuple):
            width, height = glimpse_size
        else:
            width = height = glimpse_size
        for region in list_of_regions:
            xmin = region[0].item()
            ymin = region[1].item()
            plt.gca().add_patch(Rectangle((xmin,ymin), width, height, linewidth=2, edgecolor=color, facecolor='none', **kwargs))

def plotspots(list_of_spots, color='g', **kwargs):
    for spot in list_of_spots:
        x = spot[0].item()
        y = spot[1].item()
        # Add the circle to the Axes
        plt.gca().add_patch(Circle((x,y), radius=2, edgecolor=color, facecolor=color, **kwargs))

def plotspots_at_regioncenters(list_of_regions, glimpse_size = None, color='g', **kwargs):
    if glimpse_size is None:
        for region in list_of_regions:
            xmin = region[0].item()
            ymin = region[1].item()
            width = region[2].item()
            height = region[3].item()
            x_center = xmin + (width / 2.0)
            y_center = ymin + (height / 2.0)
            plt.gca().add_patch(Circle((x_center, y_center), radius=2, edgecolor=color, facecolor=color, **kwargs))
    elif glimpse_size is not None:
        if isinstance(glimpse_size, tuple):
            width, height = glimpse_size
        else:
            width = height = glimpse_size
        for region in list_of_regions:
            xmin = region[0].item()
            ymin = region[1].item()
            x_center = xmin + (width / 2.0)
            y_center = ymin + (height / 2.0)
            plt.gca().add_patch(Circle((x_center, y_center), radius=2, edgecolor=color, facecolor=color, **kwargs))



#%% =============================================================================
#   IoU functions
# =============================================================================
# taken from torchvision.ops.boxes.py
def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

# taken from torchvision.ops.boxes.py
# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2] left-top
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2] right-bottom

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def region_iou(region1, region2):
    """
    Return intersection-over-union (Jaccard index) of regions.
    
    Here, we define region as a structure in (x1, y1, width, height) format 
    and boxes as a structure in (x1, y1, x2, y2) format.

    Hence, both sets of regions are expected to be in (x1, y1, width, height) format.

    Arguments:
        region1 (Tensor[N, 4])
        region2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in region1 and region2
    """
    boxes1 = region1.clone().detach()
    boxes1[:, 2] += boxes1[:, 0] # x2 = x1 + width
    boxes1[:, 3] += boxes1[:, 1] # y2 = y1 + height
    boxes2 = region2.clone().detach()
    boxes2[:, 2] += boxes2[:, 0]
    boxes2[:, 3] += boxes2[:, 1]
    return box_iou(boxes1, boxes2)

def region_area(regions):
    """
    Computes the area of a set of regions.
    
    Here, we define region as a structure in (x1, y1, width, height) format.

    Arguments:
        regions (Tensor[N, 4]): regions for which the area will be computed. They
            are expected to be in (x1, y1, width, height) format

    Returns:
        area (Tensor[N]): area for each region
    """
    return regions[:, 2] * regions[:, 3]


#%% Fake image generation - tao88
def visualize_shuffled_images(config, loader_type='trainval', sample_size=10):
    selected_attributes = config.selected_attributes
    correct_imbalance = config.correct_imbalance
    at_least_true_attributes = config.at_least_true_attributes
    treat_attributes_as_classes = config.treat_attributes_as_classes
    landmark_shuffle = config.landmark_shuffle # set to False when visualize, otherwise shuffled twice

    if loader_type == 'train' or loader_type == 'trainval':
        transform = custom_Compose([transforms.Resize(size=config.full_res_img_size),
                                    custom_RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor()])
        data = CelebA_inthewild(config.dataset_dir, split = loader_type, target_type=['attr', 'bbox', 'pred_init_glimpse_locs', 'landmarks'], 
                                transform = transform, 
                                selected_attributes              = selected_attributes,
                                at_least_true_attributes         = at_least_true_attributes, 
                                treat_attributes_as_classes      = treat_attributes_as_classes,
                                landmark_shuffle                 = landmark_shuffle)
    elif loader_type == 'valid' or loader_type == 'test' or loader_type == 'all':
        transform = custom_Compose([transforms.Resize(size=config.full_res_img_size),
                                    transforms.ToTensor()])
        data = CelebA_inthewild(config.dataset_dir, split = loader_type, target_type=['attr', 'bbox', 'pred_init_glimpse_locs', 'landmarks'], 
                                transform = transform, 
                                selected_attributes              = selected_attributes,
                                at_least_true_attributes         = at_least_true_attributes, 
                                treat_attributes_as_classes      = treat_attributes_as_classes,
                                landmark_shuffle                 = landmark_shuffle)
    
    # Visualization    
    # Crop 2 regions of fixed size around any 2 of the landmarks, make sure that the regions don't overlap
    # Switch the 2 regions in the image. Return the transformed image tensor.
    indices = random.sample(range(len(data)), sample_size) # indices of the images in the dataset
    save_dir = './test_landmarks'
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    
    for idx in indices:
        image_test = data.__getitem__(idx)
        image_tensor, (attr, bbox, pred_init_glimpse_locs, landmarks) = image_test
        imshow(image_tensor) # image tensor for the 1st image
#        plt.savefig(os.path.join(save_dir, 'im_tensor_' + str(idx) + '.png'))
        plotregions([bbox], glimpse_size=None, color='g')
        # print(landmarks)
        plotspots(landmarks, color='r')
        plt.savefig(os.path.join(save_dir, 'im_tensor_' + str(idx) + '_annotated.png'))

        landmark_indices = random.sample(range(5), 2) # randomly pick from the 5 landmark locations, make sure the landmarks picked are unique
        print('Image_'+str(idx)+': landmarks '+str(landmark_indices))
        landmark_1, landmark_2 = landmarks[landmark_indices] 
        # mind the image borders, and overlaps
        # if the crop region hits the border, resize the crop regions when replacing

        bbox_area = bbox[2]*bbox[3]
        bbox_img_ratio = bbox_area / (256*256)
        scaling_factor = torch.sqrt(bbox_img_ratio / 0.2).item() # 0.2 is hyperparameter

        patch_1_ymin = max(0, math.floor(landmark_1[1].item() - 10*scaling_factor)) # floor and ceil so that max-min != 0
        patch_1_ymax = min(255, math.ceil(landmark_1[1].item() + 10*scaling_factor))
        patch_1_xmin = max(0, math.floor(landmark_1[0].item() - 10*scaling_factor))
        patch_1_xmax = min(255, math.ceil(landmark_1[0].item() + 10*scaling_factor))

        patch_2_ymin = max(0, math.floor(landmark_2[1].item() - 10*scaling_factor))
        patch_2_ymax = min(255, math.ceil(landmark_2[1].item() + 10*scaling_factor))
        patch_2_xmin = max(0, math.floor(landmark_2[0].item() - 10*scaling_factor))
        patch_2_xmax = min(255, math.ceil(landmark_2[0].item() + 10*scaling_factor))

        # bi-linear interpolation to resize the crop region, can add anti-alias to make the output for PIL images and tensors closer
        
        patch_1 = F.resized_crop(image_tensor.clone(), patch_1_ymin, patch_1_xmin, 
                    patch_1_ymax - patch_1_ymin, patch_1_xmax - patch_1_xmin, 
                    (patch_2_ymax - patch_2_ymin, patch_2_xmax - patch_2_xmin))

        patch_2 = F.resized_crop(image_tensor.clone(), patch_2_ymin, patch_2_xmin,
                    patch_2_ymax - patch_2_ymin, patch_2_xmax - patch_2_xmin, 
                    (patch_1_ymax - patch_1_ymin, patch_1_xmax - patch_1_xmin))
        
        image_tensor[:, patch_2_ymin:patch_2_ymax, patch_2_xmin:patch_2_xmax] = patch_1 
        image_tensor[:, patch_1_ymin:patch_1_ymax, patch_1_xmin:patch_1_xmax] = patch_2 

        imshow(image_tensor)
        plt.savefig(os.path.join(save_dir, 'im_tensor_new_' + str(idx) + '.png'))
        imshow(patch_1)
        plt.savefig(os.path.join(save_dir, 'patch_1_' + str(idx) + '.png'))
        imshow(patch_2)
        plt.savefig(os.path.join(save_dir, 'patch_2_' + str(idx) + '.png'))


#%% specify the semantic part around which to crop, and visualize the sample and store the tensor of fixed size
def collect_semantic_samples(config, loader_type='trainval', sample_size=10, semantic_index=0):
    selected_attributes = config.selected_attributes
    correct_imbalance = config.correct_imbalance
    at_least_true_attributes = config.at_least_true_attributes
    treat_attributes_as_classes = config.treat_attributes_as_classes
    landmark_shuffle = config.landmark_shuffle # set to False when visualize, otherwise shuffled twice

    if loader_type == 'train' or loader_type == 'trainval':
        transform = custom_Compose([transforms.Resize(size=config.full_res_img_size),
                                    custom_RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor()])
        data = CelebA_inthewild(config.dataset_dir, split = loader_type, target_type=['attr', 'bbox', 'pred_init_glimpse_locs', 'landmarks'], 
                                transform = transform, 
                                selected_attributes              = selected_attributes,
                                at_least_true_attributes         = at_least_true_attributes, 
                                treat_attributes_as_classes      = treat_attributes_as_classes,
                                landmark_shuffle                 = landmark_shuffle)
    elif loader_type == 'valid' or loader_type == 'test' or loader_type == 'all':
        transform = custom_Compose([transforms.Resize(size=config.full_res_img_size),
                                    transforms.ToTensor()])
        data = CelebA_inthewild(config.dataset_dir, split = loader_type, target_type=['attr', 'bbox', 'pred_init_glimpse_locs', 'landmarks'], 
                                transform = transform, 
                                selected_attributes              = selected_attributes,
                                at_least_true_attributes         = at_least_true_attributes, 
                                treat_attributes_as_classes      = treat_attributes_as_classes,
                                landmark_shuffle                 = landmark_shuffle)

    indices = random.sample(range(len(data)), sample_size) # indices of the images in the dataset
    save_dir = './test_landmarks'
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    
    for idx in indices:
        image_test = data.__getitem__(idx)
        image_tensor, (attr, bbox, pred_init_glimpse_locs, landmarks) = image_test
        imshow(image_tensor) # image tensor for the 1st image
#        plt.savefig(os.path.join(save_dir, 'im_tensor_' + str(idx) + '.png'))
        plotregions([bbox], glimpse_size=None, color='g')
        # print(landmarks)
        plotspots(landmarks, color='r')
        plt.savefig(os.path.join(save_dir, 'Original_images', 'im_tensor_' + str(idx) + '_annotated.png'))

        print('Image_'+str(idx)+': landmarks '+str(semantic_index))
        landmark_1 = landmarks[semantic_index] 

        bbox_area = bbox[2]*bbox[3]
        bbox_img_ratio = bbox_area / (256*256)
        scaling_factor = torch.sqrt(bbox_img_ratio / 0.2).item() # 0.2 is hyperparameter

        patch_1_ymin = max(0, math.floor(landmark_1[1].item() - 10*scaling_factor)) # floor and ceil so that max-min != 0
        patch_1_ymax = min(255, math.ceil(landmark_1[1].item() + 10*scaling_factor))
        patch_1_xmin = max(0, math.floor(landmark_1[0].item() - 10*scaling_factor))
        patch_1_xmax = min(255, math.ceil(landmark_1[0].item() + 10*scaling_factor))

        patch_1_to_save = F.resized_crop(image_tensor.clone(), patch_1_ymin, patch_1_xmin, 
                    patch_1_ymax - patch_1_ymin, patch_1_xmax - patch_1_xmin, (96, 96))

        imshow(patch_1_to_save)
        plt.savefig(os.path.join(save_dir, 'mouth', 'patch_1_' + str(idx) + '.png'))
        torch.save(patch_1_to_save, os.path.join(save_dir, 'mouth', 'path_1_tensor_' + str(idx) + '.pt'))


#%% compute delta glimpse from old and new glimpses locations and dimensions
def compute_delta(old_glimpses_locs_dims, new_glimpses_locs_dims):
    delta_glimpses_locs_dims = torch.zeros_like(new_glimpses_locs_dims)

    for i in range(new_glimpses_locs_dims.shape[0]):
        # x-topleft
        if torch.equal(old_glimpses_locs_dims[i, 0], new_glimpses_locs_dims[i, 0]):
            delta_glimpses_locs_dims[i, 0] = new_glimpses_locs_dims[i, 0] + (new_glimpses_locs_dims[i, 2] - old_glimpses_locs_dims[i, 2]) # move right
        else:
            delta_glimpses_locs_dims[i, 0] = new_glimpses_locs_dims[i, 0]
        # y-topleft
        if torch.equal(old_glimpses_locs_dims[i, 1], new_glimpses_locs_dims[i, 1]):
            delta_glimpses_locs_dims[i, 1] = new_glimpses_locs_dims[i, 1] + (new_glimpses_locs_dims[i, 3] - old_glimpses_locs_dims[i, 3]) # move right
        else:
            delta_glimpses_locs_dims[i, 1] = new_glimpses_locs_dims[i, 1]
        # width
        if torch.equal(old_glimpses_locs_dims[i, 2], new_glimpses_locs_dims[i, 2]):
            delta_glimpses_locs_dims[i, 2] = new_glimpses_locs_dims[i, 2]
        else:
            delta_glimpses_locs_dims[i, 2] = new_glimpses_locs_dims[i, 2] - old_glimpses_locs_dims[i, 2]
        # height
        if torch.equal(old_glimpses_locs_dims[i, 3], new_glimpses_locs_dims[i, 3]):
            delta_glimpses_locs_dims[i, 3] = new_glimpses_locs_dims[i, 3]
        else:
            delta_glimpses_locs_dims[i, 3] = new_glimpses_locs_dims[i, 3] - old_glimpses_locs_dims[i, 3]
        
    return delta_glimpses_locs_dims














