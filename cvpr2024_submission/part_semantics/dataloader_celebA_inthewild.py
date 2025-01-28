#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 23:26:27 2021
Verified on May 25 2022

@author: tibrayev
"""

from collections import namedtuple
import csv
from functools import partial
from itertools import compress
import torch
import os
import PIL

from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg
from torchvision import transforms
from utils_custom_tvision_functions import resize_boxes, hflip_box, custom_Compose, resize_landmarks, hflip_landmark

# tao88
import pdb
import random
import torchvision.transforms.functional as TF
import numpy as np
import math
from data.custom_transforms import *
from torchvision.utils import save_image

CSV = namedtuple("CSV", ["header", "index", "data"])

class CelebA_inthewild(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.
    """

    base_folder = ""
    
    use_wild_imgs = True
    # use_wild_imgs = False # for aligned images
    # tao88 - 12.9
    use_face_cropped = False
    
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U",        "75e246fa4810816ffd6ee81facbd244c",     "Anno/list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS",   "32bd1bd63d3c78cd57e08160ec5ed1e2",     "Anno/identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0",        "00566efa6fedff7a56946cd1c10f1c16",     "Anno/list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk",        "d32c9cbf5e040fd4025c592c306e6668",     "Eval/list_eval_partition.txt"),
    ]
    
    if use_wild_imgs:
        file_list += [("0B7EVK8r0v71peklHb0pGdDl6R28",        "b6cd7e93bc7a96c2dc33f819aa3ac651",     "Imgs/img_celeba.7z"),
                      ("0B7EVK8r0v71pTzJIdlJWdHczRlU",        "063ee6ddb681f96bc9ca28c6febb9d1a",     "Anno/list_landmarks_celeba.txt"),]
    else:
        file_list += [("0B7EVK8r0v71pZjFTYXZWM3FlRnM",        "00d2c5bc6d35e252742224ab0c1e8fcb",     "Imgs/img_align_celeba.zip"),
                      # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc",        "b6cd7e93bc7a96c2dc33f819aa3ac651",     "Imgs/img_align_celeba_png.7z"),                      
                      ("0B7EVK8r0v71pd0FJY3Blby1HUTQ",        "cc24ecafdb5b50baae59b03474781f8c",     "Anno/list_landmarks_align_celeba.txt"),]
        

    def __init__(
                    self,
                    root: str,
                    # tao88
                    labeldir,
                    mode,
                    split: str = "train",
                    target_type: Union[List[str], str] = "attr",
                    transform: Optional[Callable] = None,
                    target_transform: Optional[Callable] = None,
                    download: bool = False,
                    selected_attributes: List[str] = ['all'],
                    at_least_true_attributes: int = 0,
                    treat_attributes_as_classes: bool = False,
                    landmark_shuffle: bool = False,
                    # tao88 - insert inv_list and eqv_list arguments
                    res1=256,
                    res2=512,
                    inv_list = [],
                    eqv_list = [],
                    scale = (0.5, 1)
    ) -> None:

        # in order to take into account possibility of image being resized and/or flipped,
        # and hence, requiring bounding box to be resized and/or flipped accordingly,
        # this dataloader works only with custom_Compose transform (see in custom_tvision_utils.py)
        if (transform is not None) and (not isinstance(transform, custom_Compose)):
            # in case transform is not wrapped into transforms.Compose,
            # then, we can simply wrap it into custom_Compose
            if not isinstance(transform, transforms.Compose):
                if isinstance(transform, list):
                    transform = custom_Compose(transform) # transform is already a list of transforms
                else:
                    transform = custom_Compose([transform]) # transform is only a single transform (e.g. transforms.ToTensor())
            # else, we assume that the transform is already wrapped into transforms.Compose 
            # and throw error
            else:
                raise ValueError("Expected either list of transforms or set of transforms wrapped into custom_Compose")    
        
        super(CelebA_inthewild, self).__init__(root,transform=transform,
        target_transform=target_transform)
        self.split = split
        
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                                ' You can use download=True to download it')
        
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "trainval": (0, 1),
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all", "trainval"))]
        
        splits              = self._load_csv("Eval/list_eval_partition.txt")
        identity            = self._load_csv("Anno/identity_CelebA.txt")
        bbox                = self._load_csv("Anno/list_bbox_celeba.txt", header=1)
        attr                = self._load_csv("Anno/list_attr_celeba.txt", header=1)
        if "pred_init_glimpse_locs" in self.target_type:
            pred_init_glimpse_locs 		= self._load_csv("Anno/predicted_initial_glimpse_locations_celeba.txt", header=1)
        
        if self.use_wild_imgs:
            landmarks       = self._load_csv("Anno/list_landmarks_celeba.txt", header=1)
        else:
            landmarks       = self._load_csv("Anno/list_landmarks_align_celeba.txt", header=1)
        
        # mask = slice(None) if split_ is None else (splits.data == split_).squeeze()
        # # self.filename       = splits.index
        # self.filename       = splits.index if split_ is None else [fname for fname, m in zip(splits.index, mask) if m]
        
        # pre-processing.
        if split_ is None:
            mask            = slice(None)
            self.filename   = splits.index
        else:
            if isinstance(split_, tuple):
                mask            = torch.logical_or((splits.data == split_[0]).squeeze(), (splits.data == split_[1]).squeeze())
            else:
                mask            = (splits.data == split_).squeeze()
            self.filename   = [fname for fname, m in zip(splits.index, mask) if m]
        self.identity       = identity.data[mask]
        self.bbox           = bbox.data[mask]
        self.landmarks      = landmarks.data[mask]
        self.attr           = attr.data[mask]
        # map from {-1, 1} to {0, 1}
        self.all_attr                           = torch.div((self.attr + 1), 2, rounding_mode='floor')
        self.all_attr_names                     = attr.header
        
        if "pred_init_glimpse_locs" in self.target_type:
            self.pred_init_glimpse_locs 		= pred_init_glimpse_locs.data[mask]
        
        # @tibra: added feature of selecting only certain attributes
        # preferred 11 (less subjective) attributes: 
        # selected_attributes = [ 'Bald',  'Black_Hair',  'Blond_Hair',  'Brown_Hair',  'Eyeglasses',  'Male',  'Mustache',  'No_Beard',  'Wearing_Earrings',  'Wearing_Hat',  'Wearing_Necklace',]
        self.sel_attr, self.sel_attr_names      = self._partition_attributes(selected_attributes)      
        
        # @tibra: added feature of selecting only samples which has at least one of the selected attributes
        notification = "Current version of the loader supports only selecting samples which has at least: \n "\
                       "either 0 of the selected attributes (all samples, regardless of the selected attributes) \n"\
                       "or 1 of the selected attributes (exclude samples that have none of the selected attributes)\n"    
        assert at_least_true_attributes in [0, 1], notification
        self.at_least_true_attributes = at_least_true_attributes
        
        if self.at_least_true_attributes != 0:
            mask_sel_attr       = self._select_samples_for_specific_attributes(self.sel_attr)
            self.filename       = [fname for fname, m in zip(self.filename, mask_sel_attr) if m]
            self.sel_attr       = self.sel_attr.data[mask_sel_attr]
            self.identity       = self.identity.data[mask_sel_attr]
            self.bbox           = self.bbox.data[mask_sel_attr]
            self.landmarks      = self.landmarks.data[mask_sel_attr]
			
            if "pred_init_glimpse_locs" in self.target_type:
                self.pred_init_glimpse_locs 	= self.pred_init_glimpse_locs.data[mask_sel_attr]
        
        # @tibra: added feature to treat binary attribute targets as a decimal (integer) class labels
        self.treat_attributes_as_classes = treat_attributes_as_classes        
        
        # tao88
        self.labeldir = labeldir
        self.mode = mode
        self.landmark_shuffle = landmark_shuffle
        self.scale = scale
        self.res1, self.res2 = res1, res2

        self.inv_list = inv_list
        self.eqv_list = eqv_list
        self.view = -1 # fed when performing inv. transform
        self.reshuffle()


    def _partition_attributes(self, selected_attributes):

        if isinstance(selected_attributes, list):
            self.selected_attributes = selected_attributes
        else:
            self.selected_attributes = [selected_attributes]        
        
        if len(self.selected_attributes) == 1 and self.selected_attributes[0] == 'all':
            return self.all_attr, self.all_attr_names
        
        elif len(self.selected_attributes) < len(self.all_attr_names):
            # Check if all requested attributes are present in the dataset
            for s_attr in self.selected_attributes:
                if not s_attr in self.all_attr_names:
                    raise ValueError("Received request for unknown attribute ({})!".format(s_attr))
            # If all present, then create attribute mask
            attribute_mask = torch.zeros(self.all_attr.shape[-1], dtype=torch.bool)
            selected_attrs = []
            for i, attr_name in enumerate(self.all_attr_names):
                if attr_name in self.selected_attributes:
                    attribute_mask[i] = True
                    selected_attrs.append(attr_name)
            return self.all_attr[:, attribute_mask], selected_attrs
        
        elif len(self.selected_attributes) >= len(self.all_attr_names):
            raise ValueError("Received request for more attributes than there are in the dataset!")
        else:
            raise ValueError("Unknown condition for selected_attributes! Ooops!")


    def _select_samples_for_specific_attributes(self, selected_attributes):
        mask_sel_attr = (selected_attributes.sum(dim=1) >= self.at_least_true_attributes)
        return mask_sel_attr
        

    def _convert_attr_to_class(self, attribute):
        attr_class = 0
        for a in attribute:
            attr_class = (2*attr_class) + a
        return attr_class
    

    def _load_csv(
                        self,
                        filename: str,
                        header: Optional[int] = None,
                  ) -> CSV:
        data, indices, headers = [], [], []

        fn = partial(os.path.join, self.root, self.base_folder)
        with open(fn(filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=' ', skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))


    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False
        # Should check a hash of the images
        if self.use_wild_imgs:
            return os.path.isdir(os.path.join(self.root, self.base_folder, "Imgs/img_celeba"))
        else:
            # tao88 - 12.9
            if self.use_face_cropped:
                return os.path.isdir(os.path.join(self.root, self.base_folder, "Imgs/img_face_cropped"))
            return os.path.isdir(os.path.join(self.root, self.base_folder, "Imgs/img_align_celeba"))

    def download(self) -> None:
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        index = self.shuffled_indices[index]

        if self.use_wild_imgs:
            X = PIL.Image.open(os.path.join(self.root, self.base_folder, "Imgs/img_celeba", self.filename[index]))
        elif (not self.use_wild_imgs) and (self.use_face_cropped):
            # tao88 - 12.9
            X = PIL.Image.open(os.path.join(self.root, self.base_folder, "Imgs/img_face_cropped", str(int(self.filename[index].split('.')[0])-1)+'.jpg')) # start from 0
        else:
            X = PIL.Image.open(os.path.join(self.root, self.base_folder, "Imgs/img_align_celeba", self.filename[index]))

        # tao88 - save original image
        # X.save("./assets/image_{}_aligned_orig.png".format(index))
        # tao88 - for mask testing

        w, h = X.size
        if self.transform is not None:
            X, resized, hflipped = self.transform(X)
            if torch.is_tensor(X):
                channels, h_new, w_new = X.shape
            else:
                w_new, h_new = X.size
        else:
            resized = hflipped = False
            w_new, h_new = w, h
        
        # tao88 - for mask testing
        if self.mode=='prepare_mask' and self.view == -1:
            # label = self.transform_label(index)
            return (index, ) + (X, ) + (None, )
              
        # tao88 - apply inv tranforms
        X = self.transform_image(index, X)
        label = self.transform_label(index)

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                if self.treat_attributes_as_classes:
                    # -1 is because we only consider samples which has at least one of the selected attributes true
                    # hence, we exclude the cases with all attributes being 0 and have only 2^(sel_attr.shape[1]) - 1 classes
                    target.append(self._convert_attr_to_class(self.sel_attr[index, :]).item()-self.at_least_true_attributes)
                else:
                    target.append(self.sel_attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])                
            elif t == "bbox":
                bbox = self.bbox[index, :]
                if resized:
                    bbox = resize_boxes(bbox, (h, w), (h_new, w_new))
                if hflipped:
                    bbox = hflip_box(bbox, w_new)
                target.append(bbox)
            elif t == "landmarks":
                # pdb.set_trace()
                landmarks = self.landmarks[index, :]
                if resized:
                    landmarks = resize_landmarks(landmarks, (h, w), (h_new, w_new))
                if hflipped:
                    landmarks = hflip_landmark(landmarks, w_new)
                target.append(landmarks)
            elif t == "pred_init_glimpse_locs":
                glimpse_locs = self.pred_init_glimpse_locs[index, :]
                if (w_new == 256) and (h_new == 256):
                    target.append(glimpse_locs)
                else:
                    raise ValueError("Requested pred_init_glimpse_locs, which were extracted for (256, 256) images, but got image of size ({}, {})!".format(w_new, h_new))
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None
        
        # tao88: landmark_shuffle
        if self.landmark_shuffle:
            if ('landmarks' not in self.target_type) or ('bbox' not in self.target_type):
                raise ValueError("Must have 'landmarks' and 'bbox' in target_type to perform landmark shuffling.")
            landmark_indices = random.sample(range(5), 2) # randomly pick from the 5 landmark locations, make sure the landmarks picked are unique
            landmark_1, landmark_2 = landmarks[landmark_indices] 

            # get bbox area
            bbox_area = bbox[2]*bbox[3]
            bbox_img_ratio = bbox_area / (256*256)
            scaling_factor = torch.sqrt(bbox_img_ratio / 0.2).item() # 0.2 is hyperparameter
            # pdb.set_trace()
            patch_1_ymin = max(0, math.floor(landmark_1[1].item() - 10*scaling_factor)) # floor and ceil so that max-min != 0
            patch_1_ymax = min(255, math.ceil(landmark_1[1].item() + 10*scaling_factor))
            patch_1_xmin = max(0, math.floor(landmark_1[0].item() - 10*scaling_factor))
            patch_1_xmax = min(255, math.ceil(landmark_1[0].item() + 10*scaling_factor))

            patch_2_ymin = max(0, math.floor(landmark_2[1].item() - 10*scaling_factor))
            patch_2_ymax = min(255, math.ceil(landmark_2[1].item() + 10*scaling_factor))
            patch_2_xmin = max(0, math.floor(landmark_2[0].item() - 10*scaling_factor))
            patch_2_xmax = min(255, math.ceil(landmark_2[0].item() + 10*scaling_factor))

            # bi-linear interpolation to resize the crop region, can add anti-alias to make the output for PIL images and tensors closer
            patch_1 = TF.resized_crop(X.clone(), patch_1_ymin, patch_1_xmin, 
                        patch_1_ymax - patch_1_ymin, patch_1_xmax - patch_1_xmin, 
                        (patch_2_ymax - patch_2_ymin, patch_2_xmax - patch_2_xmin)) 

            patch_2 = TF.resized_crop(X.clone(), patch_2_ymin, patch_2_xmin,
                        patch_2_ymax - patch_2_ymin, patch_2_xmax - patch_2_xmin, 
                        (patch_1_ymax - patch_1_ymin, patch_1_xmax - patch_1_xmin))

            X[:, patch_2_ymin:patch_2_ymax, patch_2_xmin:patch_2_xmax] = patch_1
            X[:, patch_1_ymin:patch_1_ymax, patch_1_xmin:patch_1_xmax] = patch_2 
        # return X, target

        # tao88 - save transformed image
        # save_image(X[0], "./visualize_inv/inv2/image_{}_inv2.png".format(index))

        return (index, ) + X + label


    def __len__(self) -> int:
        return len(self.sel_attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

    
    # tao88 - new functions
    def reshuffle(self):
        """
        Generate random floats for all images to deterministically random transform.
        Use random sampling but we have the same samples during clustering and training within the same epoch.
        """
        self.shuffled_indices = np.arange(len(self.sel_attr))
        np.random.shuffle(self.shuffled_indices)
        self.init_transforms()


    def transform_image(self, index, image):
        # Base transform
        image = self.transform_base(index, image)

        if self.mode == 'compute':
            if self.view == 1:
                image = self.transform_inv(index, image, 0)
                image = self.transform_tensor(image)
            elif self.view == 2:
                image = self.transform_inv(index, image, 1)
                image = TF.resize(image, self.res1, Image.BILINEAR)
                image = self.transform_tensor(image)
            else:
                raise ValueError('View [{}] is an invalid option.'.format(self.view))
            return (image, )
        elif 'train' in self.mode:
            # Invariance transform. 
            image1 = self.transform_inv(index, image, 0)
            image1 = self.transform_tensor(image1)

            if self.mode == 'baseline_train':
                return (image1, )
            
            image2 = self.transform_inv(index, image, 1)
            image2 = TF.resize(image2, self.res1, Image.BILINEAR)
            image2 = self.transform_tensor(image2)

            return (image1, image2)
        else:
            raise ValueError('Mode [{}] is an invalid option.'.format(self.mode))


    def transform_inv(self, index, image, ver):
        """
        Hyperparameters same as MoCo v2. 
        (https://github.com/facebookresearch/moco/blob/master/main_moco.py)
        """
        if 'brightness' in self.inv_list:
            image = self.random_color_brightness[ver](index, image)
        if 'contrast' in self.inv_list:
            image = self.random_color_contrast[ver](index, image)
        if 'saturation' in self.inv_list:
            image = self.random_color_saturation[ver](index, image)
        if 'hue' in self.inv_list:
            image = self.random_color_hue[ver](index, image)
        if 'grey' in self.inv_list:
            image = self.random_gray_scale[ver](index, image)
        if 'blur' in self.inv_list:
            image = self.random_gaussian_blur[ver](index, image)
        
        return image


    def transform_eqv(self, indice, image):
        if 'random_crop' in self.eqv_list:
            image = self.random_resized_crop(indice, image)
        if 'h_flip' in self.eqv_list:
            image = self.random_horizontal_flip(indice, image)
        if 'v_flip' in self.eqv_list:
            image = self.random_vertical_flip(indice, image)
        return image


    def init_transforms(self):
        N = len(self.sel_attr)
        
        # Base transform.
        self.transform_base = BaseTransform(self.res2)
        
        # Transforms for invariance. 
        # Color jitter (4), gray scale, blur. 
        self.random_color_brightness = [RandomColorBrightness(x=0.3, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)]
        self.random_color_contrast   = [RandomColorContrast(x=0.3, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)
        self.random_color_saturation = [RandomColorSaturation(x=0.3, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)
        self.random_color_hue        = [RandomColorHue(x=0.1, p=0.8, N=N) for _ in range(2)]      # Control this later (NOTE)
        self.random_gray_scale    = [RandomGrayScale(p=0.2, N=N) for _ in range(2)]
        self.random_gaussian_blur = [RandomGaussianBlur(sigma=[.1, 2.], p=0.5, N=N) for _ in range(2)]

        self.random_horizontal_flip = RandomHorizontalTensorFlip(N=N)
        self.random_vertical_flip   = RandomVerticalFlip(N=N)
        self.random_resized_crop    = RandomResizedCrop(N=N, res=self.res1, scale=self.scale)

        # Tensor transform. 
        self.transform_tensor = TensorTransform()
    

    # We need the labels produced by clustering
    def transform_label(self, index):
        # TODO Equiv. transform.
        if self.mode == 'train':
            label1 = torch.load(os.path.join(self.labeldir, 'label_1', '{}.pkl'.format(index)))
            label2 = torch.load(os.path.join(self.labeldir, 'label_2', '{}.pkl'.format(index)))
            label1 = torch.LongTensor(label1)
            label2 = torch.LongTensor(label2)

            X1 = int(np.sqrt(label1.shape[0]))
            X2 = int(np.sqrt(label2.shape[0]))
            
            label1 = label1.view(X1, X1)
            label2 = label2.view(X2, X2)

            return label1, label2

        elif self.mode == 'baseline_train':
            label1 = torch.load(os.path.join(self.labeldir, 'label_1', '{}.pkl'.format(index)))
            label1 = torch.LongTensor(label1)

            X1 = int(np.sqrt(label1.shape[0]))
            
            label1 = label1.view(X1, X1)

            return (label1, )

        return (None, )



