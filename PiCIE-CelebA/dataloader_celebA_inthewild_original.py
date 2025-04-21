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

import pdb
import random
import torchvision.transforms.functional as F
import math

CSV = namedtuple("CSV", ["header", "index", "data"])

class CelebA_inthewild(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)
				- ``pred_init_glimpse_locs`` (np.array shape=(2,) dtype=int): predicted initial glimpse locations (x, y),
				  specifically for the images resized to (256, 256).

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        landmark_shuffle(bool, optional): If true, perform one time shuffling of 2 bounding 
            boxes around 2 random landmarks.
    """

    base_folder = ""
    # COMMENT from original celeba.py pytorch file.
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    # This script is modified to include already downloaded and unzipped images in-the-wild.
    
    use_wild_imgs = False
    
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
                    split: str = "train",
                    target_type: Union[List[str], str] = "attr",
                    transform: Optional[Callable] = None,
                    target_transform: Optional[Callable] = None,
                    download: bool = False,
                    selected_attributes: List[str] = ['all'],
                    at_least_true_attributes: int = 0,
                    treat_attributes_as_classes: bool = False,
                    landmark_shuffle: bool = False
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
        
        super(CelebA_inthewild, self).__init__(root, transform=transform,
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
        
        # tao88: add landmark_shuffle
        self.landmark_shuffle = landmark_shuffle
        
#        # checking dist. of semantic groups among all attributes
#        eyes_mask = [1, 3, 12, 15, 23]
#        nose_mask = [7, 27]
#        cheek_mask = [19, 29, 30]
#        mouth_mask = [0, 6, 14, 16, 21, 22, 24, 36]
#        hair_mask = [4, 5, 8, 9, 11, 17, 28, 32, 33, 35]
#        general_mask = [2, 10, 13, 18, 20, 25, 26, 31, 34, 37, 38, 39]
#        num_not_empty_groups = (self.all_attr[:, eyes_mask].sum(dim=1) >= 1).to(torch.int) \
#                                 + (self.all_attr[:, nose_mask].sum(dim=1) >= 1).to(torch.int) \
#                                 + (self.all_attr[:, cheek_mask].sum(dim=1) >= 1).to(torch.int) \
#                                 + (self.all_attr[:, mouth_mask].sum(dim=1) >= 1).to(torch.int) \
#                                 + (self.all_attr[:, hair_mask].sum(dim=1) >= 1).to(torch.int)
#        # use torch.bincount or torch.numel to check the distribution
#        pdb.set_trace()

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
        if self.use_wild_imgs:
            X = PIL.Image.open(os.path.join(self.root, self.base_folder, "Imgs/img_celeba", self.filename[index]))
        else:
            X = PIL.Image.open(os.path.join(self.root, self.base_folder, "Imgs/img_align_celeba", self.filename[index]))
                
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
                
        # print("old width, height ", w, h)
        # print("new width, height ", w_new, h_new)

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
            patch_1 = F.resized_crop(X.clone(), patch_1_ymin, patch_1_xmin, 
                        patch_1_ymax - patch_1_ymin, patch_1_xmax - patch_1_xmin, 
                        (patch_2_ymax - patch_2_ymin, patch_2_xmax - patch_2_xmin)) 

            patch_2 = F.resized_crop(X.clone(), patch_2_ymin, patch_2_xmin,
                        patch_2_ymax - patch_2_ymin, patch_2_xmax - patch_2_xmin, 
                        (patch_1_ymax - patch_1_ymin, patch_1_xmax - patch_1_xmin))

            X[:, patch_2_ymin:patch_2_ymax, patch_2_xmin:patch_2_xmax] = patch_1
            X[:, patch_1_ymin:patch_1_ymax, patch_1_xmin:patch_1_xmax] = patch_2 

        return X, target

    def __len__(self) -> int:
        return len(self.sel_attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)
