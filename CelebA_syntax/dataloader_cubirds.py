#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 22:40:10 2022
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
from torchvision.datasets.utils import verify_str_arg
from torchvision import transforms
from utils_custom_tvision_functions import resize_boxes, hflip_box, custom_Compose

CSV = namedtuple("CSV", ["header", "index", "data"])

class CUBirds_2011(VisionDataset):
    """`Caltech-UCSD Birds-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>` Dataset.
    
    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``class``, ``attr``, ``bbox``,
            or ``parts (Not implemented!)``. Can also be a list to output a tuple with all specified target types.
            The targets represent:
                
                - ``class`` (int): one of 200 classes images are categorized into
                - ``attr`` (np.array shape=(312,) dtype=int): binary (0, 1) labels for attributes
                - ``bbox`` (np.array shape=(4,) dtype=float): bounding box (x, y, width, height)
                - ``parts`` (Not Implemented!): (x, y) coordinates of parts of objects
				- ``pred_init_glimpse_locs`` (np.array shape=(2,) dtype=float): predicted initial glimpse locations (x, y),
				  specifically for the images resized to (256, 256).
            
            Defaults to ``class``.
        
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
            
        This dataloader assumes that the data is already downloaded and unzipped in the root directory provided.
    """
    
    base_folder = "CUB_200_2011"
    
    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "class",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:

        # in order to take into account possibility of image being resized and/or flipped,
        # and hence, requiring bounding box to be reszied and/or flipped accordingly,
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

        super(CUBirds_2011, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        
        self.split = split
        
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]
        if "parts" in self.target_type:
            raise NotImplementedError("target_type 'parts' is not implemented by the current loader!")
        if not self.target_type: # in case of empty list as target_type
            self.target_type.append("class")
        
        split_map = {
            "train": 1,
            "test": 0,
            "valid": 2,
            "trainval": (1, 2),
            "all": None}
        
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                         ("train", "test", "valid", "all", "trainval"))]
        
        # fetching.
        splits          = self._load_csv("train_test_val_split.txt",data_type='int')
        filename        = self._load_csv("images.txt",              data_type='str')
        labels          = self._load_csv("image_class_labels.txt",  data_type='int')
        label_names     = self._load_csv("classes.txt",             data_type='str')
        bbox            = self._load_csv("bounding_boxes.txt",      data_type='float')
        attr            = self._load_csv("attributes/image_attribute_labels.txt", data_type='int', specific_columns=[1, 3])
        attr_names      = self._load_csv("attributes/attributes.txt", data_type='str')      
        if "pred_init_glimpse_locs" in self.target_type:
            pred_init_glimpse_locs 		= self._load_csv("predicted_initial_glimpse_locations_cub.txt", data_type='float')
        
        # pre-processing.
        if split_ is None:
            mask            = slice(None)
            self.filename   = [fname[0] for fname in filename.data]
        else:
            if isinstance(split_, tuple):
                mask            = torch.logical_or((splits.data == split_[0]).squeeze(), (splits.data == split_[1]).squeeze())
            else:
                mask            = (splits.data == split_).squeeze()
            self.filename   = [fname[0] for fname, m in zip(filename.data, mask) if m]
        self.labels         = (labels.data[mask] - 1).squeeze() # dataset labels start from 1
        self.label_names    = {int(k)-1: v[0] for k, v in zip(label_names.index, label_names.data)}
        self.bbox           = bbox.data[mask]
        self.attr           = attr.data.reshape(len(filename.index), len(attr_names.index), 2)[:, :, -1][mask]
        self.attr_names     = {int(k)-1: v[0] for k, v in zip(attr_names.index, attr_names.data)}
        if "pred_init_glimpse_locs" in self.target_type:
            self.pred_init_glimpse_locs 		= pred_init_glimpse_locs.data[mask]

    def _load_csv(
                        self,
                        filename: str,
                        header: Optional[int] = None,
                        data_type: Optional[str] = 'int',
                        specific_columns: Optional[List[int]] = None,
                  ) -> CSV:
        data, indices, headers = [], [], []

        fn = partial(os.path.join, self.root, self.base_folder)
        with open(fn(filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=' ', skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]

        indices = [row[0] for row in data]
        if specific_columns is not None:        
            data = [row[specific_columns[0]:specific_columns[1]] for row in data]
        else:
            data = [row[1:] for row in data]
        if data_type=='int':
            data_int = [list(map(int, i)) for i in data]
            return CSV(headers, indices, torch.tensor(data_int))
        elif data_type=='float':
            data_int = [list(map(int, map(float, i))) for i in data]
            return CSV(headers, indices, torch.tensor(data_int))
        elif data_type=='str':
            return CSV(headers, indices, data)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "images", self.filename[index]))
        
        if X.mode != 'RGB': # some of the images in the birds dataset are black and white
            X = X.convert("RGB")
        
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
        
        target: Any = []
        for t in self.target_type:
            if t == "class":
                target.append(self.labels[index])
            elif t == "attr":
                target.append(self.attr[index, :])
            elif t == "bbox":
                bbox = self.bbox[index, :]
                if resized:
                    bbox = resize_boxes(bbox, (h, w), (h_new, w_new))
                if hflipped:
                    bbox = hflip_box(bbox, w_new)
                target.append(bbox)
            elif t == "pred_init_glimpse_locs":
                glimpse_locs = self.pred_init_glimpse_locs[index, :]
                if (w_new == 256) and (h_new == 256):
                    target.append(glimpse_locs)
                else:
                    raise ValueError("Requested pred_init_glimpse_locs, which were extracted for (256, 256) images, but got image of size ({}, {})!".format(w_new, h_new))
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))
        
        target = tuple(target) if len(target) > 1 else target[0]
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return X, target
    
    def __len__(self) -> int:
        return len(self.filename)
    
    def extra_repr(self) -> str:
        lines = ["target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


    








