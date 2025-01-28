# imports
from torchvision.datasets.vision import VisionDataset
from scipy.io import loadmat
import mat73
from typing import Any, Callable, List, Optional, Union, Tuple
import torch
import numpy as np
import PIL
import os
import random
import pickle
import json
from utils import transform_label, corrupt_img_landmark_shuffle, corrupt_img_permute
import pdb


class SUN_RGBD(VisionDataset):
    def __init__(self,
                 root: str,
                 seed: Optional[int] = 1,
                 split: str="train", # "train" or "trainval" or "test",
                 target_type: Union[List[str], str] = "class",
                 transform: Optional[Callable] = None,
                 add_corruption: Optional[bool] = False,
                 corruption_type: Optional[str] = "patch_shuffle",
                 num_distortion: Optional[int] = 2,
                 patch_size: Optional[int] = 128):
    
        super(SUN_RGBD, self).__init__(root)

        self.root = root
        self.split = split
        if self.split == "train":
            self.num_imgs = 5285
            self.img_folder = os.path.join(self.root, "train_imgs")
            self.label_folder = os.path.join(self.root, "train_labels")
        elif self.split == "test":
            self.num_imgs = 5050
            self.img_folder = os.path.join(self.root, "test_imgs")
            self.label_folder = os.path.join(self.root, "test_labels")
        self.target_type = target_type
        self.transform = transform
        self.add_corruption = add_corruption
        self.num_distortion = num_distortion
        self.patch_size = patch_size

        # can create a list of shuffled indices here
        index_list = [i for i in range(self.num_imgs)]
        random.seed(seed)
        random.shuffle(index_list)
        self.indices = index_list
        # corruptions
        if self.add_corruption:
            self.corrupt_indices = random.sample(self.indices, len(self.indices) // 2)
            self.corruption_type = corruption_type


    def __len__(self):
        return len(self.indices)
    

    def __getitem__(self, index: int):
        # get the scene name
        # convert the scene string to index
        X = PIL.Image.open(os.path.join(self.img_folder, "img-{:06d}.jpg".format(self.indices[index]+1))) # note plus 1
        seg_map = PIL.Image.open(os.path.join(self.label_folder, "img13labels-{:06d}.png".format(self.indices[index]+1)))

        # transform image and seg map
        if self.transform is not None:
            X = self.transform(X)
        transform_for_labels = transform_label(resize=True, totensor=True, normalize=False, centercrop=False)
        seg_map = torch.round(transform_for_labels(seg_map)*256)

        # Add corruptions to the images and masks together, where corrupt_lbl = 1 if self.add_corruption
        if self.add_corruption:
            corrupt_lbl = 1 if self.indices[index] in self.corrupt_indices else 0
            if corrupt_lbl == 1:
                if self.corruption_type == "patch_shuffle":
                    X, seg_map = corrupt_img_landmark_shuffle(tensors=[X, seg_map],
                                                              num_distortion=self.num_distortion,
                                                              patch_size=self.patch_size)
                elif self.corruption_type == "patch_permute":
                    X, seg_map = corrupt_img_permute(tensors=[X, seg_map],
                                                     patch_size=self.patch_size)

        targets = []
        # if "class" in self.target_type:
        #     targets.append(scene_class)
        if "seg_map" in self.target_type:
            targets.append(seg_map)
        if "corrupt_label" in self.target_type:
            targets.append(corrupt_lbl)

        return self.indices[index], X, targets



#%% backup code to load seg and meta .mat files
# self.meta_matpath = os.path.join(self.root, "SUNRGBDtoolbox", "Metadata", "SUNRGBDMeta.mat")
# meta_dict = loadmat(self.meta_matpath)
# self.all_folderlist = [arr[0][0] for arr in meta_dict['SUNRGBDMeta'][0]]
# self.all_imgpathlist = [arr[5][0][len(prefix):] for arr in meta_dict['SUNRGBDMeta'][0]]
# # load seg dict
# with open("SUNRGBD2Dseg_seglabel", "rb") as fp:   # Unpickling
#     self.seg = pickle.load(fp)
# newdict = {}
# for index in range(len(self.all_folderlist)):
#     newdict[self.all_folderlist[index]] = self.seg[index]

# newdict = {}
# for index in range(len(self.all_folderlist)):
#     newdict[self.all_folderlist[index]] = self.seg[index]

# #To restore from JSON:
# json_load = json.loads(json_dump)
# a_restored = np.asarray(json_load["a"])
# print(a_restored)
# print(a_restored.shape)



