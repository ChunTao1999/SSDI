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
from utils import transform_label
import pdb


class SUN_RGBD(VisionDataset):
    def __init__(self,
                 root: str,
                 seed: Optional[int] = 1,
                 split: str="trainval", # "train" or "trainval" or "test",
                 target_type: Union[List[str], str] = "class",
                 transform: Optional[Callable] = None):
    
        super(SUN_RGBD, self).__init__(root)

        self.root = root
        self.split = split
        self.allsplit_matpath = os.path.join(self.root, "SUNRGBDtoolbox", "traintestSUNRGBD", "allsplit.mat")
        allsplit_dict = loadmat(self.allsplit_matpath)
        prefix = '/n/fs/sun3d/data/'
        self.alltrain_folderlist = [(arr[0][len(prefix):-1] if arr[0][-1]=='/' else arr[0][len(prefix):]) for arr in allsplit_dict['alltrain'][0]]
        self.alltest_folderlist = [(arr[0][len(prefix):-1] if arr[0][-1]=='/' else arr[0][len(prefix):]) for arr in allsplit_dict['alltest'][0]] # verified

        # Load 2 dictionaries, mapping from each folder seq name to image path and to 2d segmentation array
        f = open("seq_to_imgpath.pkl", "rb")
        self.seq_to_imgpath = pickle.load(f)
        f.close()

        f = open("seq_to_seg.pkl", "rb")
        self.seq_to_seg = pickle.load(f)
        f.close()

        self.target_type = target_type
        self.transform = transform
        
        # can create a list of shuffled indices here
        index_list = [i for i in range(len(self.alltrain_folderlist))]
        random.seed(seed)
        random.shuffle(index_list)
        self.indices = index_list


    def __len__(self):
        return len(self.alltrain_folderlist)
    

    def __getitem__(self, index: int):
        folder_path = (self.alltrain_folderlist[self.indices[index]])
        img_path = self.seq_to_imgpath[folder_path]
        with open(os.path.join(self.root, folder_path, "scene.txt")) as f:
            scene_class = f.readline()
        # conver the scene string to index
        seg_map = PIL.Image.fromarray(self.seq_to_seg[folder_path])
        X = PIL.Image.open(os.path.join(self.root, img_path))
        if self.transform is not None:
            X = self.transform(X)
        targets = []
        if "class" in self.target_type:
            targets.append(scene_class)
        if "seg_map" in self.target_type:
            transform_for_labels = transform_label(resize=True, totensor=True, normalize=False, centercrop=False)
            targets.append(transform_for_labels(seg_map))
        
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



