from collections import namedtuple
import csv
from functools import partial
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
import torch
import PIL
import os
from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import verify_str_arg
from torchvision import transforms
from utils_custom_tvision_functions import resize_boxes, hflip_box, custom_Compose
# 3.31.2023 - tao88
import pdb
import shutil

CSV = namedtuple("CSV", ["header", "index", "data"])


class PASCAL_VOC_2010(VisionDataset):
    """PASCAL_VOC_2010 dataset http://host.robots.ox.ac.uk/pascal/VOC/voc2010/
       PASCAL-Part datast http://roozbehm.info/pascal-parts/pascal-parts.html

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'trainval', 'val', 'test', 'all'}. According dataset is selected.
        target_type (string or list, optional):
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

        This dataloader assumes that the data is already downloaded and unzipped in the root directory provided.
    """

    base_folder = "VOCdevkit"

    def __init__(self,
                 root: str,
                 split: str="trainval",
                 target_type: Union[List[str], str] = "class",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None
                 ):

        # this dataloader works only with custom_Compose transform (see in custom_tvision_utils.py)
        if (transform is not None) and (not isinstance(transform, custom_Compose)):
            if not isinstance(transform, transforms.Compose):
                if isinstance(transform, list):
                    transform = custom_Compose(transform) # transform is already a list of transforms
                else:
                    transform = custom_Compose([transform]) # transform is only a single transform (e.g. transforms.ToTensor())
            else:
                raise ValueError("Expected either list of transforms or set of transforms wrapped into custom_Compose")
            
        super(PASCAL_VOC_2010, self).__init__(root, transform=transform, target_transform=target_transform)

        # class attributes
        self.split = split

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]
        if not self.target_type: # in case of empty list as target_type
            self.target_type.append("class")

        split_map = {
            "train": 1,
            "test": 0,
            "valid": 2,
            "trainval": (1, 2),
            "all": None}
        
        split_ = split_map[verify_str_arg(split.lower(), "split", ["train", "test", "valid", "trainval", "all"])] # verify_str_arg(value, arg, valid_values, custom_msg)
        
        # fetch the annotation csvs
        aeroplane_train = self._load_csv(filename="VOC2010/ImageSets/Main/aeroplane_train.txt", data_type='int')
        aeroplane_trainval = self._load_csv(filename="VOC2010/ImageSets/Main/aeroplane_trainval.txt", data_type='int')
        # pre-processing
        self.aeroplane_train_filenames = [fname for (i, fname) in enumerate(aeroplane_train.index) if aeroplane_train.data[i].item()==1] # omit the negative (-1) and "difficult" (0) cases
        self.aeroplane_trainval_filenames = [fname for (i, fname) in enumerate(aeroplane_trainval.index) if aeroplane_trainval.data[i].item()==1] 
        # create a dictionary that corresponds part description to part index
        self.aeroplane_partdict = {'background': 0,
                                   'engine': 1, # engine*
                                   'lwing': 2,
                                   'rwing': 3,
                                   'stern': 4,
                                   'tail': 5,
                                   'wheel': 6} # wheel*
        # path to part annotations
        self.part_seg_folder = "Annotations_Part"


    def _load_csv(self,
                  filename: str,
                  header: Optional[int]=None,
                  data_type: Optional[str]='int',
                  specific_columns: Optional[List[int]]=None) -> CSV:
        data, indices, headers = [], [], []

        fn = partial(os.path.join, self.root, self.base_folder)
        with open(fn(filename)) as csv_file: # the missing argument for os.path.join is filename
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
    

    def _load_mat(self,
                  filename: str,
                  data_type: Optional[str]='long'):
        '''function to load the part segmentation annotations in .mat format'''
        fn = partial(os.path.join, self.root, self.base_folder, self.part_seg_folder)
        mat = loadmat(fn(filename + '.mat')) # fn(filename) is the path to the part annotation of image with imgname
        # mat['anno']['objects'][0][0][0][0][3][0] contains the 7 segmentations

        # beware, .mat file may not contain all parts, use the dictionary to make the overall mask
        # deal with multiple object case
        num_objects = mat['anno']['objects'][0][0][0].shape[0]
        # for object_idx in range(num_objects):
        #     object_name = mat['anno']['objects'][0][0][0]

        part_info = mat['anno']['objects'][0][0][0][0][3][0]
        for part_idx in range(part_info.shape[0]):
            part_name, part_mask_binary = str(part_info[part_idx][0][0]), part_info[part_idx][1]
            if "_" in part_name: # incase of "engine_x" and "wheel_x"
                part_name = part_name.split("_")[0]

            if part_idx == 0:
                img_mask = torch.from_numpy(part_mask_binary).long() # (375, 500)
            else:
                img_mask += torch.from_numpy(part_mask_binary * (part_idx+1)).long() # don't use add, directly set the level
        return img_mask # (375, 500), values from 0 to 7
        

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "VOC2010", "JPEGImages", self.aeroplane_trainval_filenames[index]))

        if X.mode != 'RGB': # in case black and white
            X = X.convert("RGB")
        
        w, h = X.size
        # image transform

        target: Any = []
        for t in self.target_type:
            if t == "class":
                target.append(self.labels[index])
            elif t == "part_seg":
                target.append()

        return X, target
        




# Trial run to create dataset class
root = "/home/nano01/a/tao88/PASCAL_VOC_2010"
data = PASCAL_VOC_2010(root=root,
                       split="trainval",
                       )
# visualize the airplane class images
# for fname in data.aeroplane_trainval_filenames:
#     orig_path = os.path.join(root, data.base_folder, "VOC2010", "JPEGImages", fname+".jpg")
#     dest_path = os.path.join("/home/nano01/a/tao88/PASCAL_aeroplane_trainval", fname+".jpg")
#     shutil.copy(orig_path, dest_path)

# for fname in data.aeroplane_trainval_filenames:
#     orig_path = os.path.join(root, data.base_folder, data.part_seg_folder, fname+".mat")
#     dest_path = os.path.join("/home/nano01/a/tao88/aeroplane_trainval_partsegs", fname+".mat")
#     shutil.copy(orig_path, dest_path)

# visualize the airplane class part annotations
for fname in data.aeroplane_trainval_filenames:
    fname = "2008_000251"
    try:
        part_mask = data._load_mat(filename=fname)
    except IndexError as e: # some .mat files don't have correct part annotation for aeroplanes
        continue
    # plot masks
    save_path = '/home/nano01/a/tao88/4.3_visualize_plane_masks' # (375, 500)
    if not os.path.exists(save_path): os.makedirs(save_path)

    fig = plt.figure(figsize=(10,10))
    fig.add_subplot(1, 1, 1)
    plt.imshow(part_mask.cpu().numpy(),
               interpolation='nearest',
               cmap='Paired', # color scheme
               vmin=0,
               vmax=7)
    plt.axis('off')
    plt.savefig(os.path.join(save_path, '{}_part_mask.png'.format(fname)))

pdb.set_trace()

