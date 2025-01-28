# imports
from torchvision.datasets.vision import VisionDataset
from scipy.io import loadmat
from typing import Any, Callable, List, Optional, Union, Tuple
import numpy as np
import PIL
import os
import random
import pdb


class Stanford_Dogs(VisionDataset):
    def __init__(self,
                 root: str,
                 seed: Optional[int] = 1,
                 split: str="test", # either "train" or "test",
                 target_type: Union[List[str], str] = "class",
                 processor: Optional[Callable] = None):
    
        super(Stanford_Dogs, self).__init__(root)

        self.root = root
        self.split = split
        self.split_filename_matpath = os.path.join(self.root, self.split+"_list.mat")
        split_dict = loadmat(self.split_filename_matpath)['file_list']
        self.filelist = [arr[0][0] for arr in split_dict]
        self.target_type = target_type
        self.processor = processor
        # can create a list of shuffled indices here
        index_list = [i for i in range(len(self.filelist))]
        random.seed(seed)
        random.shuffle(index_list)
        self.indices = index_list


    def __len__(self):
        return len(self.filelist)
    

    def __getitem__(self, index: int):
        X = PIL.Image.open(os.path.join(self.root, "Images", self.filelist[self.indices[index]]))
        # X = self.processor(X) # use for demo
        if self.processor is not None: # use for exps
            X = self.processor(X, return_tensors="pt")
            X = X['pixel_values'].squeeze() # floatTensor
        target = []
        return self.indices[index], X, target
