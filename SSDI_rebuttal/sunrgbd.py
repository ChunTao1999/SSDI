import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from PIL import Image

class SUNRGBD(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=[], ps=80):
        self.transform = transform
        self.phase_train = phase_train
        self.data_dir = data_dir
        self.ps = ps
        self.imgs = []

        if isinstance(data_dir, list):
            for img_dir in data_dir:
                if os.path.exists(img_dir):
                    images = sorted(os.listdir(img_dir))
                    for img in images:
                        img_path = os.path.join(img_dir, img)                        
                        if not self._is_all_zero_image(img_path):
                            self.imgs.append(img_path)
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img

    def _is_all_zero_image(self, img_path):
        """
        Check if an image consists only of zero values.
        """
        try:
            img = Image.open(img_path).convert('L')  # Open as grayscale
            img_array = np.array(img)
            return np.all(img_array == 0)
        except (IOError, ValueError):  # Handle errors when opening or reading images
            print(f"Warning: Could not open or process {img_path}. Skipping this image.")
            return True  # Assume the image is all zero in case of an error


class scaleNorm:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img):
        return transforms.functional.resize(img, (self.height, self.width), transforms.InterpolationMode.NEAREST)
    
class ToTensor:
    def __call__(self, img):
        return transforms.functional.to_tensor(img)


