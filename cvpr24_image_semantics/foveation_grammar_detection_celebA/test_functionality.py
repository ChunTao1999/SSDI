from torchvision import transforms
import PIL
import os
import torch
import pdb

glimpse_path = "/home/nano01/a/tao88/celebA_raw/glimpse_imgs/val"
glimpses_list = []
transform = transforms.ToTensor()
for g in range(16):
    glimpse = PIL.Image.open(os.path.join(glimpse_path, 'img_{}_g_{}.jpg'.format(0, g)))
    glimpse = transform(glimpse)
    glimpses_list.append(glimpse)
torch.stack(glimpses_list)
pdb.set_trace()
