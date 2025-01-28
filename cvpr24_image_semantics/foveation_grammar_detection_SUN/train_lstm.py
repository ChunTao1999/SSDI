from scipy.io import loadmat
from PIL import Image
import torch
import torch.nn.functional as nnF
import torchvision
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
from operator import itemgetter
from dataloader_SUNrgbd import SUN_RGBD
import os
import shutil
import random
from utils import ShufflePatches
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import pyplot as plt
from utils import unfold
# Debug imports
from torchinfo import summary
import pdb

# SEED instantiation
SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
root_path = "/home/nano01/a/tao88/SUN-RGBD" # two folders inside it
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

train_data = SUN_RGBD(root=root_path,
                      seed=SEED,
                      split="train",
                      target_type=["class", "seg_map"],
                      transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, 
                                           batch_size=128, 
                                           shuffle=False)

# Train
save_path = "/home/nano01/a/tao88/4.12_verify_seg_37_classes"
if not os.path.exists(save_path): os.makedirs(save_path)

# Define color map for plotting segmentation mask
num_seg_classes = 38
viridis = mpl.colormaps['viridis'].resampled(num_seg_classes) # 38 colors in the sequential colormap
newcolors = viridis(np.linspace(0, 1, 38))
red = np.array([1, 0, 0, 1])
newcolors[0, :] = red # class interested (now floor)
newcmp = ListedColormap(newcolors)
pdb.set_trace()

for i, (index, images, targets) in enumerate(train_loader):
    translated_images = images.to(device) # (batch_size, num_channels, H, W)
    # targets[0]: scene labels; targets[1]: seg maps
    # targets[1] = torch.round(targets[1]).long()
    
    # visualize 16 images in 1st batch and corresponding seg maps
    if i == 0:
        for img_id in range(128):
            save_image(translated_images[img_id], os.path.join(save_path, 'img_{}.png'.format(img_id)))
            # predicted patch masks from model_FPN
            # fig = plt.figure(figsize=(1,1))
            fig = plt.figure()
            ax = plt.Axes(fig, [0., 0., 1., 1.]) # no borders
            ax.set_axis_off()
            fig.add_axes(ax)
            psm = ax.imshow(X=targets[1][img_id].squeeze(0).cpu().numpy(),
                            interpolation='nearest',
                            cmap=newcmp,
                            vmin=0,
                            vmax=38)
            fig.colorbar(psm, ax=ax)
            plt.savefig(os.path.join(save_path, 'img_{}__mask.png'.format(img_id)))
            plt.close(fig)
    pdb.set_trace()
    # see the bincount of all seg maps
    # print(i)
    # print(torch.bincount(targets[1].long().flatten()))
    
    # crop each image and mask into patches, and then start training LSTM for all 38 classes
    # patches = unfold(tensor=translated_images,
    #                  patch_size=128)

    # Tasks:
    # 1. fix the color visualization of masks (fix image_id=93 first), merge a few classes in seg map together
    # 2. cut the image into patches
    # 3. visualize patches and patch masks

pdb.set_trace()


# initiate LSTM model
