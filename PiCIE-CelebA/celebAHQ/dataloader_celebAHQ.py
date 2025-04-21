import torch
import torchvision.datasets as dsets
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import os
from matplotlib import pyplot as plt
# tao88 - debug
import pdb

class CelebAMaskHQ():
    def __init__(self, img_path, label_path, transform_img, transform_label, mode):
        self.img_path = img_path
        self.label_path = label_path
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        self.preprocess()
        
        if mode == True:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        
        for i in range(len([name for name in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, name))])):
            img_path = os.path.join(self.img_path, str(i)+'.jpg')
            # pdb.set_trace() # done
            label_path = os.path.join(self.label_path, str(i)+'.png')
            # print (img_path, label_path) 
            if self.mode == True:
                self.train_dataset.append([img_path, label_path])
            else:
                self.test_dataset.append([img_path, label_path])
            
        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        
        dataset = self.train_dataset if self.mode == True else self.test_dataset
        img_path, label_path = dataset[index]
        image = Image.open(img_path)
        label = Image.open(label_path)
        return self.transform_img(image), self.transform_label(label)[0] # we only need 1 channel of 3 from the label image

    def __len__(self):
        """Return the number of images."""
        return self.num_images

class Data_Loader():
    def __init__(self, img_path, label_path, image_size, batch_size, mode):
        self.img_path = img_path
        self.label_path = label_path
        self.imsize = image_size
        self.batch = batch_size
        self.mode = mode

    def transform_img(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def transform_label(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize // 4,self.imsize // 4),
                                            interpolation=InterpolationMode.NEAREST)) # 64x64
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0, 0, 0), (0, 0, 0)))
        transform = transforms.Compose(options)
        return transform

    def loader(self):
        transform_img = self.transform_img(True, True, True, False) # resize, totensor, normalize
        transform_label = self.transform_label(True, True, False, False) # resize, totensor
        dataset = CelebAMaskHQ(self.img_path, self.label_path, transform_img, transform_label, self.mode)
        # image0, label0 = dataset.__getitem__(0)
        # pdb.set_trace()
        # fig = plt.figure(figsize=(8, 8))
        # fig.add_subplot(1, 1, 1)
        # plt.imshow(label0[0])
        # plt.savefig("/home/nano01/a/tao88/1.5/dataset_label_{}".format("0"))
        # pdb.set_trace()
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch,
                                             shuffle=True,
                                             num_workers=2,
                                             drop_last=False)
        return loader

# For test of functionality of celebAHQ dataloader
if __name__ == "__main__":
    img_path = "/home/nano01/a/tao88/celebA_raw/CelebAMask-HQ/CelebA-HQ-img"
    label_path = "/home/nano01/a/tao88/celebA_raw/CelebAMask-HQ/CelebA-HQ-label"
    mode = True # whether train mode
    trainloader = Data_Loader(img_path=img_path, label_path=label_path, image_size=256, batch_size=256, mode=mode).loader()
    
    pdb.set_trace()
