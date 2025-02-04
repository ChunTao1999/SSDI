# Imports
from transformers import AutoImageProcessor, ViTImageProcessor, ViTForImageClassification
from datasets import load_dataset
from scipy.io import loadmat
from PIL import Image
import requests
import torch
import torch.nn.functional as nnF
import torchvision
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
from operator import itemgetter
from dataloader_stanford_dogs import Stanford_Dogs
import os
import shutil
import random
from corruption_functions import PermutePatches, shuffle_patches, black_patches, blur_patches
# Debug imports
from torchinfo import summary
import pdb


#%% Test single image from url
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

# processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# logits = outputs.logits
# # model predicts one of 1000 ImageNet classes
# # predicted_class_idx = logits.argmax(-1).item()
# # print("Predicted class:", model.config.id2label[predicted_class_idx])

# # convert logits to probabilities
# prob = nnF.softmax(logits, dim=1)
# top_p, top_class_indices = prob.topk(5, dim=1)
# top_p = top_p.squeeze().tolist()
# top_classes = itemgetter(*top_class_indices.squeeze().tolist())(model.config.id2label)
# print("Inference on vit-16-224:\n")
# for class_name, prob in zip(top_classes, top_p):
#     print("{0:25s}: probility {1:8.3f}".format(class_name, prob))

#%% Test on Stanford Dogs (half-corrupted)
# SEED instantiation
SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = 'cuda' if torch.cuda.is_available() else 'cpu'
root_path = "/home/nano01/a/tao88/Stanford-dogs"
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224') # for experiments
# processor_totensor = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]) # for demo
test_data = Stanford_Dogs(root=root_path,
                          seed=SEED,
                          split="test",
                          target_type="class",
                          processor=processor)
test_loader = torch.utils.data.DataLoader(test_data, 
                                          batch_size=128, 
                                          shuffle=False)
# import and build your corrupted dataloader here

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.to(device)
permute_patches_function = PermutePatches(patch_size=(112, 112))

# Visualization folder
save_path = "/home/nano01/a/tao88/5.7_dog"
demo_quantity = 2
if not os.path.exists(save_path): os.makedirs(save_path)

with torch.no_grad():
    for i, (index, images, targets) in enumerate(test_loader):
        inputs = images.to(device) # (batch_size, num_channels, H, W)
        pdb.set_trace()

        # 5.7.2023 - tao88: compare orig prediction with 2 shuffles, 1 black, and 1 blur
        # Visualize corruption
        save_image(inputs[0], os.path.join(save_path, "imgs_correct_0.png"))
        save_image(inputs[1], os.path.join(save_path, "imgs_correct_1.png"))
        # # corrupted_inputs = shuffle_patches(x=inputs[:demo_quantity],
        # #                                    num_distortion=4, # num_distortion at least 2
        # #                                    patch_size=112)
        corrupted_inputs = permute_patches_function(inputs[:demo_quantity])
        # corrupted_inputs = blur_patches(x=inputs[:demo_quantity],
        #                                  num_box=2,
        #                                  patch_size=112)
        save_image(corrupted_inputs[0], os.path.join(save_path, "imgs_shuffle2_0.png"))
        save_image(corrupted_inputs[1], os.path.join(save_path, "imgs_shuffle2_1.png"))
        pdb.set_trace()

        # correct test
        outputs = model(inputs)
        logits = outputs.logits # (batch_size, 1000)
        prob = nnF.softmax(logits, dim=1) # convert to prob.
        top_p, top_class_indices = prob.topk(5, dim=1) # top 5 prob., and corresponding class indices
        # Visualization of first 2 images and their predictions
        print("\nCorrect images results:")
        for img_id in range(demo_quantity):
            # original image
            filename = test_data.filelist[index[img_id]]
            source = os.path.join(root_path, "Images", filename)
            destination = os.path.join(save_path, "img_{}.jpg".format(img_id))
            shutil.copy(source, destination)
            probs, classes = top_p[img_id].tolist(), itemgetter(*top_class_indices[img_id].tolist())(model.config.id2label)
            print("Image {} predictions by ViT: ".format(img_id))
            for class_name, prob in zip(classes, probs):
                print("{0:50s}: probility {1:8.3f}".format(class_name, prob))

        # corrupted test
        # # 1. shuffle
        # corrupted_inputs = shuffle_patches(x=inputs[:demo_quantity],
        #                                    num_distortion=4, # num_distortion at least 2
        #                                    patch_size=112)
        # corrupted_inputs = permute_patches_function(inputs[:demo_quantity])
        # 2. black
        # corrupted_inputs = black_patches(x=inputs[:demo_quantity],
        #                                  num_box=2,
        #                                  patch_size=112)
        # 3. gaussian blur
        # corrupted_inputs = blur_patches(x=inputs[:demo_quantity],
        #                                 num_box=2,
        #                                 patch_size=112,
        #                                 kernel_size=(11, 11), 
        #                                 sigma=3)
        outputs = model(corrupted_inputs)
        logits = outputs.logits # (batch_size, 1000)
        prob = nnF.softmax(logits, dim=1) # convert to prob.
        top_p, top_class_indices = prob.topk(5, dim=1) # top 5 prob., and corresponding class indices
        # Visualization of first 2 images and their predictions
        print("\nCorrupted images results:")
        for img_id in range(demo_quantity):
            probs, classes = top_p[img_id].tolist(), itemgetter(*top_class_indices[img_id].tolist())(model.config.id2label)
            print("Image {} predictions by ViT: ".format(img_id))
            for class_name, prob in zip(classes, probs):
                print("{0:50s}: probility {1:8.3f}".format(class_name, prob))
        
        pdb.set_trace()
