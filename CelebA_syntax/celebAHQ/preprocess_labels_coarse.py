from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.io import read_image
from matplotlib import pyplot as plt
from matplotlib import colors
import os
import pdb

face_part_to_id = {'skin': 1, # background is 0, skin is 1
                   'hair': 2, # hair is 2
                   'l_brow': 3, # eye is 3
                   'l_eye': 3,
                   'l_lip': 5, # mouth is 5
                   'mouth': 5,
                   'neck': 6, # neck is 6
                   'nose': 4, # nose is 4
                   'r_brow': 3,
                   'r_eye': 3,
                   'u_lip': 5}
num_classes = max(face_part_to_id.values()) + 1 # 7 classes
print("Number of classes: {}".format(num_classes))
img_path = "/home/nano01/a/tao88/celebA_raw/CelebAMask-HQ/CelebA-HQ-img"
label_path = "/home/nano01/a/tao88/celebA_raw/CelebAMask-HQ/CelebAMask-HQ-mask-anno"
save_path = "/home/nano01/a/tao88/celebA_raw/CelebAMask-HQ/CelebA-HQ-label-coarse"
if not os.path.exists(save_path): os.mkdir(save_path)

# From labels, identify the indices corresponding to each facial part for each image in img_path, then assign the pixels to these labels and save in save_path.
for i in range(len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])):
    # label_path_set = []
    label_tensor = torch.zeros(512, 512, dtype=float)
    for face_part in ['skin', 'hair', 'l_brow', 'l_eye', 'l_lip', 'mouth', 'neck', 'nose', 'r_brow', 'r_eye', 'u_lip']: # can add "skin"
        face_part_label_path = os.path.join(label_path, format(i, '05') + '_' + face_part + '.png') # need to convert the image index to 5 digits
        if not os.path.isfile(face_part_label_path):
            continue
        # label_path_set.append(face_part_label_path)
        face_part_tensor = read_image(face_part_label_path)[0] # we only need the first channel
        indices = (face_part_tensor == 255).nonzero()
        id = face_part_to_id[face_part]
        label_tensor[indices[:,0], indices[:,1]] = id

    # save the pre-processed label to the label folder
    save_image(label_tensor / num_classes, os.path.join(save_path, str(i)+'.png'))
    if (i + 1) % 1000 == 0:
        print("{}/{}".format(i + 1, len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])))

print("Preprocesssing of all labels finished.")

    # pdb.set_trace()
    # Plotting
    # fig = plt.figure(figsize=(8, 8))
    # fig.add_subplot(1, 1, 1)
    # # plt.imshow(label_tensor.permute(1, 2, 0) / 10)
    # plt.imshow(label_tensor / 11)
    # plt.savefig("/home/nano01/a/tao88/1.1/processed_label_{}".format("0"))
    # pdb.set_trace()


# img = Image.open(label_path)
# transform = transforms.ToTensor()
# img_tensor = transform(img)
# pdb.set_trace()