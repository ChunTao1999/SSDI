# Author: tao88 Tao
# Date: 2023-10-02
import argparse
from celebAHQ import dataloader_celebAHQ
import json
import os
import time as t
import torch
import torch.nn as nn
from utils import *
# Huggingface
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
# Debug tools
from torchinfo import summary
import pdb


#%% Arguments
parser = argparse.ArgumentParser()
# train data paths
parser.add_argument('--celebahq_img_path', type=str, default="", required=True, help="path to celebahq images")
parser.add_argument('--celebahq_label_path', type=str, default="", required=True, help='path to celebahq seg labels')
parser.add_argument('--label_mapping_path', type=str, default="", required=True, help='path to label to class mapping json file')
# train configs
parser.add_argument('--batch_size_train', type=int, default=128, required=False, help="train batch size")
parser.add_argument('--num_epochs', type=int, default=40, required=False, help='number of train epochs')

args = parser.parse_args()


#%% Logger setup
logger = set_logger(os.path.join('logs', 'train_vit_on_celeba.log'))
logger.info(args)


#%% Label mapping
# id2label = {1:"skin", 2:"hair", 3:"eyes", 4:"nose", 5:"mouth", 6:"neck"}
# json.dump(id2label, open("./celebAHQ/celebahq-id2label.json", 'w'))
id2label = json.load(open(args.label_mapping_path, 'r'))
label2id = {v:k for k,v in id2label.items()}
num_semantic_classes = len(id2label.keys())


#%% Dataset and Dataloader
trainloader = dataloader_celebAHQ.Data_Loader(img_path=args.celebahq_img_path, 
                                              label_path=args.celebahq_label_path,
                                              image_size=256, 
                                              batch_size=args.batch_size_train, 
                                              mode=True).loader()


#%% Preprocess, Models
checkpoint = "nvidia/mit-b0"
image_processor = AutoImageProcessor.from_pretrained(checkpoint,
                                                     do_reduce_labels=True)
model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, 
                                                         id2label=id2label, 
                                                         label2id=label2id)
model.cuda()


# Loss criterion
criterion = nn.CrossEntropyLoss(weight=None,
                                reduction="mean")

#%% Train loop
t_start = t.time()
logger.info("Start training......")
model.train()
for epoch in range(args.num_epochs):
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.cuda(), labels.cuda() # (128, 3, 256, 256), (128, 64, 64)
        labels = torch.round(labels * 7).long()
        # divide the labels into 6 mask copies for each semantic class
        # class_labels = torch.zeros((labels.shape[0], num_semantic_classes, labels.shape[-2], labels.shape[-1]), dtype=torch.long)
        # for class_idx in range(1, num_semantic_classes + 1):
        #     class_labels[class_idx - 1] = (labels == class_idx).long()
        pdb.set_trace()


        out = model(images) # (128, 6, 64, 64)
        # loss = criterion(out, labels)
        pdb.set_trace()

pdb.set_trace()
