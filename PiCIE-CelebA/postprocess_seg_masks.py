import os 
import torch  
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import numpy as np
import matplotlib.pyplot as plt 

from torchvision import transforms
from modules import fpn 
from PIL import Image 
import argparse

import time as t
import torchvision.transforms.functional as TF

from utils import *
from commons import * 

# tao88 - for debugging
import pdb
from torchinfo import summary
from torchvision.utils import save_image


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--restart_path', type=str)
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed for reproducability.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers.')
    parser.add_argument('--restart', action='store_true', default=False)
    parser.add_argument('--num_epoch', type=int, default=10) 
    parser.add_argument('--repeats', type=int, default=0)  

    # Train. 
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--res', type=int, default=320, help='Input size.')
    parser.add_argument('--res1', type=int, default=320, help='Input size scale from.')
    parser.add_argument('--res2', type=int, default=640, help='Input size scale to.')
    parser.add_argument('--batch_size_cluster', type=int, default=256)
    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optim_type', type=str, default='Adam')
    parser.add_argument('--num_init_batches', type=int, default=30)
    parser.add_argument('--num_batches', type=int, default=30)
    parser.add_argument('--kmeans_n_iter', type=int, default=30)
    parser.add_argument('--in_dim', type=int, default=128)
    parser.add_argument('--X', type=int, default=80)

    # Loss. 
    parser.add_argument('--metric_train', type=str, default='cosine')   
    parser.add_argument('--metric_test', type=str, default='cosine')
    parser.add_argument('--K_train', type=int, default=27) # COCO Stuff-15 / COCO Thing-12 / COCO All-27
    parser.add_argument('--K_test', type=int, default=27) 
    parser.add_argument('--no_balance', action='store_true', default=False)
    parser.add_argument('--mse', action='store_true', default=False)

    # Dataset. 
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--equiv', action='store_true', default=False)
    parser.add_argument('--min_scale', type=float, default=0.5)
    parser.add_argument('--stuff', action='store_true', default=False)
    parser.add_argument('--thing', action='store_true', default=False)
    parser.add_argument('--jitter', action='store_true', default=False)
    parser.add_argument('--grey', action='store_true', default=False)
    parser.add_argument('--blur', action='store_true', default=False)
    parser.add_argument('--h_flip', action='store_true', default=False)
    parser.add_argument('--v_flip', action='store_true', default=False)
    parser.add_argument('--random_crop', action='store_true', default=False)
    parser.add_argument('--val_type', type=str, default='train')
    parser.add_argument('--version', type=int, default=7)
    parser.add_argument('--fullcoco', action='store_true', default=False)

    # Eval-only
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_path', type=str)

    # Cityscapes-specific.
    parser.add_argument('--cityscapes', action='store_true', default=False)
    parser.add_argument('--label_mode', type=str, default='gtFine')
    parser.add_argument('--long_image', action='store_true', default=False)

    # tao88 - Celeba-specific
    parser.add_argument('--celeba', action='store_true', default=False)
    parser.add_argument('--full_res_img_size', type=tuple, default=(256, 256)) # (height, width), like res1 and res2
    parser.add_argument('--num_classes', type=int, default=40)
    parser.add_argument('--selected_attributes', type=str, default='all')
    parser.add_argument('--correct_imbalance', action='store_true', default=False)
    parser.add_argument('--at_least_true_attributes', type=int, default=0)
    parser.add_argument('--treat_attributes_as_classes', action='store_true', default=False)
    parser.add_argument('--landmark_shuffle', action='store_true', default=False)

    # tao88 - clustering specifc
    parser.add_argument('--with_mask', action='store_true', default=False)
    # tao88 - 1.15
    parser.add_argument('--model_finetuned', action='store_true', default=False)
    parser.add_argument('--finetuned_model_path',type=str, default='')
    
    return parser.parse_args()


def compute_dist(featmap, metric_function, euclidean_train=True):
    centroids = metric_function.module.weight.data
    if euclidean_train:
        return - (1 - 2*metric_function(featmap)\
                    + (centroids*centroids).sum(dim=1).unsqueeze(0)) # negative l2 squared 
    else:
        return metric_function(featmap)


def main(args, logger):
    logger.info(args)

    # Use random seed.
    fix_seed_for_reproducability(args.seed)

    # Start time.
    t_start = t.time()

    # Get model and optimizer.
    model, optimizer, classifier1 = get_model_and_optimizer(args, logger)

    # New trainset inside for-loop.
    # inv_list, eqv_list = get_transform_params(args)
    trainset = get_dataset(args, mode='prepare_mask', inv_list=[], eqv_list=[]) # only one output now

    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=args.batch_size_cluster,
                                              shuffle=False, # important for saving the correct indices
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              collate_fn=collate_train,
                                              worker_init_fn=worker_init_fn(args.seed))

    # Important ! 
    model.eval()
    model.cuda()
    classifier1.eval()
    classifier1.cuda()

 # create a dictionary to record region_idx-region_name correspondances
    region_idx_to_name_dict =  {0: 'hair',
                                1: 'neck_edge',
                                2: 'eye_forehead_edge',
                                3: 'forehead_down',
                                4: 'background',
                                5: 'forehead_up',
                                6: 'background',
                                7: 'nose_eye_edge',
                                8: 'mouth',
                                9: 'hair',
                                10: 'hair',
                                11: 'background',
                                12: 'neck',
                                13: 'background',
                                14: 'hair',
                                15: 'background',
                                16: 'eyes',
                                17: 'nose',
                                18: 'background',
                                19: 'background'}

    region_name_to_idx_after_transform = {'background': 0,
                                            'hair': 1,
                                            'forehead':2,
                                            'eyes':3,
                                            'nose':4,
                                            'mouth':5,
                                            'neck':6}
    for batch_idx, (index, images) in enumerate(trainloader):
        # if batch_idx > 0: break # only visualize images in the first batch
        images = images.cuda() # (256, 3, 256, 256)
        # save_image(images[:16], '/home/nano01/a/tao88/1.23/verify_PiCIE/img_orig.jpg')

        out = model(images)
        out = F.normalize(out, dim=1, p=2)
        prb = compute_dist(out, classifier1)
        lbl = prb.topk(1, dim=1)[1]
        lbl = lbl.squeeze(1) # shape (256, 64, 64)
                             # 6 is the label for the face region

        # 1.26 - tao88, Visualize the region_idx-region_name correspondance 
        # Pick the first image in the first batch as the example lbl to visualize. Later we will visualize region for each lbl index.
        # example_img_index = 0
        # example_lbl = lbl[example_img_index]
        # for region_idx in range(args.K_train):
        #     # zero-out all regions other than region_idx, in lbl
        #     # a. figure out the coord indices for that lbl region
        #     region_coords = (example_lbl == region_idx).nonzero()
        #     # b. prepare an empty tensor to visualize
        #     zero_lbl = torch.zeros_like(example_lbl)
        #     # c. fill the region coord indices with one's
        #     zero_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = 1
        #     save_image(zero_lbl.float(), '/home/nano01/a/tao88/1.26/img_{}_region_{}.png'.format(0, region_idx))
        

        # post process the labels so that same-semantic clusters are merged
        # for example_img_index in range(16):
        #     example_lbl = lbl[example_img_index]
        new_lbl = torch.zeros_like(lbl)
        for region_idx in range(args.K_train):
            region_name = region_idx_to_name_dict[region_idx]
            region_coords = (lbl == region_idx).nonzero()
            # can use match-case instead
            if region_name == 'hair':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['hair']
            elif region_name == 'neck':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['neck']
            elif region_name == 'neck_edge':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['neck']
            elif region_name == 'eye_forehead_edge':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['forehead']
            elif region_name == 'forehead_down':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['forehead']
            elif region_name == 'forehead_up':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['forehead']
            elif region_name == 'background':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['background']
            elif region_name == 'eyes':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['eyes']
            elif region_name == 'nose':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['nose']
            elif region_name == 'nose_eye_edge':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['nose']
            elif region_name == 'mouth':
                new_lbl[region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]] = region_name_to_idx_after_transform['mouth']

        # 1.26 - Plotting after merging clusters / refining seg masks (16 image of the 1st batch)
        # if batch_idx == 0:
        #     fig = plt.figure(figsize=(8, 8))
        #     for r in range(2):
        #         for c in range(8):
        #             fig.add_subplot(2, 8, 8*r+c+1)
        #             plt.imshow(new_lbl[8*r+c].detach().cpu())
        #     plt.savefig('/home/nano01/a/tao88/1.26/transformed_labels/img_{}_seg_merged_clusters_8_epochs.png'.format('0-16'))

        # save all transformed labels in the designated label folder
        new_lbl = new_lbl / len(region_name_to_idx_after_transform.keys()) # normalize to 0-1
        # pdb.set_trace()

        
        for i in range(new_lbl.shape[0]):
            save_image(new_lbl[i], '/home/nano01/a/tao88/celebA_raw/Labels/labels_face_parts_trainval_aligned/{}.jpg'.format(index[i]+1)) # remember to +1 to match the image indices
        if (batch_idx+1) % 100 == 0:
            print("Processed: {}/{}".format(batch_idx + 1, len(trainloader)))
    pdb.set_trace()

if __name__=='__main__':
    args = parse_arguments()

    # Setup the path to save.
    if not args.pretrain:
        args.save_root += '/scratch'
    # tao88
    if args.with_mask:
        args.save_root += '/with_mask'
    if args.augment:
        args.save_root += '/augmented/res1={}_res2={}/jitter={}_blur={}_grey={}'.format(args.res1, args.res2, args.jitter, args.blur, args.grey)
    if args.equiv:
        args.save_root += '/equiv/h_flip={}_v_flip={}_crop={}/min_scale\={}'.format(args.h_flip, args.v_flip, args.random_crop, args.min_scale)
    if args.no_balance:
        args.save_root += '/no_balance'
    if args.mse:
        args.save_root += '/mse'

    args.save_model_path = os.path.join(args.save_root, args.comment, 'K_train={}_{}'.format(args.K_train, args.metric_train))
    args.save_eval_path  = os.path.join(args.save_model_path, 'K_test={}_{}'.format(args.K_test, args.metric_test))
    
    if not os.path.exists(args.save_eval_path):
        os.makedirs(args.save_eval_path)

    # Setup logger.
    logger = set_logger(os.path.join(args.save_eval_path, 'train.log'))
    
    # Start.
    main(args, logger)