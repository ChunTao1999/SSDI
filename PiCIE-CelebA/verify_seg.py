import torch
import argparse
import os
import time as t
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from utils import *
from commons import * 
from modules import fpn 

# tao88 - for debugging
import pdb
from torchinfo import summary
from torchvision.utils import save_image

# tao88 - from foveation folder
from utils_custom_tvision_functions import get_dataloaders, plot_curve, plot_curve_multiple

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

# preprocess = transforms.Compose([transforms.ToTensor(), 
#                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                                       std=[0.229, 0.224, 0.225])]) # can instead use 0.5 normalization

def selective_mask_t(image_src, mask, channels=[]):
    mask = mask[:, torch.tensor(channels).long()]
    mask = torch.sgn(torch.sum(mask, dim=1)).to(dtype=image_src.dtype).unsqueeze(-1)
    return mask * image_src


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
                                              shuffle=False, 
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              collate_fn=collate_train,
                                              worker_init_fn=worker_init_fn(args.seed))

    # Important ! 
    model.eval()
    model.cuda()
    classifier1.eval()
    classifier1.cuda()

    for batch_idx, (index, images) in enumerate(trainloader):
        images = images.cuda() # (256, 3, 256, 256)
        # pdb.set_trace()
        out = model(images)
        out = F.normalize(out, dim=1, p=2)
        prb = compute_dist(out, classifier1)
        # pdb.set_trace() # print prb.shape
        lbl = prb.topk(1, dim=1)[1]
        lbl = lbl.squeeze(1) # shape (256, 64, 64)
                             # 6 is the label for the face region
        pdb.set_trace() # identify the face region label index here, from lbl.
        
        lbl_binary = torch.where(lbl==6, 1, 0) # In case of --eval_path /home/nano01/a/tao88/PiCIE-CelebA/results/picie_on_celeba/train/1/augmented/res1=256_res2=512/jitter=True_blur=True_grey=True/equiv/h_flip=True_v_flip=False_crop=True/min_scale\\=0.5/K_train=20_cosine/checkpoint.pth.tar, face region label is 6
        lbl_binary_resized = TF.resize(lbl_binary, (images.shape[-2], images.shape[-1])) # can specify BILINEAR
        lbl_binary_resized = lbl_binary_resized.to(dtype=images.dtype) # (256, 256, 256)
        images_masked = torch.mul(lbl_binary_resized.unsqueeze(1), images) # verified
        save_image(images_masked[:16], '/home/nano01/a/tao88/1.17/img_masked.jpg')
        pdb.set_trace()
        # for i in range(images_masked.size(0)):
        #    save_image(images_masked[i, :, :, :], '/home/nano01/a/tao88/celebA_raw/Imgs/img_face_cropped/{}.jpg'.format(index[i]))
        # if (batch_idx+1) % 100 == 0:
        #     print("{}/{}".format(batch_idx+1, len(trainloader)))

        # we can extract the features otherwise


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