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
from celebAHQ import dataloader_celebAHQ
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
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--model_finetuned', action='store_true', default=False)
    return parser.parse_args()


def compute_dist(featmap, metric_function, euclidean_train=True):
    centroids = metric_function.module.weight.data
    if euclidean_train:
        return - (1 - 2*metric_function(featmap)\
                    + (centroids*centroids).sum(dim=1).unsqueeze(0)) # negative l2 squared 
    else:
        return metric_function(featmap)

preprocess = transforms.Compose([transforms.ToTensor(), 
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                      std=[0.229, 0.224, 0.225])]) # can instead use 0.5 normalization

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
    model, _, classifier1 = get_model_and_optimizer(args, logger) # we ignore the pretrained classifier
    # classifier1 = initialize_classifier(args) # could add a softmax layer
    logger.info('Adam optimizer is used.')

    # optimizers
    # pdb.set_trace() # check if parameters are set to requires_grad (done)
    optimizer1 = torch.optim.Adam(filter(lambda x: x.requires_grad, model.module.parameters()), lr=args.lr)
    lr_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, gamma = 0.1, milestones=[40, 80], verbose=True)

    optimizer2 = torch.optim.Adam(filter(lambda x: x.requires_grad, classifier1.module.parameters()), lr=args.lr)
    lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, gamma = 0.1, milestones=[40, 80], verbose=True)
    # pdb.set_trace() # model imported, done

    # Loss criterion
    criterion = nn.CrossEntropyLoss()

    img_path = "/home/nano01/a/tao88/celebA_raw/CelebAMask-HQ/CelebA-HQ-img"
    label_path = "/home/nano01/a/tao88/celebA_raw/CelebAMask-HQ/CelebA-HQ-label-coarse"
        
    mode = True # whether train mode
    trainloader = dataloader_celebAHQ.Data_Loader(img_path=img_path, label_path=label_path, image_size=256, batch_size=args.batch_size_train, mode=mode).loader()
    # pdb.set_trace() # dataloader fine, done

    # Important ! 
    model.train()
    model.cuda()
    classifier1.train()
    classifier1.cuda()

    # Training (Finetuning)
    t1 = t.time()
    logger.info(' Start Training.')
    for epoch in range(args.num_epoch):
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.cuda(), labels.cuda() # (128, 3, 256, 256), (128, 64, 64)
            # if batch_idx == 0:
            #     save_image(images[:16], "/home/nano01/a/tao88/1.5/orig_img_{}.png".format("0-16"))
            # sanity check, visualize the labels too
            # fig = plt.figure(figsize=(8, 8))
            # fig.add_subplot(1, 1, 1)
            # plt.imshow(labels[0].detach().cpu())
            # plt.savefig("/home/nano01/a/tao88/1.5/during_finetune_label_{}".format("0"))
            # pdb.set_trace()

            features = model(images) # (128, 128, 64, 64)
            out = classifier1(features) # (128, 12, 64, 64)
            # pdb.set_trace() # done
            
            # process the labels, use torch.round instead of .long()
            labels = torch.round(labels * 7).long() # (128, 64, 64)

            # compute loss between out and label
            # out = torch.argmax(out, dim=1) # (128, 64, 64)
            loss = criterion(out, labels)
            optimizer2.zero_grad()
            optimizer1.zero_grad()
            loss.backward()
            optimizer2.step()
            optimizer1.step()

        # report loss
        logger.info('============== Epoch [{}] =============='.format(epoch))
        logger.info('  Time: [{}]'.format(get_datetime(int(t.time())-int(t1))))
        logger.info('  Training CE Loss: {:.5f}'.format(loss))
        logger.info('========================================\n')

        # lr scheduler, can add a min lr
        lr_scheduler1.step()
        lr_scheduler2.step()

    # save the finetuned models   
    logger.info('Training finished, saving model and classifier1 parameters.')         
    torch.save({'epoch': epoch+1, 
                'args' : args,
                'state_dict': model.state_dict(),
                'classifier1_state_dict' : classifier1.state_dict(),
                'optimizer1' : optimizer1.state_dict(),
                'optimizer2' : optimizer2.state_dict()
                },
                os.path.join(args.save_model_path, 'checkpoint_{}_epochs.pth.tar'.format(epoch+1)))

    pdb.set_trace()


if __name__=='__main__':
    args = parse_arguments()

    # Setup the path to save.
    if not args.pretrain:
        args.save_root += '/scratch'
    # tao88
    if args.finetune:
        args.save_root += 'finetune'
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
