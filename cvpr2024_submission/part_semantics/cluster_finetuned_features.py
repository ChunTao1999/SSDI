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
    model, optimizer, classifier1 = get_model_and_optimizer(args, logger) # comment line 46/47 in commons.py
    # pdb.set_trace()
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

    # Cluster Setup
    kmeans_loss  = AverageMeter()
    faiss_module = get_faiss_module(args)
    data_count   = np.zeros(args.K_train)
    featslist    = []
    num_batches  = 0
    first_batch  = True
    # target_img_idx = 0
    with torch.no_grad():
        for batch_idx, (index, images) in enumerate(trainloader):
            images = images.cuda() # (256, 3, 256, 256)
            if batch_idx == 0:
                save_image(images[:16], "/home/nano01/a/tao88/1.23/orig_img_{}.png".format("0-16"))
            feats = model(images) # (256, 128, 64, 64)

            # 1.23 - tao88, debug purpose
            # prb = classifier1(feats)
            # lbl = prb.topk(1, dim=1)[1].squeeze(1)
            # lbl = lbl.detach().cpu()
            # fig = plt.figure(figsize=(8, 8))
            # for r in range(2):
            #     for c in range(8):
            #         fig.add_subplot(2, 8, 8*r+c+1)       
            #         # plt.imshow(mapper_train(lbl_list[8*r+c]))
            #         plt.imshow(lbl[8*r+c])
            # plt.savefig('/home/nano01/a/tao88/1.23/finetuned_model_with_cluster_param_finetuned_{}_epochs.png'.format(100))
            # pdb.set_trace()

            feats = feature_flatten(feats).detach().cpu()
            # clustering all features here, could cluster only features within the face region, produced from previously result
            # could just save the indices of first 20 batches here
            # pdb.set_trace()

            # out = F.normalize(out, dim=1, p=2)
            # prb = compute_dist(out, classifier1)
            # lbl = prb.topk(1, dim=1)[1]
            # lbl = lbl.squeeze(1) # shape (256, 64, 64)
            #                     # 6 is the label for the face region

            # This section collects features only inside the face region. Should be commented if we want all features to be clustered.
            # face_region_feat_indices = (lbl == 6).nonzero() # (number of matches * tensor_dimension)
            # if batch_idx == 0:
            #     first_batch_indices = face_region_feat_indices.detach().cpu()
            # feats = out.transpose(0, 1)
            # # Extract the features from the output
            # feats = feats[:, face_region_feat_indices[:,0], 
            #                 face_region_feat_indices[:,1], 
            #                 face_region_feat_indices[:,2]] # (128, number of matches)
            # feats = feats.transpose(0, 1) # (number of matches, 128)
            # feats = feature_flatten(feats).detach().cpu()

            # Collect the features over 20 (KM_INIT) batches, and then cluster
            if num_batches < args.num_init_batches:
                featslist.append(feats)
                num_batches += 1
                if num_batches == args.num_init_batches:
                    if first_batch:
                        featslist = torch.cat(featslist).cpu().numpy().astype('float32')
                        centroids = get_init_centroids(args, args.K_train, featslist, faiss_module).astype('float32')
                        D, I = faiss_module.search(featslist, 1)
                        kmeans_loss.update(D.mean())
                        logger.info('Initial k-means loss: {:.4f} '.format(kmeans_loss.avg))
                        # Compute counts for each cluster. 
                        for k in np.unique(I):
                            data_count[k] += len(np.where(I == k)[0])
                        first_batch = False
                
            # we have the featslist and the centroids now, retrive the labels for the face-region feats
            if not first_batch:
                # compute dist
                # we note that: d(X, Y)^2 / 2 = 1 - <X, Y>
                 # featslist # (6458153, 128)
                # centroids # (20, 128)
                prb = (np.matmul(featslist, centroids.T) - 1) * 2
                lbl = np.argmax(prb, axis=1)
                lbl = lbl.reshape((args.num_init_batches, 128, 64, 64))
                # pdb.set_trace()

                # use first_batch indices to plot the first 16 images' labels of the 1st batch
                # empty_array = np.zeros((256, 64, 64)) + 20
                # for i in range(first_batch_indices.shape[0]):
                #     empty_array[first_batch_indices[i][0],
                #                 first_batch_indices[i][1],
                #                 first_batch_indices[i][2]] = lbl[i]
                # Plotting
                fig = plt.figure(figsize=(8, 8))
                for r in range(2):
                    for c in range(8):
                        fig.add_subplot(2, 8, 8*r+c+1)       
                        # plt.imshow(mapper_train(lbl_list[8*r+c]))
                        plt.imshow(lbl[0][8*r+c])
                plt.savefig("/home/nano01/a/tao88/1.23/cluster_result_{}_clusters_coarsely_finetuned_100_epochs.png".format(args.K_train, "0-16"))
                # fig = plt.figure(figsize=(8, 8))
                # fig.add_subplot(1, 1, 1)       
                # # plt.imshow(mapper_train(lbl))
                # plt.imshow(empty_array[:16])
                # # plt.show()
                # plt.savefig("/home/nano01/a/tao88/12.17/cluster_result_img_{}.png".format("0-16"))
                pdb.set_trace()
        

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