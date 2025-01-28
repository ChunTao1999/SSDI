#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:32:59 2021
Verified on Wed May 25 2022

@author: saketi, tibrayev

Defines all hyperparameters.
"""

class AVS_config(object):
    # SEED
    seed                    = 7
    
    # dataset
    dataset                     = 'celeba'
    # dataset_dir                 = '/path/to/celeba/dataset/'
    dataset_dir                 = '/home/nano01/a/tao88/celebA_raw'
    in_num_channels             = 3
    full_res_img_size           = (256, 256)
    num_classes                 = 40    # 40 binary attributes in celebA
    selected_attributes         = 'all'
    correct_imbalance           = False
    at_least_true_attributes    = 0
    treat_attributes_as_classes = False

    # model
    vgg_name                = 'vgg8_narrow_k2'
    pretrained              = False
    downsampling            = 'M'
    fc1                     = 512
    fc2                     = 256
    dropout                 = 0.5
    norm                    = None
    adaptive_avg_pool_out   = (2, 2)
    init_weights            = True

    # training
    train_loader_type       = 'trainval'
    if train_loader_type == 'train':
        valid_loader_type   = 'valid'
    elif train_loader_type == 'trainval':
        valid_loader_type   = 'test'
        print("Warning: selected training on trainval split, hence validation is going to be performed on test split!")
    else:
        raise ValueError("Unrecognized type of split to train on: ({})".format(train_loader_type))
    
    experiment_name         = (dataset + '/trained_on_{}_split/train_M1/{}/'.format(train_loader_type, vgg_name) + 
                               'pretrained_{}_normalization_{}_loss_correct_imbalance_{}_seed_{}/'.format(pretrained, norm, correct_imbalance, seed))
    save_dir                = './results_new/' + experiment_name
    batch_size_train        = 128
    batch_size_eval         = 50
    epochs                  = 200
    lr_start                = 1e-3
    lr_min                  = 1e-5
    milestones              = [100, 150]
    weight_decay            = 0.0001
    
    # testing
    ckpt_dir                = save_dir + 'model.pth'
    attr_detection_th       = 0.5
