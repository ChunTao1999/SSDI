#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:32:59 2021
Verified on Wed May 25 2022

@author: saketi, tibrayev, tao

Defines all hyperparameters.
"""

class config_test(object):
    # SEED
    seed                        = 1
    # Dataset
    dataset                     = 'celeba'
    dataset_dir                 = '/home/nano01/a/tao88/celebA_raw'
    in_num_channels             = 3
    full_res_img_size           = (256, 256) #(height, width) as used in transforms.Resize
    num_classes                 = 40    # 40 binary attributes in celebA
    selected_attributes         = 'all'
    correct_imbalance           = False
    at_least_true_attributes    = 0
    treat_attributes_as_classes = False
    # Dataset extra configs
    with_seg_masks              = False # whether or not to use seg masks in the target
    aligned_mask                = False
    with_glimpses               = False
    with_glimpse_masks          = False
    with_glimpse_actions        = False
    data_parallel               = False
    num_devices                 = 4 # enter the number of gpu devices

    # Dataset corruption configs
    add_corruption              = True
    all_corrupt                 = False # set to True for puzzle solving task
    # (please only set one type of corruption to True)
    landmark_shuffle            = False # always set to False when training
    black_box                   = False
    gaussian_blur               = True
    puzzle_solving              = False
    puzzle_solving_all_perms    = False

    num_distortion              = 2
    scaling_factor              = 1 # ratio of face to image, 1 for aligned images
    num_box                     = 1
    box_size                    = 20
    num_permute                 = 3

    # 3.7.2023 - tao88: add option to use half corrupted testset (those should be False when corruption type is specified)
    use_corrupted_testset       = False  
    use_corrupted_valset        = False

    # Note: the script is written such that the configs of M1 and M2 should be given as a separate file,
    # but only the path to pretrained models is provided by this config file (with settings for M3)
    # model_M1
    ckpt_dir_model_M1       = './results/celeba/trained_on_trainval_split/train_M1/vgg8_narrow_k2/pretrained_False_normalization_None_loss_correct_imbalance_False_seed_7/model.pth'
    # model_M2
    ckpt_dir_model_M2       = './results/celeba/trained_on_trainval_split/train_M2/model.pth'
    
    # model_M3
    vgg_name                = 'vgg8_narrow_k2'
#    initialize_M3           = 'from_M1'
    initialize_M3           = 'from_checkpoint' # for M4 training
    downsampling            = 'M'
    fc1                     = 512
    fc2                     = 256
    dropout                 = 0.5
    norm                    = None
    adaptive_avg_pool_out   = (2, 2)
    init_weights            = False
    
    # training
    # train_loader_type       = 'trainval' # use trainval for training
    train_loader_type       = 'trainval'
    if train_loader_type == 'train':
        valid_loader_type   = 'valid'
    elif train_loader_type == 'trainval':
        valid_loader_type   = 'test'
        print("Warning: selected training on trainval split, hence validation is going to be performed on test split!")
    else:
        raise ValueError("Unrecognized type of split to train on: ({})".format(train_loader_type))
    test_loader_type        = 'test'

    if landmark_shuffle:
        experiment_name         = (dataset + '/trained_on_{}_split_landmark_shuffle/train_M3/v7_extE/'.format(train_loader_type) + 
                               'arch_{}_initialized_{}_normalization_{}_loss_correct_imbalance_{}_seed_{}/'.format(vgg_name, initialize_M3, norm, correct_imbalance, seed))
    else:
        experiment_name         = (dataset + '/trained_on_{}_split/train_M3/v7_extE/'.format(train_loader_type) + 
                               'arch_{}_initialized_{}_normalization_{}_loss_correct_imbalance_{}_seed_{}/'.format(vgg_name, initialize_M3, norm, correct_imbalance, seed))
    
    save_dir                = './results_new/' + experiment_name
    batch_size_train        = 128 # 16 works for all glimpse iterations; better be divisible by num_devices
    batch_size_eval         = 128 # changed from 100 to 128
    epochs                  = 40
    lr_start                = 1e-4
    lr_min                  = 1e-6
    milestones              = [40, 80]
    weight_decay            = 0.0001
    
    # testing
    # ckpt_dir                = save_dir + 'model.pth'
    # For M4 training:
    ckpt_dir_model_M3 = './results/celeba/trained_on_trainval_split/train_M3/v7_extE/arch_vgg8_narrow_k2_initialized_from_M1_normalization_None_loss_correct_imbalance_False_seed_19/model.pth'
    attr_detection_th       = 0.5
#    attr_detection_th       = -5
    
    # AVS-specific parameters
    num_glimpses            = 8*2
    fovea_control_neurons   = 5
    glimpse_size_init       = (20, 20) #(width, height)
    glimpse_size_fixed      = (96, 96) #(width, height)
    glimpse_size_step       = (20, 20) #in (x, y) directions, respectively
    glimpse_change_th       = 0.5
    iou_th                  = 0.5
    discount                = 0.5
