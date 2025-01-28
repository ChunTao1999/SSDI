
"""
Verified on May 25 2022
"""

class AVS_config(object):
    # SEED
    seed                    = 7
    
    # dataset
    dataset                 = 'celeba'
    if dataset == 'birds':
        dataset_dir         = '/path/to/CUB_200-2011/dataset/'
        num_classes         = 200
        in_num_channels     = 3
        full_res_img_size   = (256, 256)
    elif dataset == 'celeba':
        dataset_dir                 = '/home/nano01/a/tao88/celebA_raw'
        in_num_channels             = 3
        full_res_img_size           = (256, 256)
        num_classes                 = 40    # 40 binary attributes in celebA
        selected_attributes         = 'all'
        correct_imbalance           = False
        at_least_true_attributes    = 0
        treat_attributes_as_classes = False
    else:
        raise ValueError("The script is not implemented for {} dataset!".format(dataset))

    # model
    low_res  = 64
    avg_size = 21

    # training
    train_loader_type       = 'trainval'
    if train_loader_type == 'train':
        valid_loader_type   = 'valid'
    elif train_loader_type == 'trainval':
        valid_loader_type   = 'test'
        print("Warning: selected training on trainval split, hence validation is going to be performed on test split!")
    else:
        raise ValueError("Unrecognized type of split to train on: ({})".format(train_loader_type))

    experiment_name         = (dataset + '/trained_on_{}_split/train_M2/'.format(train_loader_type))
    save_dir                = './results_new/' + experiment_name
    batch_size_train        = 128
    batch_size_eval         = 100
    epochs                  = 100
    lr_start                = 1e-2
    lr_min                  = 1e-4
    milestones              = [50, 75]
    weight_decay            = 0
    
    # testing
    ckpt_dir                = save_dir + 'model.pth'
    
