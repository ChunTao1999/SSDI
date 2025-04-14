#%% LSTM configuration
class LSTM_config(object):
    # Train and test configs
    epochs = 40
    batch_size_train        = 32
    batch_size_eval         = 128

    # Dataset corruption configs
    all_corrupt             = True # only when doing "puzzle solving"
    corruption_type         = "puzzle_solving" # choose among ["patch_shuffling", "puzzle_solving", "black_box", "gaussian_blurring"]
    patch_size              = 80 # 160 or 80 when input image is 640x480
    num_distortion          = 16 # number of patches that will be permuted, used in "patch_shuffling"
    num_permute             = 3 # number of different permutations that will be created and compared against grouth truth, used in "puzzle_solving". Each permutation is a shuffling of all patches
    num_box                 = 4 # number of patches that will be made black boxes or blurred boxes, used in "black_box" and "gaussian_blurring"

    # LSTM architecture configs
    mask_dim = patch_size*patch_size
    latent_dim = 128
    semantics_dim = 13 # 13 or 37
    input_size = latent_dim+semantics_dim # 128+14 concatted
    hidden_size = input_size # uniform with input_size
    num_layers = 1
    bias = True
    batch_first = True
    dropout = 0.2 # no effect when num_layers=1, unless append external fc layer
    bidirectional = True
    proj_size = semantics_dim # if no output projection, set to 0
    mask_with_semantics = True

    # LSTM optimizer configs
    lr_start = 1e-4
    weight_decay = 0.0001
    # LSTM lr scheduler configs
    lr_min = 1e-6
    milestones = [5, 10, 15, 20, 25, 30, 35] # 1e-4 till 20th epoch, then 1e-5 till 40th
    gamma = 0.8

    # Initialize weights
    init_weights = True

    # Pretrained
    # from_pretrained = True # False when train, True when test
    ckpt_dir_model_M4 = "/home/nano01/a/tao88/cvpr24_image_semantics/RedNet/LSTM_models_trained_on_RedNet_generated_labels/num_classes={}/input_size=640x480/ps={}_bi-lstm_numlayers=1_startlr=0.0001_epoch=40/checkpoint_40.pth.tar".format(semantics_dim, patch_size)