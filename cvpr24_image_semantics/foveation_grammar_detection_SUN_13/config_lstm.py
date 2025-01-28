#%% LSTM configuration
class LSTM_config(object):
    # train and test configs
    epochs = 40
    batch_size_train        = 128
    batch_size_eval         = 100
    patch_size              = 64

    # LSTM architecture configs
    mask_dim = patch_size*patch_size
    latent_dim = 128
    semantics_dim = 14
    input_size = 142
    hidden_size = 142 # uniform with input_size
    num_layers = 1
    bias = True
    batch_first = True
    dropout = 0.2 # no effect when num_layers=1, unless append fc layer
    bidirectional = True
    proj_size = 14 # if no output projection, set to 0
    mask_with_semantics = True

    # LSTM optimizer configs
    lr_start = 1e-4
    weight_decay = 0.0001
    # LSTM lr scheduler configs
    lr_min = 1e-6
    milestones = [20] # 1e-4 till 20th epoch, then 1e-5 till 40th
    gamma = 0.1

    # Initialize weights
    init_weights = True

    # Pretrained
    from_pretrained = True # False when train, True when test
    ckpt_dir_model_M4 = "./ps=64_bi-lstm_numlayers=1_startlr=0.0001_epoch=40/checkpoint_40.pth.tar"