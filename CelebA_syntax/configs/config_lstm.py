class AVS_config(object):
    # SEED
    seed = 99

    # Train configuration (LSTM optimizer and lr_scheduler)
    tally_interval          = 100
    tally_interval_sample   = 50
    num_samples             = 50
    sampling                = False # sampling 50 images every tally_interval_sample batches
    end_sampling            = False # sampling 50 images from the last training batch
    r_as_hidden             = False
    use_contr_loss          = False # True if contrastive loss, False if MSE loss
    batch_size_train        = 128
    batch_size_eval         = 100
    # lr schedule configs
    # start_warmup            = 0
    # warmup_epochs           = 2
    # base_lr                 = 1e-3
    # final_lr                = 1e-6

    lr_start                = 1e-4 # 1e-4 for LSTM, 1e-3 for AE
    lr_min                  = 1e-6
    milestones              = [20]
    gamma                   = 0.1
    weight_decay            = 0.0001
    
    # for projection, quantization, and clustering
    feat_dim                = 1024 # dimension of the feature vectors
    num_prototypes          = 10
    omit_full_glimpse       = True
    pca_reduce              = True
#    nmb_prototypes = [50, 50, 50] # number of prototypes - it can be multihead

    # SWaV configurations
    swav = False # important to keep False for softmax layer to be included in the prediction model
    crops_for_assign = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    nmb_crops = [16]
    temperature = 0.1 # temperature parameter in training loss
    epsilon=0.03 # regularization parameter for Sinkhorn-Knopp algorithm, set smaller to avoid collapsing
    sinkhorn_iterations = 3 # number of iterations in Sinkhorn-Knopp algorithm
    freeze_prototypes_nepochs = 0 # not freeze for now


    # Training and Inference Checkpoint
    initialize_M4 = 'from_checkpoint'
    # ckpt_dir_model_M4 = '/home/nano01/a/tao88/cvpr24_image_semantics/foveation_grammar_detection/bi-lstm_models_crop_masks_and_semantics_lr=0.0001/mse_loss/checkpoint_40.pth.tar'
    ckpt_dir_model_M4 = 'models_trained_on_cropeed_masks/celeba_bilstm_checkpoint_40.pth.tar'
    
    # Inference config
    save_dir_base           = './test_inference/g_approach/feature_based/bi_lstm_approach_1_concat_output_projection/double_contrastive_loss'
    save_action             = False
    use_action              = False
    saved_action_path       = './test_inference/delta_approach/feature_based/ae_leakyrelu/correct_samples/action_array.npy' # where to save and load actions

    # LSTM configuration
    input_size = 135 # 7 or 128 or 135 (semantics or masks or together)
    hidden_size = 135 # uniform with input_size
    num_layers = 1
    bias = True
    batch_first = True
    dropout = 0.2
    output_dropout = 0.2
    bidirectional = True
    proj_size = 7 # if we don't need projections on the hidden states, set proj_size to 0
    # 3.23.2023 - tao88: add option to pass concated embedded mask and semantics to LSTM
    mask_with_semantics = True


    # Initialize weights
    init_weights = True

    # Plot configuration
    beautify_plots = True