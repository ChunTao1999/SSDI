import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import copy

class customizable_LSTM(nn.Module):
    def __init__(self, config):
        super(customizable_LSTM, self).__init__()
        
        # Mask projection layer config
        self.fc_mask = get_linear(config.mask_dim, config.latent_dim) # (4096, 128)
        self.fc_mask_1 = get_linear(config.mask_dim, 1024)
        self.fc_mask_2 = get_linear(1024, config.latent_dim)
        self.semantics_dim = config.semantics_dim

        # LSTM layer config
        self.input_size = config.input_size # 135
        self.hidden_size = config.hidden_size # 135
        self.num_layers = config.num_layers # 1
        self.bias = config.bias # True
        self.batch_first = config.batch_first # True
        self.bidirectional = config.bidirectional # True
        self.proj_size = config.proj_size # 7
        self.mask_with_semantics = config.mask_with_semantics # True
        self.lstm = nn.LSTM(input_size=config.input_size, 
                            hidden_size=config.hidden_size, 
                            num_layers=config.num_layers, 
                            bias=config.bias, 
                            batch_first=config.batch_first, 
                            dropout=config.dropout, 
                            bidirectional=config.bidirectional, 
                            proj_size=config.proj_size)
        
        # Weight init
        if config.init_weights: self._initialize_weights()
    

    def forward(self, x, hidden): 
        # x[0] is mask (bs, num_patches, mask_dim), x[1] is semantics (bs, num_patches, semantics_dim)
        # or x is mask (bs, num_patches, mask_dim)
        if self.mask_with_semantics:
            # x[0] = self.fc_mask_1(x[0])
            # x[0] = self.fc_mask_2(x[0])
            x[0] = self.fc_mask(x[0])
            output, hidden = self.lstm(torch.cat((x[0], x[1]), dim=2))
        else:
            x = self.fc_mask(x)
            output, hidden = self.lstm(x, hidden)
        return output, hidden


    def _initialize_weights(self):
        for param in self.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    

    def _init_hidden(self, batch_size):
        '''Function to initialize hidden and cell states'''
        weight = next(self.parameters()).data
        if self.bidirectional:
            hidden = weight.new(self.num_layers * 2, batch_size, self.proj_size).zero_()
            cell = weight.new(self.num_layers * 2, batch_size, self.hidden_size).zero_()
        else:
            hidden = weight.new(self.num_layers, batch_size, self.proj_size).zero_()
            cell = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
        return hidden, cell


def get_linear(indim, outdim):
    fc = nn.Linear(indim, outdim)
    fc.weight.data.normal_(0, 0.01)
    fc.bias.data.zero_()
    return fc
    

#%% LSTM configuration
class LSTM_config(object):
    # Train and test configs
    epochs                  = 40
    patch_size              = 80 # 160 or 80 when input image is 640x480
    # 4.26.2023 - tao88: corruption configs
    all_corrupt             = False # only when doing "puzzle solving"
    corruption_type         = "black_box" # choose among ["patch_shuffling", "puzzle_solving", "black_box", "gaussian_blurring"]
    num_distortion          = 4 # number of patches that will be permuted, used in "patch_shuffling"
    num_permute             = 3 # number of different permutations that will be created and compared against grouth truth, used in "puzzle_solving". Each permutation is a shuffling of all patches
    num_box                 = 16 # number of patches that will be made black boxes or blurred boxes, used in "black_box" and "gaussian_blurring"

    # LSTM architecture configs
    mask_dim = patch_size*patch_size
    latent_dim = 128
    semantics_dim = 14 # 14 or 38
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
    ckpt_dir_model_M4 = "/content/drive/MyDrive/LSTM_models_trained_on_sam2_generated_labels/num_classes={}/input_size=640x480/ps={}_bi-lstm_numlayers=1_startlr=0.0001_epoch=40/checkpoint_40.pth.tar".format(semantics_dim, patch_size)


