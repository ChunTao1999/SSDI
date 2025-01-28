import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import copy

class customizable_LSTM(nn.Module):
    def __init__(self, config):
        super(customizable_LSTM, self).__init__()
        # Training configuration
        self.batch_size_train = config.batch_size_train
        self.batch_size_eval = config.batch_size_eval

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