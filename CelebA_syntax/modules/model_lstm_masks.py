import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import copy
import pdb

class customizable_LSTM(nn.Module):
    def __init__(self, config):
        super(customizable_LSTM, self).__init__()
        # Training configuration
        self.batch_size_train = config.batch_size_train
        self.batch_size_eval = config.batch_size_eval

        # Projection layer configuration
        # self.fc_action = nn.Linear(4, 4096) # keep bias=True
#        self.fc_glimpse = nn.Linear(1024, 1024)
        self.fc_mask = get_linear(4096, 128)

        # LSTM configuration
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.bias = config.bias
        self.batch_first = config.batch_first
        # self.dropout = config.dropout
        self.bidirectional = config.bidirectional
        self.proj_size = config.proj_size # proj_size determines the size of output
        # self.lstm = nn.LSTM(2*config.input_size, config.hidden_size, config.num_layers, config.bias, config.batch_first, config.dropout, config.bidirectional, config.proj_size) # concat
        self.lstm = nn.LSTM(input_size=config.input_size, 
                            hidden_size=config.hidden_size, 
                            num_layers=config.num_layers, 
                            bias=config.bias, 
                            batch_first=config.batch_first, 
                            dropout=config.dropout, 
                            bidirectional=config.bidirectional, 
                            proj_size=config.proj_size)
        # self.fc_projection_1 = get_linear(135, 7)
        # self.fc_projection_2 = get_linear(135, 7)
        
        # 3.23.2023 - tao88: add option to concat embedded mask and semantics
        self.mask_with_semantics = config.mask_with_semantics
        # 3.30.2023 - tao88: dropout in the end
        # self.dropout = nn.Dropout(config.output_dropout)
        # Weight initialization
        if config.init_weights: self._initialize_weights()


    def forward(self, x, hidden): # x is the input to LSTM, here it is the patch mask cropped
        if self.mask_with_semantics:
            x[0] = self.fc_mask(x[0]) # (128, 5, 128), x[1] has shape (128, 5, 7)
            output, hidden = self.lstm(torch.cat((x[0], x[1]), dim=2))
        else:
            # 3.21.2023 - tao88: 
            x = self.fc_mask(x) # (128, 5, 128)
            output, hidden = self.lstm(x, hidden)
        return output, hidden
        
        # if use output projection as fc layers outside lstm
        # output = self.dropout(output)
        # output_1 = self.fc_projection_1(output[:, :, :135])
        # output_2 = self.fc_projection_2(output[:, :, 135:])
        # return torch.cat((output_1, output_2), dim=2), hidden
       

    def _initialize_weights(self):
        for param in self.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    

    def _init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if not self.bidirectional:
            hidden = weight.new(self.num_layers, batch_size, self.proj_size).zero_()
            cell = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
        else:
            hidden = weight.new(self.num_layers * 2, batch_size, self.proj_size).zero_()
            cell = weight.new(self.num_layers * 2, batch_size, self.hidden_size).zero_()
        return hidden, cell
    

def get_linear(indim, outdim):
    fc = nn.Linear(indim, outdim)
    fc.weight.data.normal_(0, 0.01)
    fc.bias.data.zero_()
    return fc