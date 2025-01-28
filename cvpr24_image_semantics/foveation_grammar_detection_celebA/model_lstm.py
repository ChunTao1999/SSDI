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
        # self.fc_action = nn.Linear(4, 1024) # keep bias=True
#        self.fc_glimpse = nn.Linear(1024, 1024)

        # LSTM configuration
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.bias = config.bias
        self.batch_first = config.batch_first
        self.dropout = config.dropout
        self.bidirectional = config.bidirectional
        self.proj_size = config.proj_size # proj_size determines the size of output
        # self.lstm = nn.LSTM(2*config.input_size, config.hidden_size, config.num_layers, config.bias, config.batch_first, config.dropout, config.bidirectional, config.proj_size) # concat
        self.lstm = nn.LSTM(config.input_size, config.hidden_size, config.num_layers, config.bias, config.batch_first, config.dropout, config.bidirectional, config.proj_size) # action appended to input
        
        # Weight initialization
        if config.init_weights: self._initialize_weights()

    def forward(self, x, hidden): # may need a more complicated model
#         pdb.set_trace()
        # # Approch 1:
        # x[1] = F.relu(self.fc_action(x[1]), inplace=True)
        # x[1] = torch.unsqueeze(x[1], 1)
        # x[0] = F.relu(x[0], inplace=True)
        # x = F.relu(x[0] + x[1], inplace=True) 
    
        # Approach 1_concat:
        # x[1] = F.relu(self.fc_action(x[1]), inplace=True)
        # x[1] = torch.unsqueeze(x[1], 1)
        # x[0] = F.relu(x[0], inplace=True)
        # x = F.relu(torch.concat((x[0], x[1]), 2), inplace=True) # 2048-d
        
        # Approach 2:
#        x[1] = self.fc_action(x[1])
#        x[1] = torch.unsqueeze(x[1], 1)
#        x = F.relu(x[0] + x[1], inplace=True)

        # Approach 3:
        # project both x[0] and x[1] to 1024-d or lower space
#        x[1] = self.fc_action(x[1])
#        x[1] = torch.unsqueeze(x[1], 1)
#        x[0] = self.fc_glimpse(x[0])
#        x = F.relu(x[0] + x[1], inplace=True)
        
        output, hidden = self.lstm(x, hidden)
        return output, hidden
       

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