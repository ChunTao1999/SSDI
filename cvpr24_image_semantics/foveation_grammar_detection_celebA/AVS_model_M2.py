
"""
Verified on May 25 2022
"""

import torch
import torch.nn as nn
import copy
from evonorm import EvoNormSample2d as enorm
#from AVS_config_a1 import config

class context_network(nn.Module):
    """ context network: provides a good starting location"""
    def __init__(self, in_channels=3, res_size = 32, avg_size=7, device='cuda', out_dim=2):
        super(context_network, self).__init__() 
      
        self.conv1   = nn.Conv2d(in_channels, 16, 7, stride=2)
        self.bn1     = enorm(16)
        self.conv2   = nn.Conv2d(16, 32, 5, stride=1)
        self.bn2     = enorm(32)
        self.conv3   = nn.Conv2d(32, 64, 3, stride=1)
        self.bn3     = enorm(64)
        self.conv4   = nn.Conv2d(64, 64, 3, stride=1)
        self.bn4     = enorm(64)
        self.relu    = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(avg_size, stride=1)
        self.linear  = nn.Linear(64, out_dim)
        self.tanh    = nn.Tanh()
        self.low_res = res_size
        self.device  = device
        
    def forward(self, inp):
        coarse_inp = torch.nn.functional.interpolate(inp, [self.low_res, self.low_res], mode='bilinear', align_corners=False)
        #print(inp.shape, coarse_inp.shape)
        out = self.relu(self.conv1(coarse_inp.detach()))
        out = self.bn1(out)
        out = self.relu(self.conv2(out))
        out = self.bn2(out)
        out = self.relu(self.conv3(out))
        out = self.bn3(out)
        out = self.relu(self.conv4(out))
        out = self.bn4(out)
        #print(out.shape)
        out = self.avgpool(out)
        #print(out.shape)
        out  = out.view(out.size(0), -1)
        mean = self.tanh(self.linear(out))
        return mean
  
def test():
    model = context_network(res_size=32)
    x = torch.rand(32, 3, 200, 200)
    y,_ = model(x)
              

if __name__ == '__main__':
    test()