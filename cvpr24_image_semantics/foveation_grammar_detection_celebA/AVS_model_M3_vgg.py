#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:55:07 2020
Verified on Wed May 25 2022

@modified by: tibrayev
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from evonorm import EvoNormSample2d as enorm
import pdb

# 'D' - stands for downsampling, for which choices are: 'M' - max pooling, 'A' - average pooling, 'C' - strided convolution
cfgs = {
    # first, custom ones, not present in original VGG paper:
    'vgg6_narrow' :     [16, 'D', 32,  'D', 32,  'D'],
    'vgg6' :            [64, 'D', 128, 'D', 128, 'D'],
    'vgg9' :            [64, 'D', 128, 'D', 256, 256, 'D', 512, 512, 'D'],
    'vgg8_narrow_k4':   [16, 'D', 32,  'D', 64,  'D', 128, 'D', 128],
    'vgg8_narrow_k2':   [32, 'D', 64,  'D', 128, 'D', 256, 'D', 256],
    'vgg11_narrow_k4':  [16, 'D', 32,  'D', 64,  64,  'D', 128, 128, 'D', 128, 128],
    'vgg11_narrow_k2':  [32, 'D', 64,  'D', 128, 128, 'D', 256, 256, 'D', 256, 256],
    
    # next, default ones under VGG umbrella term:
    'vgg11': [64, 'D', 128, 'D', 256, 256, 'D', 512, 512, 'D', 512, 512],
    'vgg13': [64, 64, 'D', 128, 128, 'D', 256, 256, 'D', 512, 512, 'D', 512, 512],
    'vgg16': [64, 64, 'D', 128, 128, 'D', 256, 256, 256, 'D', 512, 512, 512, 'D', 512, 512, 512],
    'vgg19': [64, 64, 'D', 128, 128, 'D', 256, 256, 256, 256, 'D', 512, 512, 512, 512, 'D', 512, 512, 512, 512],
}

class customizable_VGG(nn.Module):
    def __init__(self, config):
        super(customizable_VGG, self).__init__()

        # Dataset configuration
        self.dataset        = config.dataset
        self.num_classes    = config.num_classes
        self.in_channels    = config.in_num_channels
        self.in_feat_dim    = config.glimpse_size_fixed if isinstance(config.glimpse_size_fixed, tuple) else (config.glimpse_size_fixed, config.glimpse_size_fixed)
        self.data_parallel = config.data_parallel
        self.num_devices = config.num_devices


        # Network configuration
        self.vgg_name       = config.vgg_name.lower()
        if config.downsampling in ['M', 'A', 'C']: 
            self.downsampling   = config.downsampling
        else:
            raise ValueError("Error: Unknown downsampling. Choices are 'M' - max pooling, 'A' - average pooling, 'C' - strided convolution!") 
        self.fc1            = config.fc1
        self.fc2            = config.fc2
        self.dropout        = config.dropout
        self.norm           = config.norm
        self.feat_avg_pool  = config.adaptive_avg_pool_out if isinstance(config.adaptive_avg_pool_out, tuple) else (config.adaptive_avg_pool_out, config.adaptive_avg_pool_out)
        # #@Version 1:
        # self.int_to_action  = {0:torch.Tensor([0,0,0,0]), 1:torch.Tensor([0,0,0,1]), 2:torch.Tensor([0,0,1,0]), 3:torch.Tensor([0,0,1,1]), 
        #                        4:torch.Tensor([0,1,0,0]), 5:torch.Tensor([0,1,0,1]), 6:torch.Tensor([0,1,1,0]), 7:torch.Tensor([0,1,1,1]), 
        #                        8:torch.Tensor([1,0,0,0]), 9:torch.Tensor([1,0,0,1]), 10:torch.Tensor([1,0,1,0]), 11:torch.Tensor([1,0,1,1]), 
        #                        12:torch.Tensor([1,1,0,0]), 13:torch.Tensor([1,1,0,1]), 14:torch.Tensor([1,1,1,0]), 15:torch.Tensor([1,1,1,1])}

        #@Version 6:
        self.int_to_action  = {0:torch.Tensor([0,0,0,0]), 1:torch.Tensor([0,0,0,1]), 2:torch.Tensor([0,0,1,0]), 3:torch.Tensor([0,1,0,0]), 4:torch.Tensor([1,0,0,0])}

        # Creating layers
        ### Feature extraction
        self.features, feature_channels, feature_dim    = self._make_feature_layers(cfgs[self.vgg_name])
        ### Classifier
        self.classifier                                 = self._make_classifier_layers(feature_channels, feature_dim)
        ### Fovea control
        self.fovea_control_neurons                      = config.fovea_control_neurons
        self.RLfeat, feature_channels, feature_dim      = self._make_RLfeature_layers(cfgs['vgg8_narrow_k4']) #@Version 7
        self.fovea_control                              = self._make_fovea_layers(feature_channels, feature_dim, self.fovea_control_neurons) #@Version 1
        self.fovea_control_baseline                     = self._make_fovea_layers(feature_channels, feature_dim, 1) #@Version 1
        
        # Weight initialization
        if config.init_weights: self._initialize_weights()
        
        # RL storage initialization
        self.reward_memory  = []
        self.action_memory  = []
        self.baseline       = []
        self.discount       = config.discount


    def forward(self, x, with_latent = False, sampled_actions = True, rl_actions = True):
        x_features          = self.features(x)
        features            = torch.flatten(x_features, 1)
        ### classifier predictions
        outputs             = self.classifier(features)
        if not rl_actions: # only want the features, not using rl for action prediction
            return features
    
        ### RL actions predictions
        RLfeatures          = self.RLfeat(x)
        RLfeatures_flat     = torch.flatten(RLfeatures, 1)
        actions             = self.fovea_control(RLfeatures_flat)
        baseline            = self.fovea_control_baseline(RLfeatures_flat)
        #@Version 6:
        if sampled_actions:
            probabilities       = F.softmax(actions, dim=1)
            actions_dist        = torch.distributions.Categorical(probabilities)
            actions_taken       = actions_dist.sample()
            log_probabilities   = actions_dist.log_prob(actions_taken)
            glimpses            = self.get_glimpse_change(actions_taken).detach().to(x.device)
            # Store actions
            self.action_memory.append(log_probabilities)
            self.baseline.append(baseline)
        else:
            actions_taken       = F.softmax(actions, dim=1)
            glimpses            = self.get_glimpse_change(actions_taken).detach().to(x.device)
            # Store actions
            self.action_memory.append(actions_taken)
            self.baseline.append(baseline)
        
        # Outputs
        if with_latent:
            return outputs, glimpses, features
        else:
            return outputs, glimpses

    def get_glimpse_change(self, actions_taken):
        return torch.stack([self.int_to_action[a.item()] for a in actions_taken], dim=0)
        
    def store_rewards(self, reward):
        self.reward_memory.append(reward)
        
    def store_all_rewards(self, rewards):
        self.reward_memory  = [r for r in rewards]
        
    def clear_memory(self):
        self.action_memory  = []
        self.reward_memory  = []
        self.baseline       = []
        
    def compute_rl_loss(self):
        if self.data_parallel: # 2.15.2023 - tao88, when using dataparallel (abandon the last batch in dataset)
            self.baseline_new = []
            # pdb.set_trace()
            self.baseline = [b.to("cuda:0") for b in self.baseline] # 64(or less)*(less than 36)*1
            for g in range(len(self.reward_memory)):
                self.baseline_new.append(torch.cat(self.baseline[g*self.num_devices:(g+1)*self.num_devices], dim=0))
            baseline = torch.stack(self.baseline_new[1:]).squeeze()
        else:
            baseline = torch.stack(self.baseline[1:]).squeeze()

        rewards             = torch.stack(self.reward_memory[1:])

        if self.discount:
            for g in range(rewards.size(0)-1):
                rewards[g] += sum([rewards[g+j]*(self.discount**(j)) for j in range(1, rewards.size(0)-g)])

        if self.data_parallel: # 2.15.2023 - tao88, when using dataparallel
            self.action_memory_new = []
            self.action_memory = [a.to("cuda:0") for a in self.action_memory] 
            for g in range(len(self.reward_memory)):
                self.action_memory_new.append(torch.cat(self.action_memory[g*self.num_devices:(g+1)*self.num_devices], dim=0))
            G = torch.stack(self.action_memory_new[:-1])
        else:
            G = torch.stack(self.action_memory[:-1])

        logllratio          = -G*(rewards - baseline.detach())
        baseline_mse        = F.mse_loss(baseline, rewards)
        loss                = torch.mean(logllratio) + torch.mean(baseline_mse)
        return loss



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, enorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def _make_feature_layers(self, cfg):
        layers = []
        in_channels     = copy.deepcopy(self.in_channels)
        feature_dim     = copy.deepcopy(self.in_feat_dim)
        
        for v in cfg:
            if v == 'D':
                if self.downsampling == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                elif self.downsampling == 'A':
                    layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                elif self.downsampling == 'C':
                    layers += [nn.Conv2d(kernel_size=2, stride=2, bias=False)]
                feature_dim = tuple(f//2 for f in feature_dim)
                
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.norm is None:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                elif self.norm.lower() == 'batchnorm':
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                elif self.norm.lower() == 'evonorm':
                    layers += [conv2d, enorm(v), nn.ReLU(inplace=True)]
                else:
                    raise ValueError("Received unknown type of normalization layer {}. Allowed types are: (None, 'batchnorm', 'evonorm').".format(self.norm))
                in_channels = v
        
        layers += [nn.AdaptiveAvgPool2d(self.feat_avg_pool)]
        if (feature_dim[0]%self.feat_avg_pool[0] != 0) or (feature_dim[1]%self.feat_avg_pool[1] != 0):
            print("Warning! Expected feature size map is {}, but adaptive average pooling output requested is {},\n".format(feature_dim, self.feat_avg_pool) +
                  "meaning that some portion of the feature map will be dropped due to mismatch.")
            print("Consider changing the size of input {} or the size of adaptive average pooling output {} to process entire feature map!".format(self.in_feat_dim, self.feat_avg_pool))
        feature_dim = self.feat_avg_pool
        
        return nn.Sequential(*layers), in_channels, feature_dim


    def _make_classifier_layers(self, feature_channels, feature_dim) :
        layers = []
        feature_flat_dims = feature_channels*feature_dim[0]*feature_dim[1]
        
        if self.fc1 == 0 and self.fc2 == 0:
            layers += [nn.Linear(feature_flat_dims, self.num_classes)]
        elif self.fc1 == 0 and self.fc2 != 0:
            raise ValueError("Received ambiguous pair of classifier parameters: fc1 = 0, but fc2 = {}. ".format(self.fc2) + 
                             "If only two FC layers are needed (including last linear classifier), please specify its dims as fc1 and set fc2=0.")
        elif self.fc1 != 0 and self.fc2 == 0:
            layers += [nn.Linear(feature_flat_dims, self.fc1)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(self.dropout)]
            layers += [nn.Linear(self.fc1, self.num_classes)]   
        else:
            layers += [nn.Linear(feature_flat_dims, self.fc1)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(self.dropout)]
            layers += [nn.Linear(self.fc1, self.fc2)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(self.dropout)]
            layers += [nn.Linear(self.fc2, self.num_classes)]        
        return nn.Sequential(*layers)    



    def _make_RLfeature_layers(self, cfg):
        layers = []
        in_channels     = copy.deepcopy(self.in_channels)
        feature_dim     = copy.deepcopy(self.in_feat_dim)
        
        for v in cfg:
            if v == 'D':
                if self.downsampling == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                elif self.downsampling == 'A':
                    layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                elif self.downsampling == 'C':
                    layers += [nn.Conv2d(kernel_size=2, stride=2, bias=False)]
                feature_dim = tuple(f//2 for f in feature_dim)
                
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.norm is None:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                elif self.norm.lower() == 'batchnorm':
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                elif self.norm.lower() == 'evonorm':
                    layers += [conv2d, enorm(v), nn.ReLU(inplace=True)]
                else:
                    raise ValueError("Received unknown type of normalization layer {}. Allowed types are: (None, 'batchnorm', 'evonorm').".format(self.norm))
                in_channels = v
        
        layers += [nn.AdaptiveAvgPool2d(self.feat_avg_pool)]
        if (feature_dim[0]%self.feat_avg_pool[0] != 0) or (feature_dim[1]%self.feat_avg_pool[1] != 0):
            print("Warning! Expected feature size map is {}, but adaptive average pooling output requested is {},\n".format(feature_dim, self.feat_avg_pool) +
                  "meaning that some portion of the feature map will be dropped due to mismatch.")
            print("Consider changing the size of input {} or the size of adaptive average pooling output {} to process entire feature map!".format(self.in_feat_dim, self.feat_avg_pool))
        feature_dim = self.feat_avg_pool
        
        return nn.Sequential(*layers), in_channels, feature_dim


    def _make_fovea_layers(self, feature_channels, feature_dim, output_nodes) :
        layers = []
        feature_flat_dims = feature_channels*feature_dim[0]*feature_dim[1]
        
        if self.fc1 == 0 and self.fc2 == 0:
            layers += [nn.Linear(feature_flat_dims, output_nodes)]
        elif self.fc1 == 0 and self.fc2 != 0:
            raise ValueError("Received ambiguous pair of classifier parameters: fc1 = 0, but fc2 = {}. ".format(self.fc2) + 
                             "If only two FC layers are needed (including last linear classifier), please specify its dims as fc1 and set fc2=0.")
        elif self.fc1 != 0 and self.fc2 == 0:
            layers += [nn.Linear(feature_flat_dims, self.fc1)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Linear(self.fc1, output_nodes)]   
        else:
            layers += [nn.Linear(feature_flat_dims, self.fc1)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Linear(self.fc1, self.fc2)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Linear(self.fc2, output_nodes)]        
        return nn.Sequential(*layers)  


def sim(vec_1, vec_2, temperature=1.0):
    # or, just use torch.cosine_similarity
    vec_1_normalized    = vec_1/(vec_1.norm(p=2, dim=-1, keepdim=True))
    vec_2_normalized    = vec_2/(vec_2.norm(p=2, dim=-1, keepdim=True))
    sim = torch.sum(vec_1_normalized * vec_2_normalized, dim=-1) / temperature
    return sim


def test():
    for d in ['MNIsT', 'FashionMnISt', 'Cifar10', 'CiFAR100']:
        for a in cfgs.keys():
            print("test under {} for {}".format(d.lower(), a))
            model = customizable_VGG(dataset=d, vgg_name=a)
            print("feature_flat_dims: {}".format(model.classifier[0].in_features))
            if d == 'MNIsT' or d == 'FashionMnISt':
                x = torch.rand(2, 1, 28, 28)
                y = model(x)
                assert y.shape == (2, 10)
            elif d == 'Cifar10':
                x = torch.rand(5, 3, 32, 32)
                y = model(x)
                assert y.shape == (5, 10)
            elif d == 'CiFAR100':
                x = torch.rand(3, 3, 32, 32)
                y = model(x)
                assert y.shape == (3, 100)

if __name__ == '__main__':
    test()
            