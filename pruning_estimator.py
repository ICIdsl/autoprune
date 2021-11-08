import os
import sys
import copy
import glob
import time
import math
import random
import logging
import subprocess
from tqdm import tqdm

import pickle
import numpy as np
import pandas as pd

import torch

class NetworkSizeTracker():
    def __init__(self, model, tracked_modules=None):
        if tracked_modules is None:
            self.tracked_modules = [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear]
        self.size_dict = {n:(m, torch.tensor(m.weight.shape)) for n,m in model.named_modules() 
                            if any(isinstance(m,x) for x in self.tracked_modules)}

    def prune_single_filter(self, layer, connectivity, join_points):
        m, size = self.size_dict[layer] 
        if isinstance(m, torch.nn.Conv2d):
            pruned_params = self.conv_effect(m, size, axis=0, bias=(m.bias is not None))
        else:
            raise ValueError(f"Pruning filter from non Conv2d module")

        for l in connectivity[layer]:
            next_m, next_size = self.size_dict[l] 
            if not self.remove_ic_channel_after_join(join_points, m, l, next_m):
                continue
            
            if isinstance(next_m, torch.nn.Conv2d):
                pruned_params += self.conv_effect(next_m, next_size, axis=1, 
                                                  bias=(m.bias is not None))
            elif isinstance(next_m, torch.nn.BatchNorm2d):
                pruned_params += self.bn_effect(next_m, next_size)
            elif isinstance(next_m, torch.nn.Linear):
                pruned_params += self.linear_effect(next_m, next_size, m)
        
        return pruned_params

    def remove_ic_channel_after_join(self, join_nodes, curr_m, next_layer, next_m):
        for join_type, n in join_nodes.items():
            if any(next_layer in x for x in n):
                if join_type == 'aten::add_':
                    if isinstance(next_m, torch.nn.Conv2d):
                        if curr_m.out_channels == next_m.in_channels:
                            return False
                    elif isinstance(next_m, torch.nn.BatchNorm2d):
                        if curr_m.out_channels == next_m.num_features:
                            return False
                    elif isinstance(next_m, torch.nn.Linear):
                        if curr_m.out_channels == next_m.in_features:
                            return False
        return True
    
    def conv_effect(self, m, size, axis, bias=False):
        if axis == 0:
            m.out_channels -= 1
        else:
            m.in_channels -= 1
            if m.groups != 1:
                m.groups = m.in_channels
        
        size[axis] -= 1
        pruned_params = torch.prod(size[[x for x in range(len(size)) if x != axis]]).item()
        if bias:
            pruned_params += 1
        return pruned_params
    
    def bn_effect(self, m, size):
        size[0] -= 1
        m.num_features -= 1
        # returns 4 as we have weight, bias, run_mean, run_var
        return 4
    
    def linear_effect(self, m, size, prev_module):
        spatial_dims = int(m.weight.shape[1] / prev_module.weight.shape[0])
        size[1] -= spatial_dims
        m.in_features -= spatial_dims
        pruned_params = int(spatial_dims * size[0])
        return pruned_params

