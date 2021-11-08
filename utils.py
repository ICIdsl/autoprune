import os
import sys
import copy
import time
import random
import logging
import subprocess
from tqdm import tqdm

import torch

class PathTruncatingFormatter(logging.Formatter):
    def format(self, record): 
        filename = record.pathname.split('autoprune/')[-1]
        record.pathname = filename
        return super(PathTruncatingFormatter, self).format(record)

def initialise_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)
    syslog = logging.StreamHandler(sys.stdout)
    formatter = PathTruncatingFormatter('[%(pathname)s:%(lineno)d] %(levelname)s : %(message)s')
    syslog.setFormatter(formatter)
    logger.addHandler(syslog)

def layer_to_ignore(layer_name, ignore_layers):
    if ignore_layers is None:
        return False 
    
    outcomes = []
    for x in ignore_layers:
        cond = x.split('::')[0]
        kw = x.split('::')[1]
        if cond == 'is':
            outcomes.append(kw == layer_name)
        elif cond == 'contains':
            outcomes.append(kw in layer_name)
        else:
            raise ArgumentError(f"Condition {cond} not addressed.")
    return any(outcomes)

def reshape_tensor(tensor, filters, axis):
    if axis == 0:
        new_tensor = tensor[filters] 
    elif axis == 1:
        new_tensor = tensor[:,filters]
    return new_tensor

def reshape_conv_layer(m, filters, ofm):
    axis = 0 if ofm else 1 
    if ofm:
        if m.weight.shape[axis] == len(filters):
            return
    else:
        if m.weight.shape[axis] == len(filters):
            return
    
    new_weight = reshape_tensor(m.weight, filters, axis) 
    m.weight = torch.nn.Parameter(new_weight)
    if m.bias is not None:
        new_bias = reshape_tensor(m.bias, filters, axis) 
        m.bias = torch.nn.Parameter(new_bias)

def reshape_bn_layer(m, filters):
    new_weight = reshape_tensor(m.weight, filters, axis=0) 
    m.weight = torch.nn.Parameter(new_weight)
    new_bias = reshape_tensor(m.bias, filters, axis=0)
    m.bias = torch.nn.Parameter(new_bias)
    m.running_mean = reshape_tensor(m.running_mean, filters, axis=0)
    m.running_var = reshape_tensor(m.running_var, filters, axis=0)

def reshape_linear_layer(m, prev_m, filters):
    spatial_dims = int(m.in_features / prev_m.out_channels)
    if m.weight.shape[1] == len(filters) * spatial_dims:
        return
    _filters = [x for y in filters for x in range(y,y+spatial_dims)]
    new_weight = reshape_tensor(m.weight, _filters, axis=1) 
    m.weight = torch.nn.Parameter(new_weight)

