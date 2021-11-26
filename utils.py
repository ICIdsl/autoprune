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

def initialiseLogging():
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)
    syslog = logging.StreamHandler(sys.stdout)
    formatter = PathTruncatingFormatter('[%(pathname)s:%(lineno)d] %(levelname)s : %(message)s')
    syslog.setFormatter(formatter)
    logger.addHandler(syslog)

def layerToIgnore(layerName, ignoreLayers):
    if ignoreLayers is None:
        return False 
    
    outcomes = []
    for x in ignoreLayers:
        cond = x.split('::')[0]
        kw = x.split('::')[1]
        if cond == 'is':
            outcomes.append(kw == layerName)
        elif cond == 'contains':
            outcomes.append(kw in layerName)
        else:
            raise ArgumentError(f"Condition {cond} not addressed.")
    return any(outcomes)

def reshapeTensor(tensor, filters, axis):
    if axis == 0:
        newTensor = tensor[filters] 
    elif axis == 1:
        newTensor = tensor[:,filters]
    return newTensor

def reshapeConvLayer(m, filters, ofm):
    axis = 0 if ofm else 1 
    if ofm:
        if m.weight.shape[axis] == len(filters):
            return
    else:
        if m.weight.shape[axis] == len(filters):
            return
    
    newWeight = reshapeTensor(m.weight, filters, axis) 
    m.weight = torch.nn.Parameter(newWeight)
    if m.bias is not None:
        newBias = reshapeTensor(m.bias, filters, axis) 
        m.bias = torch.nn.Parameter(newBias)

def reshapeBnLayer(m, filters):
    newWeight = reshapeTensor(m.weight, filters, axis=0) 
    m.weight = torch.nn.Parameter(newWeight)
    newBias = reshapeTensor(m.bias, filters, axis=0)
    m.bias = torch.nn.Parameter(newBias)
    m.runningMean = reshapeTensor(m.runningMean, filters, axis=0)
    m.runningVar = reshapeTensor(m.runningVar, filters, axis=0)

def reshapeLinearLayer(m, prevM, filters):
    spatialDims = int(m.inFeatures / prevM.outChannels)
    if m.weight.shape[1] == len(filters) * spatialDims:
        return
    _filters = [x for y in filters for x in range(y,y+spatialDims)]
    newWeight = reshapeTensor(m.weight, _filters, axis=1) 
    m.weight = torch.nn.Parameter(newWeight)

