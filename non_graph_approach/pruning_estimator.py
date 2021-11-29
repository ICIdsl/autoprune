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
    def __init__(self, model, trackedModules=None):
        if trackedModules is None:
            self.trackedModules = [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear]
        self.sizeDict = {n:(m, torch.tensor(m.weight.shape)) for n,m in model.named_modules() 
                            if any(isinstance(m,x) for x in self.trackedModules)}

    def pruneSingleFilter(self, layer, connectivity, joinPoints):
        m, size = self.sizeDict[layer] 
        if isinstance(m, torch.nn.Conv2d):
            prunedParams = self.convEffect(m, size, axis=0, bias=(m.bias is not None))
        else:
            raise ValueError(f"Pruning filter from non Conv2d module")

        for l in connectivity[layer]:
            nextM, nextSize = self.sizeDict[l] 
            if not self.removeIcChannelAfterJoin(joinPoints, m, l, nextM):
                continue
            
            if isinstance(nextM, torch.nn.Conv2d):
                prunedParams += self.convEffect(nextM, nextSize, axis=1, 
                                                  bias=(m.bias is not None))
            elif isinstance(nextM, torch.nn.BatchNorm2d):
                prunedParams += self.bnEffect(nextM, nextSize)
            elif isinstance(nextM, torch.nn.Linear):
                prunedParams += self.linearEffect(nextM, nextSize, m)
        
        return prunedParams

    def removeIcChannelAfterJoin(self, joinNodes, currM, nextLayer, nextM):
        for joinType, n in joinNodes.items():
            if any(nextLayer in x for x in n):
                if joinType == 'addJoins':
                    if isinstance(nextM, torch.nn.Conv2d):
                        if currM.out_channels == nextM.in_channels:
                            return False
                    elif isinstance(nextM, torch.nn.BatchNorm2d):
                        if currM.out_channels == nextM.numFeatures:
                            return False
                    elif isinstance(nextM, torch.nn.Linear):
                        if currM.out_channels == nextM.in_features:
                            return False
        return True
    
    def convEffect(self, m, size, axis, bias=False):
        if axis == 0:
            m.out_channels -= 1
        else:
            m.in_channels -= 1
            if m.groups != 1:
                m.groups = m.in_channels
        
        if m.groups == 1:
            size[axis] -= 1
            prunedParams = torch.prod(size[[x for x in range(len(size)) if x != axis]]).item()
        else:
            size[0] -= 1
            prunedParams = torch.prod(size[1:]).item()

        if bias:
            prunedParams += 1
        return prunedParams
    
    def bnEffect(self, m, size):
        size[0] -= 1
        m.num_features -= 1
        # returns 4 as we have weight, bias, runMean, runVar
        return 4
    
    def linearEffect(self, m, size, prevModule):
        spatialDims = int(m.weight.shape[1] / prevModule.weight.shape[0])
        size[1] -= spatialDims
        m.in_features -= spatialDims
        prunedParams = int(spatialDims * size[0])
        return prunedParams
