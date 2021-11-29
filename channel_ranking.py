import os
import sys
import logging

import torch
import numpy as np

from .utils import * 

def l1_norm(model, ignore=None):
    localRanking = {} 
    globalRanking = []

    # create global ranking
    layers = []
    for n,m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            p = m.weight.data.cpu().numpy()
            metric = np.absolute(p).reshape(p.shape[0], -1).sum(axis=1)
            metric /= (p.shape[1]*p.shape[2]*p.shape[3])
            
            localRanking[n] = sorted([(i, x) for i,x in enumerate(metric)], key=lambda tup:tup[1])

            if not layerToIgnore(n, ignore):
                globalRanking += [(n, i, x) for i,x in enumerate(metric)]

    globalRanking = sorted(globalRanking, key=lambda i: i[2]) 
    
    return localRanking, globalRanking

