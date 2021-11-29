import os
import sys
import logging

import torch
import numpy as np

from ..utils import * 

def l1Norm(model, ignore=None):
    layers = []
    globalRanking = []
    for n,m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            p = m.weight.data.cpu().numpy()
            metric = np.absolute(p).reshape(p.shape[0], -1).sum(axis=1)
            metric /= (p.shape[1]*p.shape[2]*p.shape[3])
            
            if not layerToIgnore(n, ignore):
                globalRanking += [(n, i, x) for i,x in enumerate(metric)]
    
    globalRanking = sorted(globalRanking, key=lambda i: i[2]) 
    return globalRanking

