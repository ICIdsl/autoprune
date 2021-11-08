import os
import sys
import logging

import torch
import numpy as np

import utils

def l1_norm(model, ignore=None):
    local_ranking = {} 
    global_ranking = []

    # create global ranking
    layers = []
    for n,m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            p = m.weight.data.cpu().numpy()
            metric = np.absolute(p).reshape(p.shape[0], -1).sum(axis=1)
            metric /= (p.shape[1]*p.shape[2]*p.shape[3])
            
            local_ranking[n] = sorted([(i, x) for i,x in enumerate(metric)], key=lambda tup:tup[1])

            if not utils.layer_to_ignore(n, ignore):
                global_ranking += [(n, i, x) for i,x in enumerate(metric)]

    global_ranking = sorted(global_ranking, key=lambda i: i[2]) 
    
    return local_ranking, global_ranking

