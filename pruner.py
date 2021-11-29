import os
import sys
import copy
import math
import logging

import torch
import numpy as np

from .utils import *
from .channel_ranking import *
from .dependency_extractor import *
from .pruning_estimator import NetworkSizeTracker

def rankFilters(rankingType, model, ignoreKws, customRanker):
    if customRanker is None:
        logging.info(f"Performing {rankingType} ranking of filters")
        if rankingType == 'l1-norm':
            channelsPerLayer, globalRanking = l1_norm(model, ignore=ignoreKws)
        else:
            raise ArgumentError(f"Ranking Type {rankingType}, unsupported by default")
    else:
        logging.info(f"Performing custom ranking of filters")
        channelsPerLayer, globalRanking = customRanker(model, ignore=ignoreKws) 

    return channelsPerLayer, globalRanking

def limitGlobalDepsPruning(pl, prnLimits, channelsPerLayer, globalDeps):
    for depLayers in globalDeps:
        for layer in depLayers:
            channels = len(channelsPerLayer[layer])
            if pl >= 0.85:
                prnLimits[layer] = int(math.ceil(channels * 0.5))
            else:
            # This strategy ensures that at as many pruning levels as possible we maintain wider 
            # networks. At very high pruning levels (those in the if case above), we can't apply the 
            # same strategy as it limits the total amount of pruning possible (memory reduction) --> 
            # this is capped at at most pruning 50% of the layer which allows desired memory reduction 
            # with some width maintenance. This is enforced only for external dependencies, 
            # but internally within a block the internal layers can be pruned heavily
                if pl <= 0.5: 
                    prnLimits[layer] = int(math.ceil(channels * (1.0 - pl)))
                else:
                    prnLimits[layer] = int(math.ceil(channels * pl))
    return prnLimits

def identifyFiltersToPrune(pl,
                             _model,
                             channelsPerLayer,
                             globalRanking,
                             connectivity,
                             joinNodes,
                             dependencies,
                             prnLimits):
    logging.info(f"Identifying Filters to Prune")
    paramCalc = lambda x : sum(np.prod(p.shape) for p in x.parameters())
    model = copy.deepcopy(_model)
    unprunedModelParams = paramCalc(model) 
    _channelsPerLayer = channelsPerLayer.copy()
    networkSizeTracker = NetworkSizeTracker(model)
    
    currPr = 0
    filterIdx = 0
    prunedModelParams = unprunedModelParams
    while (currPr < pl and filterIdx < len(globalRanking)):
        layer, filterNum, _ = globalRanking[filterIdx]
        depLayers = [x for l in dependencies for x in l if layer in l]
        depLayers = [layer] if depLayers == [] else depLayers 
        for layer in depLayers:
            if len(_channelsPerLayer[layer]) <= prnLimits[layer]:
                # prevent overpruning of certain layers
                continue
            
            if filterNum not in [x[0] for x in _channelsPerLayer[layer]]:
                # layer has already been pruned (due to dependencies)
                continue

            _channelsPerLayer[layer].pop([i for i,x in enumerate(_channelsPerLayer[layer])\
                                        if x[0] == filterNum][0])
            
            prunedParams = networkSizeTracker.pruneSingleFilter(layer, connectivity, joinNodes)
            prunedModelParams -= prunedParams
            currPr = 1. - (prunedModelParams / unprunedModelParams)
        filterIdx += 1
    return model, _channelsPerLayer

def performPruning(prunedModel, filtersToKeep, connectivity):
    logging.info(f"Removing Filters")
    modelDict = dict(prunedModel.named_modules())
    for n,m in prunedModel.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            opFilters = [x for x,y in filtersToKeep[n]]
            reshapeConvLayer(m, opFilters, ofm=True)
            for layer in connectivity[n]:
                module = modelDict[layer]
                if isinstance(module, torch.nn.Conv2d):
                    if module.groups == 1:
                        reshapeConvLayer(module, opFilters, ofm=False)
                elif isinstance(module, torch.nn.BatchNorm2d):
                    reshapeBnLayer(module, opFilters)
                elif isinstance(module, torch.nn.Linear):
                    reshapeLinearLayer(module, m, opFilters)
    checkPruning(prunedModel)
    return prunedModel

def checkPruning(model):
    logging.info(f"Checking prunining process")
    for n,m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            assert m.out_channels == m.weight.shape[0], f"Layer {n} pruned incorrectly"
            assert (m.in_channels // m.groups) == m.weight.shape[1],\
                                                            f"Layer {n} pruned incorrectly"
        elif isinstance(m, torch.nn.BatchNorm2d):
            assert m.num_features == m.weight.shape[0], f"Layer {n} pruned incorrectly"
        elif isinstance(m, torch.nn.Linear):
            try:
                assert m.in_features == m.weight.shape[1], f"Layer {n} pruned incorrectly"
            except Exception as e:
                breakpoint()

def pruneNetwork(pl,
                  model,
                  ignoreKws=None,
                  minFiltersKept=2,
                  customRanker=None,
                  rankingType='l1-norm',
                  maintainNetworkWidth=True):
    
    assert 0 <= pl < 1, "Pruning level must be value in range [0,1)" 
    
    connectivity, joinNodes, localDeps, globalDeps = getDependencies(model)
    
    channelsPerLayer, globalRanking = rankFilters(rankingType, model, ignoreKws, customRanker)
    
    prnLimits = {k:minFiltersKept for k in channelsPerLayer.keys()}
    if maintainNetworkWidth:
        prnLimits = limitGlobalDepsPruning(pl, prnLimits, channelsPerLayer, globalDeps)
    # After this step, "prunedModel" will only have the description parameters set correctly, but
    # not the weights themselves. "filtersToKeep" will have the channels to keep for each layer 
    prunedModel, filtersToKeep = identifyFiltersToPrune(pl, model, channelsPerLayer,\
                                                              globalRanking, connectivity,\
                                                              joinNodes, localDeps+globalDeps,\
                                                              prnLimits)
    prunedModel = performPruning(prunedModel, filtersToKeep, connectivity)
    return prunedModel, filtersToKeep
