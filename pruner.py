import os
import sys
import copy
import math
import logging

import torch
import numpy as np

from .utils import *
from .model_graph import *
from .pruner_passes import *
from .channel_ranking import *

def iterateGraph(node, model, seenNodes=[]):
    if any(node.node == x.node for x in seenNodes):
        return 
    seenNodes.append(node)
    
    print(node.name)
    # if node.name == 'concatJoin':
    #     print("Convs that feed add node", [x.name for x in node.feederConvs])
    
    # if isinstance(node.module, torch.nn.Conv2d):
    #     print(f"Conv pruning limit: {node.name, node.pruningLimit}")
    #     if node.module.groups != 1:
    #         print("Convs connected to dw conv", node.dwLinkedConvs[0].name,\
    #                                                     node.name, node.module.groups)
    # 
    # if isinstance(node.module, torch.nn.Linear):
    #     print(f"Spatial Dims: {node.name, node.spatialDims}")

    for i,nextNode in enumerate(node.nextNodes):
        iterateGraph(nextNode, model)

def printLayerSizes(node, seenNodes):
    if any(node.node == x.node for x in seenNodes):
        return 
    seenNodes.append(node)
    
    if isinstance(node.module, torch.nn.Conv2d):
        print(f"Conv {node.name}: {node.module}")
    
    if isinstance(node.module, torch.nn.Linear):
        print(f"FC {node.name}: {node.module}")

    for i,nextNode in enumerate(node.nextNodes):
        printLayerSizes(nextNode, seenNodes)

def pruneNetwork(pl,
                 model,
                 network,
                 ignoreKws=None,
                 minFiltersKept=2,
                 customRanker=None,
                 rankingType='l1-norm',
                 maintainNetworkWidth=True):
    
    assert 0 <= pl < 1, "Pruning level must be value in range [0,1)" 
    prunedModel= copy.deepcopy(model)
    execGraph = getExecutionGraph(network, prunedModel)
    root= buildLayerGraph(prunedModel, execGraph) 
    runPrePruningPasses(root)
    updatePruningLimits(root, pl, minFiltersKept, maintainNetworkWidth)
    globalRanking = rankFilters(rankingType, prunedModel, ignoreKws, customRanker)
    identifyFiltersToPrune(root, globalRanking, prunedModel, pl)
    root.prune([])
    checkPruning(prunedModel)
    return prunedModel

def getExecutionGraph(network, model):
    print(f"Tracing model")
    tracedModelF= f"/home/ar4414/torch_script_pruning_wrapper/traced_models/{network}.pt"
    if os.path.isfile(tracedModelF):
        tracedModel= torch.jit.load(tracedModelF)
    else:
        tracedModel= torch.jit.trace(model, torch.Tensor(1,3,224,224))
        torch.jit.save(tracedModel, tracedModelF)
    networkGraph= tracedModel.inlined_graph
    return networkGraph

def runPrePruningPasses(root): 
    updateConcatNodeInputConvs(root, []) 
    updateDwConvInputConvs(root, [])
    updateAddNodeInputConvs(root, []) 
    root.propagateChannelCounts([])

def rankFilters(rankingType, model, ignoreKws, customRanker):
    if customRanker is None:
        logging.info(f"Performing {rankingType} ranking of filters")
        if rankingType == 'l1-norm':
            globalRanking = l1Norm(model, ignore=ignoreKws)
        else:
            raise ArgumentError(f"Ranking Type {rankingType}, unsupported by default")
    else:
        logging.info(f"Performing custom ranking of filters")
        globalRanking = customRanker(model, ignore=ignoreKws) 
    return globalRanking

def identifyFiltersToPrune(root, globalRanking, model, pl):
    paramCalc = lambda x : sum(np.prod(p.shape) for p in x.parameters())
    currPr = 0
    filterIdx = 0
    unprunedParams= paramCalc(model)
    while (currPr < pl) and (filterIdx < len(globalRanking)):
        print(f"Current prune rate= {currPr:.4f}/{pl:.4f}", end='\r')
        layer, filterNum, _ = globalRanking[filterIdx]
        pruneLayer(root, layer, filterNum, [])
        paramsPruned= getPrunedParams(root, [])
        currPr = paramsPruned / unprunedParams
        filterIdx += 1
    print()

def pruneLayer(node, layer, filterNum, seenNodes=[]):
    if node in seenNodes:
        return
    if node.name == layer:
        node.pruneOpFilter(filterNum)
    seenNodes.append(node)
    for nextNode in node.nextNodes:
        pruneLayer(nextNode, layer, filterNum, seenNodes)

def getPrunedParams(node, seenNodes=[]):
    if node in seenNodes:
        return 0
    elif len(node.nextNodes) == 0:
        return node.prunedParams
    seenNodes.append(node)
    prunedParams= node.prunedParams
    for nextNode in node.nextNodes:
        prunedParams += getPrunedParams(nextNode, seenNodes)
    return prunedParams

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
        
