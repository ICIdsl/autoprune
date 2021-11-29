import math
import torch

def updateAddNodeInputConvs(node, seenNodes=[]):
    if node in seenNodes:
        return
    if isinstance(node.module, torch.nn.Conv2d):
        updateConnectedAddJoin(node)
    seenNodes.append(node)
    for nextNode in node.nextNodes:
        updateAddNodeInputConvs(nextNode, seenNodes)

def updateConnectedAddJoin(node, convNode=None):
    if (convNode is not None) and isinstance(node.module, torch.nn.Conv2d):
        # isn't connected to add before next conv
        return
    if node.name == 'addJoin':
        node.feederConvs.append(convNode)
    for nextNode in node.nextNodes:
        updateConnectedAddJoin(nextNode, (convNode if convNode is not None else node))

def updateDwConvInputConvs(node, seenNodes=[]):
    if node in seenNodes:
        return
    if isinstance(node.module, torch.nn.Conv2d):
        updateConnectedDwConvs(node)
    seenNodes.append(node)
    for nextNode in node.nextNodes:
        updateDwConvInputConvs(nextNode, seenNodes)

def updateConnectedDwConvs(node, convNode=None):
    if (convNode is not None) and isinstance(node.module, torch.nn.Conv2d):
        if node.module.groups != 1:
            node.dwLinkedConvs.append(convNode)    
        return
    for nextNode in node.nextNodes:
        updateConnectedDwConvs(nextNode, (convNode if convNode is not None else node))

def updatePruningLimits(node, pl, minFiltersKept, maintainNetworkWidth, seenNodes=[]):
    if node in seenNodes:
        return
    if isinstance(node.module, torch.nn.Conv2d):
        try:
            node.pruningLimit= minFiltersKept
        except AttributeError as e:
            # don't change if it has already been set by AddNode dependencies
            pass
    if (node.name == 'addJoin') and maintainNetworkWidth:
        [limitGlobalPruning(x, pl) for x in node.feederConvs] 
    seenNodes.append(node)
    for nextNode in node.nextNodes:
        updatePruningLimits(nextNode, pl, minFiltersKept, maintainNetworkWidth, seenNodes)

def limitGlobalPruning(node, pl):
    channels= node.module.out_channels
    node.makeAttrMutable('pruningLimit')
    if pl >= 0.85:
        node.pruningLimit= int(math.ceil(channels * 0.5))
    else:
        if pl <= 0.5:
            node.pruningLimit= int(math.ceil(channels * (1.0 - pl)))
        else:
            node.pruningLimit= int(math.ceil(channels * pl))

def updateSpatialDimsToFc(node, seenNodes=[]):
    if node in seenNodes:
        return
    if isinstance(node.module, torch.nn.Conv2d):
        updateConnectedFc(node, node.module.weight.shape[2:].numel(), [])
    seenNodes.append(node)
    for nextNode in node.nextNodes:
        updateSpatialDimsToFc(nextNode, seenNodes)

def updateConnectedFc(node, spatialDims, seenNodes):
    if node in seenNodes:
        return
    if isinstance(node.module, torch.nn.Linear)

    
