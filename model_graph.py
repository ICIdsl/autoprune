import os
import sys
import copy
import logging

import torch
import numpy as np

from .dependency_extractor import *

def getGraph(model, traced=True):
    execGraph= getExecutionGraph(model)    
    layerGraph= buildLayerGraph(model, execGraph) 
    breakpoint()

def graphOpName(module):
    if isinstance(module, torch.nn.Conv2d):
        return "aten::_convolution"
    elif isinstance(module, torch.nn.ReLU):
        return "aten::relu_"
    elif isinstance(module, torch.nn.MaxPool2d):
        return "aten::max_pool2d"
    elif isinstance(module, torch.nn.BatchNorm2d):
        return "aten::batch_norm"
    elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
        return "aten::adaptive_avg_pool2d"
    elif isinstance(module, torch.nn.Linear):
        return "aten::addmm"
    else:
        raise NotImplementedError(f"Graph Op Name for module {module}")

def getExecutionGraph(model):
    print(f"Tracing model")
    tracedModel= torch.jit.trace(model, torch.Tensor(1,3,224,224))
    networkGraph= tracedModel.inlined_graph
    return networkGraph
    
def buildLayerGraph(model, execGraph):
    modules= baseModules(model) 
    translations= getExecGraphToLayerNameTranslations(modules, model, execGraph)
    root= getRootNode(model, translations)
    breakpoint()

def baseModules(model):
    bm= []
    for n,m in model.named_modules():
        if len(list(m.children())) == 0:
            if not any(type(m) == type(x) for x in bm):
                bm.append(m)
    return bm

def getExecGraphToLayerNameTranslations(modules, model, execGraph):
    translations= {}
    for m in modules:
        execNodes= findAllNodesinExecGraph(execGraph, graphOpName(m))  
        layerNames= findAllLayersinModel(model, type(m))
        for n,e in zip(layerNames, execNodes):
            translations[e] = n
    return translations
          
def findAllNodesinExecGraph(graph, opName):
    return graph.findAllNodes(opName)

def findAllLayersinModel(model, moduleType):
    return [n for n,m in model.named_modules() if isinstance(m, moduleType)]

def getRootNode(model, translations)
    firstConv= [n for n,m in model.named_modules() if isinstance(m, torch.nn.Conv2d)][0]
    root= [k for k,v in translations.items() if v == firstConv][0]
    return root 

