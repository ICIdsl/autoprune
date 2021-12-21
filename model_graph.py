import os
import sys
import copy
import logging

import torch
import numpy as np

from .utils import *
from .graph_nodes import *

IGNORED_NODES=[]

def lookupTranslation(t, node=None, name=None, mod=None):
    if node is not None:
        if node in t.keys():
            return (node, t[node].n, t[node].m)
        else:
            raise KeyError("Not a standard layer")
    else:
        assert (name is not None) and (mod is not None), "Can't reverse lookup without name or mod"
        for _node,_t in t.items():
            if (name is not None) and (_t.n == name):
                return _node, _t.n, _t.m
            if (mod is not None) and (_t.m == mod):
                return _node, _t.n, _t.m
        raise KeyError("Not a layer in the model")

def validBaseModules():
    return [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear]

def graphOpName(module):
    if isinstance(module, torch.nn.Conv2d):
        return "aten::_convolution"
    elif isinstance(module, torch.nn.BatchNorm2d):
        return "aten::batch_norm"
    elif isinstance(module, torch.nn.Linear):
        return ["aten::addmm", "aten::matmul"]
    else:
        raise NotImplementedError(f"Graph Op Name for module {module}")

def buildLayerGraph(model, execGraph):
    modules= baseModules(model) 
    translations= getExecGraphToLayerNameTranslations(modules, model, execGraph)
    print(f"Creating Graph")
    createdNodes= []
    root= getRootNode(model, translations, createdNodes)
    createGraph(root, root.node, translations, createdNodes) 
    return root

def baseModules(model):
    bm= []
    for n,m in model.named_modules():
        if any(isinstance(m, x) for x in validBaseModules()):
            # if not any(type(m) == type(x) for x in bm):
            if not any(m == x for x in bm):
                bm.append(m)
    return bm

class Translation(ImmutableClass):
    def __init__(self, name, module):
        self.n= name
        self.m= module

def getExecGraphToLayerNameTranslations(modules, model, execGraph):
    translations= {}
    for m in modules:
        layers= findAllLayersinModel(model, type(m))
        execNodes= findAllNodesinExecGraph(execGraph, graphOpName(m))  
        for (n,_m),e in zip(layers, execNodes):
            translations[e] = Translation(n,_m)
    return translations
          
def findAllNodesinExecGraph(graph, opName):
    if isinstance(opName, list):
        nodes = {}
        for name in opName:
            nodes[name] = graph.findAllNodes(name)
        assert sum(len(x) > 0 for x in nodes.values()),\
              (f"Multiple options in {list(nodes.keys())} returned graph nodes. This is currently"
                "not handled. This suggests different graph ops are being used for the same"
                "network module.")
        return [v for v in nodes.values() if len(v) > 0][0]
    else:
        return graph.findAllNodes(opName)

def findAllLayersinModel(model, moduleType):
    return [(n,m) for n,m in model.named_modules() if isinstance(m, moduleType)]

def getRootNode(model, translations, createdNodes):
    firstConv= [(n,m) for n,m in model.named_modules() if isinstance(m, torch.nn.Conv2d)][0]
    root= [k for k,t in translations.items() if t.n == firstConv[0]][0]
    _, rootNode= createNode(translations, createdNodes, root, *firstConv)
    return rootNode

def createGraph(prevNode, prevExecGraphNode, translations, createdNodes):
    if not prevExecGraphNode.hasUses():
        return
    if prevExecGraphNode.hasMultipleOutputs():
        raise NotImplementedError("Handling of graph node with multiple outputs") 
    
    for user in prevExecGraphNode.output().uses():
        isNewNode, nextNode= createNode(translations, createdNodes, node=user.user)
        prevNode.updateNextNodes(nextNode)
        if not isNewNode:
            return
        currNode= nextNode if nextNode is not None else prevNode
        createGraph(currNode, user.user, translations, createdNodes)

def createNode(translations, createdNodes, node=None, name=None, module=None):
    layerInModel, node, name, module= lookupNodeTranslation(translations, node, name, module) 
    existing= checkIfCreated(node, createdNodes)
    if existing is None:
        if layerInModel:
            newNode= createLayerNode(name, module, node)
            createdNodes.append(newNode)
            return True, newNode
        else:
            newNode= createFunctionalNode(node)
            if newNode is not None:
                createdNodes.append(newNode)
            return True, newNode
    else:
        return False, existing

def lookupNodeTranslation(translations, node, name, module):
    layerInModel= True 
    if (name is None) or (module is None) or (node is None):
        try:
            node, name, module= lookupTranslation(translations, node=node, name=name, mod=module)
        except KeyError as e:
            layerInModel= False 
    return layerInModel, node, name, module

def checkIfCreated(newNode, createdNodes):
    for graphNode in createdNodes:
        if graphNode.node == newNode:
            return graphNode
    return None

def createLayerNode(name, module, node):
    if isinstance(module, torch.nn.Conv2d):
        return Conv2DNode(name, module, node) 
    elif isinstance(module, torch.nn.BatchNorm2d):
        return BatchNorm2DNode(name, module, node) 
    elif isinstance(module, torch.nn.Linear):
        return LinearNode(name, module, node) 
    else:
        raise NotImplementedError(f"Handling {module} not implemented")

def createFunctionalNode(node):
    assert node is not None, "Cannot create node without a node"
    if "aten::add" in node.kind():
        return AddNode('addJoin', None, node)
    elif "aten::cat" in node.kind():
        return ConcatNode('concatJoin', None, node)
    elif "aten::relu" in node.kind():
        return ReLUNode('relu', None, node)
    elif "aten::max_pool2d" in node.kind():
        return PoolingNode('maxpool2d', None, node)
    elif "aten::adaptive_avg_pool2d" in node.kind():
        return PoolingNode('adaptive_avgpool2d', None, node)
    elif "aten::flatten" in node.kind():
        return FlattenNode('flatten', None, node)
    else:
        global IGNORED_NODES
        if node.kind() not in IGNORED_NODES:
            print(f"Ignoring {node.kind()} node")
            IGNORED_NODES.append(node.kind())
        return None


