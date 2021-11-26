import os
import sys
import copy
import logging

import torch
import numpy as np

def convOpName(traced=True):
    return 'aten::_convolution' if traced else 'aten::conv2d'

def bnOpName(traced=True):
    return 'aten::batch_norm'

def fcOpName(traced=True):
    return 'aten::addmm' if traced else 'aten::matmul'

def addOpNames(traced=True):
    return ['aten::add', 'aten::add_']

def findAllConvs(graph, traced=True):
    return graph.findAllNodes(convOpName(traced))

def findAllBn(graph, traced=True):
    return graph.findAllNodes(bnOpName(traced))

def findAllFcs(graph, traced=True):
    return graph.findAllNodes(fcOpName(traced))

def findAllAddNodes(graph, traced=True):
    nodes = []
    nodes += graph.findAllNodes(addOpNames(traced)[0])
    nodes += graph.findAllNodes(addOpNames(traced)[1])
    return nodes 

def node2Str(node):
    return str(node).strip()

def translateNode(node, translator):
    return node if node not in translator.keys() else translator[node]

def reverseLookup(modelName, translator):
    return [k for k,v in translator.items() if v == modelName][0]

def translateIdomTree(tree, translator, _print=False):
    trans= {translateNode(k, translator):\
                            [translateNode(x, translator) for x in v if x in translator.keys()]\
                                                                         for k,v in tree.items()}
    if _print:
        for k,v in trans.items():
            print(k)
            print(v)
    return trans

def translateSequentialDependencies(deps, convTranslate):
    newDeps = {convTranslate[k] : [convTranslate[x] for x in v] for k,v in deps.items()}
    return newDeps

def translateAddDependencies(deps, convTranslate):
    removeIdx = []
    newDeps = [[convTranslate[x] for x in v] for k,v in deps.items()]
    for i,deps in enumerate(newDeps):
        if any(set(deps) < set(x) for j,x in enumerate(newDeps) if i != j):
            removeIdx.append(i)
    noRedundencyDeps = [x for i,x in enumerate(newDeps) if i not in removeIdx]
    
    duplicates = []
    for i,deps in enumerate(noRedundencyDeps):
        _dup = [j for j,x in enumerate(noRedundencyDeps) if i<j and set(deps) == set(x)]
        if len(_dup) != 0:
            duplicates += _dup
    noRedundencyDeps = [x for i,x in enumerate(noRedundencyDeps) if i not in duplicates]
    return noRedundencyDeps

def getIdomTree(root, nodes, stopAt=None):
    if not root.hasMultipleOutputs():
        node = root.output()
        for user in node.uses():
            if user.user.kind() == stopAt:
                nodes.append(user.user)
            else:
                if user.user.hasUses():
                    nodes.append(user.user)
                    getIdomTree(user.user, nodes, stopAt)
    else:
        logging.warning(f"Node {root} has multiple outputs!")
        sys.exit()

def findNodesIdomBy(root, traced=True):
    nodes = []
    getIdomTree(root, nodes, stopAt=convOpName(traced))
    return nodes

def filterImportantLayersInIdom(idomTrees, traced=True):
    sanitisedIdom = {c:[] for c in idomTrees.keys()}
    for conv, tree in idomTrees.items():
        for node in tree:
            if node.kind() == convOpName(traced):
                sanitisedIdom[conv].append(node)
            elif node.kind() == bnOpName(traced):
                sanitisedIdom[conv].append(node)
            elif node.kind() == fcOpName(node):
                sanitisedIdom[conv].append(node)
    return sanitisedIdom

def getAddConnectedDepsOld(idomTrees, addNodes, traced=True):
    connectedConvs = {n:[] for n in addNodes}
    for node in addNodes:
        for inp in node.inputs():
            for conv, tree in idomTrees.items():
                if conv.output().debugName() == inp.debugName():
                    connectedConvs[node].append(conv)
                else:
                    for idomNode in tree:
                        if not any(idomNode.kind() == x for x in addOpNames(traced)):
                            if idomNode.output().debugName() == inp.debugName():
                                connectedConvs[node].append(conv)

    keysToRemove = []
    for k,v in connectedConvs.items():
        if len(v) == 1:
            keysToRemove.append(k)
    [connectedConvs.pop(k) for k in keysToRemove]
    
    return connectedConvs

def getAddConnectedDeps(convIdomTrees, allAdds, traced=True):
    feederConvs= [] 
    for addNode in allAdds

    return connectedConvs

def getAddIdoms(allAdds):
    addIdomTrees = {}
    for add in allAdds:
        nodes = findNodesIdomBy(add)
        addIdomTrees[add] = nodes
    return addIdomTrees

def getConvIdoms(allConvs):
    convIdomTrees = {}
    for conv in allConvs:
        nodes = findNodesIdomBy(conv)
        convIdomTrees[conv] = nodes
    return convIdomTrees

def getDwConvs(model, translator):
    dwConvs= []
    for n,m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            if m.groups != 1: 
                if m.groups == m.in_channels:
                    dwConvs.append(reverseLookup(n, translator))
                else:
                    raise NotImplementedError("Grouped convs not implemented")
    return dwConvs

def identifyDependencies(model, graph, translator):
    allConvs = findAllConvs(graph)         
    convIdomTrees = getConvIdoms(allConvs) 
    
    allAdds = findAllAddNodes(graph)
    addIdomTrees= getAddIdoms(allAdds)

    dwConvs= getDwConvs(model, translator)
    
    addDeps = getAddConnectedDeps(convIdomTrees, allAdds)
    breakpoint()
    add2convConnections = filterImportantLayersInIdom(addIdomTrees) 
    conv2convConnections = filterImportantLayersInIdom(convIdomTrees)

    return {'seqDeps': convNodeConnections, 
            'addDeps': addDeps,
            'addNodeConnections': addNodeConnections}

def getModelGraph(model, traced):
    if not traced:
        print(f"Scripting model")
        scriptedModel = torch.jit.script(model)
        networkGraph = scriptedModel.inlined_graph
    else:
        print(f"Tracing model")
        tracedModel= torch.jit.trace(model, torch.Tensor(1,3,224,224))
        networkGraph= tracedModel.inlined_graph
    return networkGraph

def getGraphToModelTranslations(model, networkGraph, traced):
    modelConvs = [n for n,m in model.named_modules() if isinstance(m, torch.nn.Conv2d)]
    graphConvs = findAllConvs(networkGraph, traced)
    convTranslate = {g:m for g,m in zip(graphConvs, modelConvs)}
    
    modelBns = [n for n,m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
    graphBns = findAllBn(networkGraph, traced)
    for mBn, gBn in zip(modelBns, graphBns):
        convTranslate[gBn] = mBn
    
    firstModelFc = [n for n,m in model.named_modules() if isinstance(m, torch.nn.Linear)][0]
    firstGraphFc = findAllFcs(networkGraph, traced)[0]
    convTranslate[firstGraphFc] = firstModelFc
    
    return convTranslate    

def getDependencies(model, traced=True):
    networkGraph= getModelGraph(model, traced)
    translator= getGraphToModelTranslations(model, networkGraph, traced) 
    dependencies = identifyDependencies(model, networkGraph, translator)
    
    modelDeps = {}
    for depType, deps in dependencies.items():
        if depType == 'seqDeps':
            modelDeps['layerConnectivity'] = translateSequentialDependencies(deps, convTranslate)
        elif depType == 'addDeps':
            modelDeps['addDependencies'] = translateAddDependencies(deps, convTranslate)
        elif depType == 'addNodeConnections':
            modelDeps['addNodeConnections'] = translateAddDependencies(deps, convTranslate)
    
    return modelDeps

def categoriseDependencies(dependencies):
    localDepTypes = ['dwDependencies']
    globalDepTypes = ['addDependencies']
    localDeps = [x for k,l in dependencies.items() for x in l if k in localDepTypes]
    globalDeps = [x for k,l in dependencies.items() for x in l if k in globalDepTypes]
    
    connectivity = dependencies['layerConnectivity']
    
    joinNodes = {}
    if 'addNodeConnections' in dependencies.keys():
        joinNodes['aten::add_'] = dependencies['addNodeConnections']
    
    return connectivity, joinNodes, localDeps, globalDeps

