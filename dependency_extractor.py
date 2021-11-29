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

def removeRedundantDeps(depDict):
    noRedundancy= {}
    for k,v in depDict.items():
        isSubset = [all(x in v_ for x in v) for k_, v_ in depDict.items() if k != k_]
        if not any(isSubset):
            noRedundancy[k] = v
    return noRedundancy

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

def filterImportantLayersInIdom(idomTrees, traced=True, drop=[]):
    sanitisedIdom = {c:[] for c in idomTrees.keys()}
    for conv, tree in idomTrees.items():
        for node in tree:
            if node.kind() == convOpName(traced):
                if 'conv' not in drop:
                    sanitisedIdom[conv].append(node)
            elif node.kind() == bnOpName(traced):
                if 'bn' not in drop:
                    sanitisedIdom[conv].append(node)
            elif node.kind() == fcOpName(node):
                if 'fc' not in drop:
                    sanitisedIdom[conv].append(node)
    return sanitisedIdom

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

def getConvsThatFeedAdds(convIdomTrees, allAdds, traced=True):
    feederConvs = {addNode: [k for k,v in convIdomTrees.items() if addNode in v]\
                                                                    for addNode in allAdds}
    return removeRedundantDeps(feederConvs)

def getDependencies(model, traced=True):
    networkGraph= getModelGraph(model, traced)
    translator= getGraphToModelTranslations(model, networkGraph, traced) 
    dependencies = identifyDependencies(model, networkGraph, translator)
    return dependencies

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

def identifyDependencies(model, graph, translator):
    connectivity= layerConnectivity(graph, translator)
    joinNodes= joinConnectedConvs(graph, translator)
    localDeps= localDependencies(model, graph, translator)
    globalDeps= globalDependencies(graph, translator)
    return connectivity, joinNodes, localDeps, globalDeps

def layerConnectivity(graph, translator):
    allConvs = findAllConvs(graph)         
    convIdomTrees = getConvIdoms(allConvs) 
    relevantLayerConnections = filterImportantLayersInIdom(convIdomTrees)
    connectivity= {translateNode(k, translator): [translateNode(x, translator) for x in v]\
                                                    for k,v in relevantLayerConnections.items()}
    noRepeateConnectivity= {k: list(set(v)) for k,v in connectivity.items()}
    return noRepeateConnectivity 

def localDependencies(model, graph, translator):
    dwDeps= depthwiseDeps(model, graph, translator)
    return dwDeps

def depthwiseDeps(model, graph, translator):
    dwConvs= [translateNode(x, translator) for x in getDwConvs(model, translator)]
    connectivity= layerConnectivity(graph, translator)
    dwDeps= [[k,dwConv] for dwConv in dwConvs for k,v in connectivity.items() if dwConv in v] 
    return dwDeps    

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

def globalDependencies(graph, translator):
    addJoinDeps= addNodeInputs(graph, translator)
    return addJoinDeps

def addNodeInputs(graph, translator):
    allConvs = findAllConvs(graph)         
    allAdds = findAllAddNodes(graph)
    convIdomTrees = getConvIdoms(allConvs) 
    conv2addConnections= getConvsThatFeedAdds(convIdomTrees, allAdds)
    conv2addDeps= [[translateNode(x, translator) for x in v] for v in conv2addConnections.values()]
    return conv2addDeps

def joinConnectedConvs(graph, translator):
    addJoinConnectedConvs= addNodeOutputs(graph, translator)
    return {'addJoins': addJoinConnectedConvs}

def addNodeOutputs(graph, translator):
    allAdds = findAllAddNodes(graph)
    addIdomTrees= getAddIdoms(allAdds)
    add2convConnections = removeRedundantDeps(filterImportantLayersInIdom(addIdomTrees))
    add2convDeps= [[translateNode(x, translator) for x in v] for v in add2convConnections.values()]
    return add2convDeps

