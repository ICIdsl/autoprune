
import torch
from ..utils import *
from .pruner_node import PrunerNode

class LinearNode(PrunerNode):
    def __init__(self, name, module, node):
        super().__init__(name, module, node)

    def pruneIpFilter(self, filterNum):
        if filterNum in self.ipFilters:
            self.module.in_features -= self.spatialDims
            paramsPruned= self.spatialDims * self.module.out_features
            self.updatePrunedParams(int(paramsPruned))
            idx= self.ipFilters.index(filterNum)
            self.ipFilters.pop(idx)

    def propagateChannelCounts(self, channelCounts):
        prevOpChannels= channelCounts[1]
        spatialDims= int(self.module.in_features / prevOpChannels)
        if not hasattr(self, 'spatialDims'):
            self.spatialDims= spatialDims 
            self.ipFilters= list(range(prevOpChannels))
        else:
            assert spatialDims == self.spatialDims,\
            f"Received different values for prev op channels ({self.spatialDims},{spatialDims})"
    
    def prune(self, seenNodes):
        if self in seenNodes:
            return
        seenNodes.append(self)

        if self.module.weight.shape[1] != self.module.in_features:
            idx= [x for y in self.ipFilters for x in range(y, y+self.spatialDims)]
            self.module.weight= torch.nn.Parameter(reshapeTensor(self.module.weight, idx, axis=1))
