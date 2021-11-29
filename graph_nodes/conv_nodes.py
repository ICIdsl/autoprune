
import torch
from .pruner_node import PrunerNode

class Conv2DNode(PrunerNode):
    def __init__(self, name, module, node):
        super().__init__(name, module, node)
        self.dwLinkedConvs= []
        self.filters= list(range(module.out_channels))
        self.inChannels= list(range(module.in_channels))
        self.spatialDims= self.module.weight.shape[2:]

    def getParamsPruned(self, ofm=True):
        if self.module.groups == 1:
            if ofm:
                return self.module.in_channels * self.spatialDims.numel() 
            else:
                return self.module.out_channels * self.spatialDims.numel()
        else:
            return self.spatialDims.numel()

    def pruneOpFilter(self, filterNum):
        if filterNum in self.filters:
            print(f"Pruning op filter: {self.name}")
            self.module.out_channels -= 1 
            filterIdx= self.filters.index(filterNum)
            self.filters.pop(filterIdx)
            self.updatePrunedParams(self.getParamsPruned(ofm=True))
        
            for convNode in self.dwLinkedConvs:
                convNode.pruneOpFilter(filterNum)

            for node in self.nextNodes:
                node.pruneIpFilter([self], filterNum)

    def pruneIpFilter(self, layersPruned, filterNum):
        if filterNum in self.inChannels:
            print(f"Pruning ip filter: {self.name}")
            self.module.in_channels -= 1 
            if self.module.groups != 1:
                self.module.groups = self.module.in_channels
            filterIdx= self.inChannels.index(filterNum)
            self.inChannels.pop(filterIdx)
            self.updatePrunedParams(self.getParamsPruned(ofm=False))
            
            for convNode in self.dwLinkedConvs:
                convNode.pruneOpFilter(filterNum)

