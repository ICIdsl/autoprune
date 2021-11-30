
import torch
from ..utils import reshapeTensor
from .pruner_node import PrunerNode

class Conv2DNode(PrunerNode):
    def __init__(self, name, module, node):
        super().__init__(name, module, node)
        self.dwLinkedConvs= []
        self.filters= list(range(module.out_channels))
        self.spatialDims= self.module.weight.shape[2:]
        self.inChannels= list(range(module.in_channels))

    def getParamsPruned(self, ofm=True):
        if self.module.groups == 1:
            if ofm:
                return (self.module.in_channels * self.spatialDims.numel()) + 1
            else:
                return self.module.out_channels * self.spatialDims.numel()
        else:
            return self.spatialDims.numel()

    def pruneOpFilter(self, filterNum):
        if len(self.filters) <= self.pruningLimit:
            return 
        if filterNum in self.filters:
            self.module.out_channels -= 1 
            filterIdx= self.filters.index(filterNum)
            self.filters.pop(filterIdx)
            self.updatePrunedParams(self.getParamsPruned(ofm=True))
        
            for convNode in self.dwLinkedConvs:
                convNode.pruneOpFilter(filterNum)

            for node in self.nextNodes:
                node.pruneIpFilter(filterNum)

    def pruneIpFilter(self, filterNum):
        if filterNum in self.inChannels:
            self.module.in_channels -= 1 
            if self.module.groups != 1:
                self.module.groups = self.module.in_channels
            filterIdx= self.inChannels.index(filterNum)
            self.inChannels.pop(filterIdx)
            self.updatePrunedParams(self.getParamsPruned(ofm=False))
            
            for convNode in self.dwLinkedConvs:
                convNode.pruneOpFilter(filterNum)
    
    def propagateChannelCounts(self, channelCounts):
        for nextNode in self.nextNodes:
            nextNode.propagateChannelCounts((self.name, int(self.module.out_channels)))

    def prune(self, seenNodes):
        if self in seenNodes:
            return
        seenNodes.append(self)
        
        if ((self.module.in_channels/self.module.groups) != self.module.weight.shape[1]): 
            self.module.weight= torch.nn.Parameter(reshapeTensor(self.module.weight,\
                                                                        self.inChannels, 1))
        
        if (self.module.out_channels != self.module.weight.shape[0]):
            self.module.weight= torch.nn.Parameter(reshapeTensor(self.module.weight,\
                                                                        self.filters, 0))
            if self.module.bias is not None:
                self.module.bias= torch.nn.Parameter(reshapeTensor(self.module.bias,\
                                                                        self.filters, 0))
        
        for nextNode in self.nextNodes:
            nextNode.prune(seenNodes)

