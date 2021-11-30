
import torch
from ..utils import reshapeTensor 
from .pruner_node import PrunerNode

class BatchNorm2DNode(PrunerNode):
    def __init__(self, name, module, node):
        super().__init__(name, module, node)
        self.filters= list(range(module.num_features))

    def pruneIpFilter(self, filterNum):
        if filterNum in self.filters:
            self.module.num_features -= 1 
            filterIdx= self.filters.index(filterNum)
            self.filters.pop(filterIdx)
            self.updatePrunedParams(4) # weight, bias, runMean, runVar
        
        for node in self.nextNodes:
            node.pruneIpFilter(filterNum)

    def prune(self, seenNodes):
        if self in seenNodes:
            return
        seenNodes.append(self)

        if int(self.module.weight.shape[0]) != self.module.num_features:
            newWeight = reshapeTensor(self.module.weight, self.filters, axis=0) 
            self.module.weight = torch.nn.Parameter(newWeight)
            newBias = reshapeTensor(self.module.bias, self.filters, axis=0)
            self.module.bias = torch.nn.Parameter(newBias)
            self.module.running_mean =\
                    reshapeTensor(self.module.running_mean, self.filters, axis=0)
            self.module.running_var = reshapeTensor(self.module.running_var, self.filters, axis=0)

        for nextNode in self.nextNodes:
            nextNode.prune(seenNodes)

