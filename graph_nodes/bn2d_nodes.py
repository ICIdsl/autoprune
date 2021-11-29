
from .pruner_node import PrunerNode

class BatchNorm2DNode(PrunerNode):
    def __init__(self, name, module, node):
        super().__init__(name, module, node)
        self.filters= list(range(module.num_features))

    def pruneIpFilter(self, layersPruned, filterNum):
        # print(f"Pruning ip filter (pass through): {self.name}")
        if filterNum in self.filters:
            self.module.num_features -= 1 
            filterIdx= self.filters.index(filterNum)
            self.filters.pop(filterIdx)
            self.updatePrunedParams(4)
        
        for node in self.nextNodes:
            node.pruneIpFilter([self]+layersPruned, filterNum)

