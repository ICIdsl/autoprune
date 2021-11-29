
from .pruner_node import PrunerNode

class AddNode(PrunerNode):
    def __init__(self, name, module, node):
        super().__init__(name, module, node)
        self.feederConvs= []

    def pruneIpFilter(self, layersPruned, filterNum):
        # print(f"Pruning ip filter (pass through): {self.name}")
        for node in self.feederConvs:
            if node not in layersPruned:
                node.pruneOpFilter(filterNum)
        for node in self.nextNodes:
            node.pruneIpFilter(layersPruned, filterNum)
