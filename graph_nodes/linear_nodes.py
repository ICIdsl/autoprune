
from .pruner_node import PrunerNode

class LinearNode(PrunerNode):
    def __init__(self, name, module, node):
        super().__init__(name, module, node)

    def pruneIpFilter(self, layersPruned, filterNum):
        breakpoint()

