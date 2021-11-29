
from .pruner_node import PrunerNode

class ReLUNode(PrunerNode):
    def __init__(self, name, module, node):
        super().__init__(name, module, node)

