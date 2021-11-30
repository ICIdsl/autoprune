
from ..utils import ImmutableClass

class PrunerNode(ImmutableClass):
    def __init__(self, name, module, node):
        self.name= name
        self.node= node
        self.module= module
        self.nextNodes= []
        self.prunedParams= 0

    def updateNextNodes(self, nextNode):
        if nextNode is not None:
            if not any(x.node == nextNode.node for x in self.nextNodes):
                self.nextNodes.append(nextNode)

    def updatePrunedParams(self, paramsPruned):
        self.makeAttrMutable('prunedParams')
        self.prunedParams += paramsPruned

    def pruneIpFilter(self, filterNum):
        for node in self.nextNodes:
            node.pruneIpFilter(filterNum)

    def propagateChannelCounts(self, channelCounts):
        for nextNode in self.nextNodes:
            nextNode.propagateChannelCounts(channelCounts)

    def prune(self, seenNodes):
        if self in seenNodes:
            return
        seenNodes.append(self)
        for nextNode in self.nextNodes:
            nextNode.prune(seenNodes)




