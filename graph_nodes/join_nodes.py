
from .pruner_node import PrunerNode

class AddNode(PrunerNode):
    def __init__(self, name, module, node):
        super().__init__(name, module, node)
        self.feederConvs= []
        self.feederChannelCounts= {}

    def pruneIpFilter(self, filterNum, layer=None):
        for node in self.feederConvs:
            node.pruneOpFilter(filterNum)
        for node in self.nextNodes:
            node.pruneIpFilter(filterNum, layer)

    def propagateChannelCounts(self, channelCounts):
        layerName, channelCount= channelCounts
        if any(layerName==x.name for x in self.feederConvs):
            if layerName not in self.feederChannelCounts.keys():
                self.feederChannelCounts[layerName] = channelCount

                if len(self.feederChannelCounts) == len(self.feederConvs):
                    channels= list(self.feederChannelCounts.values())
                    assert channels.count(channels[0]) == len(channels),\
                                                    "Inputs to add have been pruned differently"
                    [self.feederChannelCounts.pop(x.name) for x in self.feederConvs]
        
        for nextNode in self.nextNodes:
            nextNode.propagateChannelCounts(('addJoin', channelCount))

class ConcatNode(PrunerNode):
    def __init__(self, name, module, node):
        super().__init__(name, module, node)
        self.feederConvs= []
        self.feederChannelCounts= {}
    
    def pruneIpFilter(self, filterNum, layer=None):
        startIdx= 0
        for i,l in enumerate(self.feederConvs):
            if layer == l.name:
                break
            startIdx += self.feederChannelCounts[l.name]
        for node in self.nextNodes:
            node.pruneIpFilter(startIdx+filterNum)
    
    def propagateChannelCounts(self, channelCounts):
        layerName, channelCount= channelCounts
        if any(layerName==x.name for x in self.feederConvs):
            if layerName not in self.feederChannelCounts.keys():
                self.feederChannelCounts[layerName] = channelCount
        
        for nextNode in self.nextNodes:
            nextNode.propagateChannelCounts(channelCounts)
    
