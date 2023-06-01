import numpy as np
class GraphDataStore():
    def __init__(self,nodeNames):
        self.nodeNames = nodeNames
        self.edgeDict = {}
        self.nodeNumber = len(nodeNames)
        self.nodeAttributes = {nodeName:{} for nodeName in nodeNames}
        self.globalAttributes = {}
        for firstName in nodeNames:
            for secondName in nodeNames:
                self.edgeDict[f'{firstName}->{secondName}'] = {}
    def addEdgeAttributes(self,fromNode,toNode,attributes):
        if not f'{fromNode}->{toNode}' in self.edgeDict:
            raise LookupError('Edge doesnt exist')
        for key, value in attributes:
            self.edgeDict[f'{fromNode}->{toNode}'][key]=value
    def addNodeAttributes(self,node,attributes):
        for attribute,value in attributes.items():
            self.nodeAttributes[node][attribute] = value
    def toEdgeMatrix(self):
        matrixSize = len(self.nodeNames)
        edgeMatrix = np.zeros(shape=(matrixSize,matrixSize))
        for fromIndex,nodeNameFrom in enumerate(self.nodeNames):
            for toIndex,nodeNameTo in enumerate(self.nodeNames):
                edgeName = f'{nodeNameFrom}->{nodeNameTo}'
                edgeMatrix[toIndex,fromIndex] = self.edgeDict[edgeName]
        return edgeMatrix
    def addGlobalAttributes(self,attributes):
        for key,value in attributes.items():
            self.globalAttributes[key] = value
    def toAttributeMatrix(self,attribute):
        attributeArr = np.array([self.nodeAttributes[node][attribute] for node in self.nodeNames])
        return np.expand_dims(attributeArr,axis=1)
    def fromEdgeMatrix(self,edgeMatrix):
        if edgeMatrix.shape[0] != edgeMatrix.shape[1]:
            raise ValueError('Matrix must be square')
        if len(self.nodeNames) != edgeMatrix.shape[0]:
            raise ValueError('nodeNames must have same size as matrix')
        edges = {}
        for i in range(edgeMatrix.shape[0]):
            for j in range(edgeMatrix.shape[1]):
                edges[f'{self.nodeNames[i]}->{self.nodeNames[j]}'] = edgeMatrix[j,i]
        self.edgeDict = edges
        return self
    
