import cPickle as pickle
import code.readData.readData as readData
from sklearn.linear_model import LogisticRegression
from config import config
import code.graph_helper.graph_helper as graph_helper
import numpy as np

class RelationalLRAVG(object):

    def __init__(self, G, testNodes=None, netType='', rnnCollective=False, dynamLabel=False, **kwargs):
        self.GFirst=G
        self.attr = 'attr'
        self.trainNodes = kwargs['trainNodes']
        self.validationNodes = kwargs['validationNodes']
        self.testNodes = testNodes
        self.netType=netType
        print("LRAVG Type: "+self.netType)

        #finally, we can generate our training, validation, and testing data
        #trainDataLR is for LR
        self.trainDataLR = self.createData(kwargs['trainNodes']+kwargs['validationNodes'])
        
        #train, valid, and test
        self.trainData = self.createData(kwargs['trainNodes'])
        self.validData = self.createData(kwargs['validationNodes'])
        self.testData = self.createData(testNodes)
        self.lr = LogisticRegression()


    def train(self):  
        self.lr.fit(self.trainDataLR[0], self.trainDataLR[1])

    def makePredictions(self, testNodes, maxNeighbors=1000, changeLabel=True, maskNames=['x'], lastH=False):
        if testNodes==self.trainNodes:
            testSet = "train"
        elif testNodes==self.validationNodes:
            testSet = "valid"
        elif testNodes==self.testNodes:
            testSet = "test"
        else:
            print("must provide existing train,valid, or test set")
            sys.exit(0)
        accuracy,predictions = self.predictBAE(testSet)
        return (accuracy, predictions)


    def predictBAE(self, testSet = "test"):
        if testSet=="test":
            testData = self.testData
        elif testSet=="valid":
            testData = self.validData
        elif testSet=="train":
            testData = self.trainData
        else:
            print("must provide existing train,valid, or test set")
            sys.exit(0) 

        err1=0.0
        err0=0.0
        count1=0
        count0=0
        predictions={}
        probs_posneg = self.lr.predict_proba(testData[0])
        for i in range(0, probs_posneg.shape[0]):
            pred = probs_posneg[i][1]
            actual = testData[1][i][0]
            nodeID = testData[2][i]
            predictions[nodeID]=pred
            if actual==1:
                err1 += (1-pred)
                count1 += 1
            elif actual == 0:
                err0 += pred
                count0 += 1

        divideBy=0
        if count1!=0:
            err1=err1/count1
            divideBy+=1
        if count0!=0:
            err0=err0/count0
            divideBy+=1

        if divideBy==0:
            return (1.0, predictions)

        BAE=(err1+err0)/divideBy

        return (BAE, predictions)


    def avgNeighbors(self, node, attrVector, attrVectorLen):
        neighborAttr = []
        if attrVector!=None:
            neighborAttr.append(attrVector)

        for neighbor in self.GFirst.neighbors(node):
            neighborAttr.append(self.GFirst.node[neighbor][self.attr])

        if len(neighborAttr)==0:
            neighborAttr.append([0]*attrVectorLen)
        AvgAttr = np.mean(np.array(neighborAttr), axis=0)
        return AvgAttr

    def createData(self, nodes):
        examplesX=[]
        examplesY=[]
        examplesIDs=[]
        for node in nodes:
            examplesIDs.append(node)
            attrVector = self.GFirst.node[node][self.attr]
            if self.netType=="average":
                AvgAttr = self.avgNeighbors(node, attrVector, len(attrVector))
                examplesX.append(AvgAttr)
            elif self.netType=="averageOrig":
                AvgAttr = self.avgNeighbors(node, None, len(attrVector)).tolist()
                examplesX.append(np.array(attrVector+AvgAttr))
            else:
                examplesX.append(np.array(attrVector))
            examplesY.append(self.GFirst.node[node]['label'])
        return (examplesX, examplesY, examplesIDs)
  

if __name__ == "__main__":
    unitTest1()