from sklearn.linear_model import LogisticRegression
import sys
import numpy as np

#combine node2vec representation with attributes and perform classification
class node2vecLR(object):
    def __init__(self, input_name, train_nodes, validation_nodes, test_nodes):
        self.input_name=input_name
        self.lr = LogisticRegression()
        self.train_all_attr= self.readInput(train_nodes+validation_nodes)
        self.train_all_y = self.readLabels(train_nodes+validation_nodes)
        self.train_attr= self.readInput(train_nodes)
        self.train_y = self.readLabels(train_nodes) 

        self.valid_attr= self.readInput(validation_nodes)
        self.valid_y = self.readLabels(validation_nodes)   
        self.test_attr= self.readInput(test_nodes)
        self.test_y = self.readLabels(test_nodes)             

    def readInput(self, nodes):
        nodes = set(nodes)
        f = open(self.input_name+".attr")
        attrs = {}
        for line in f:
            fields = line.replace("\n", "").split("::")
            nodeID = int(fields[0])

            attr = map(float, fields[1:])
            if nodeID in nodes:
                attrs[nodeID]=attr

        f = open(self.input_name+".nattr")

        for i, line in enumerate(f):
            if i==0:
                continue
            fields = line.replace("\n", "").split()
            nodeID = int(fields[0])
            attr = map(float, fields[1:])
            if nodeID in nodes:
                attrs[nodeID]=attrs[nodeID] + attr

        attrsSorted = [attrs[key] for key in sorted(attrs)]

        return np.array(attrsSorted)


    def readLabels(self, nodes):
        nodes = set(nodes)
        f = open(self.input_name+".lab")
        labs = {}
        for line in f:
            fields = line.replace("\n", "").split("::")
            nodeID = int(fields[0])
            lab = float(fields[1])
            if nodeID in nodes:
                labs[nodeID]=lab
        labsSorted = [labs[key] for key in sorted(labs)]
        return np.ravel(np.array(labsSorted))


    def train(self):  
        self.lr.fit(self.train_all_attr, self.train_all_y)

    def predictBAE(self, testSet = "test"):
        if testSet=="train":
            test_attr = self.train_attr
            test_y = self.train_y
        elif testSet=="valid":
            test_attr = self.valid_attr
            test_y = self.valid_y
        elif testSet=="test":
            test_attr = self.test_attr
            test_y = self.test_y


        err1=0.0
        err0=0.0
        count1=0
        count0=0
        probs_posneg = self.lr.predict_proba(test_attr)
        for i in range(0, probs_posneg.shape[0]):
            pred = probs_posneg[i][1]
            actual = test_y[i]
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
            return 1.0

        BAE=(err1+err0)/divideBy

        return BAE
