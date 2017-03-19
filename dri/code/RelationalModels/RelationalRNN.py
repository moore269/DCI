import numpy as np
import theano
from theano import shared

from code.readData.encodeData import *
import sys

from fuel.streams import DataStream
from fuel.datasets import IndexableDataset

from blocks.extensions import FinishAfter
from code.BlocksModules.myTrackBest import myTrackBest
from fuel.transformers import Padding
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.transformers import Mapping
from collections import defaultdict

import code.graph_helper.graph_helper as graph_helper
from code.BlocksModules.IntStream import *
from code.BlocksModules.FinishIfNoImprovementEpsilonAfter import *

sys.setrecursionlimit(1000000)
"""Create a neural net from a graph G, trainNodes, validationNodes, dim - hidden unit size
"""
class RelationalRNN(object):

    def __init__(self,G=0,trainNodes=0, validationNodes=0, dim=10, batch_size=100, num_epochs=0, save_path='', max_epochs=1000, maxNeighbors=100, 
        attrKey = 'attr', debug = False, load_path='', epsilon=0.0001, useActualLabs=False, onlyLabs=False, usePrevWeights=False, dataAug="none", 
        pageRankOrder="F", sharedName="sharedData", batchesInferences=False, usePro=False, lastH=False):
        self.epsilon=epsilon
        self.attrKey=attrKey
        self.batch_size=batch_size
        self.G=G
        self.useActualLabs=useActualLabs
        self.onlyLabs=onlyLabs
        self.trainNodes=trainNodes
        self.validationNodes = validationNodes
        self.maxNeighbors = maxNeighbors
        self.usePrevWeights = usePrevWeights
        self.dataAug = dataAug
        self.pageRankOrder=pageRankOrder
        self.sharedName=sharedName
        self.batchesInferences=batchesInferences
        self.dim=dim
        self.num_epochs=num_epochs
        self.max_epochs=max_epochs
        self.debug = debug
        self.save_path=save_path
        self.usePro=usePro
        self.lastHH=lastH

        #put data into RNN format
        self.indexData()

        self.sharedData={}
        #share data for gpu
        self.sharedData=self.dataToShare(makeShared=True)
        self.sharedBatch={}
        for key in self.sharedData:
            if key!='nodeID':
                self.sharedBatch[key] = shared(self.sharedData[key][0].get_value(), name="sharedBatch_"+key+"_myinput")
            else:
                self.sharedBatch[key] = self.sharedData[key][0]

    #function used for collective classification when we want pretrained weights
    def resetData(self):
        self.indexData()
        sharedData = self.dataToShare(makeShared=False)
        #reset our data
        for key in sharedData:
            for i in range(0, self.totalBatches):
                if key!='nodeID':
                    self.sharedData[key][i].set_value(sharedData[key][i], borrow=True)
                else:
                    self.sharedData[key][i] = sharedData[key][i]

            #reset sharedbatch as well
            if key!='nodeID':
                self.sharedBatch[key].set_value(self.sharedData[key][0].get_value(borrow=True), borrow=True)
            else:
                self.sharedBatch[key] = self.sharedData[key][0]

    #given a graph G, inputs node v and its neighbors as a sequential input
    def indexData(self):
        labCounts = graph_helper.getLabelCounts(self.G, self.trainNodes+self.validationNodes)
        trainXY, trainIDs = encode_data_VarLen(self.G, self.trainNodes, self.attrKey, self.maxNeighbors, 
            usePrevWeights = self.usePrevWeights, useActualLabs=self.useActualLabs, onlyLabs=self.onlyLabs, useInputX2=self.useInputX2, 
            labCounts=labCounts, dataAug=self.dataAug, pageRankOrder=self.pageRankOrder, usePro=self.usePro, lastH=self.lastHH, nodeIDs=True)
        validationXY,testIDs = encode_data_VarLen(self.G, self.validationNodes, self.attrKey, self.maxNeighbors, 
            labCounts=labCounts, usePrevWeights = self.usePrevWeights, useActualLabs=self.useActualLabs, onlyLabs=self.onlyLabs, useInputX2=self.useInputX2, pageRankOrder=self.pageRankOrder, usePro=self.usePro, lastH=self.lastHH, nodeIDs=True)
        self.input_dimx1 = trainXY['x'][0].shape[1]
        if 'x2' in trainXY:
            self.input_dimx2 = trainXY['x2'].shape[1]

        dataset_train=IndexableDataset(trainXY)
        dataset_valid=IndexableDataset(validationXY)
        self.num_examples_train = dataset_train.num_examples
        self.num_examples_valid = dataset_valid.num_examples
        if self.usePro:
            transpose_stream = self.transpose_streamPro
        else:
            transpose_stream = self.transpose_stream

        self.stream_train = DataStream(dataset=dataset_train, iteration_scheme=ShuffledScheme(examples=dataset_train.num_examples, batch_size=self.batch_size))
        self.stream_train = Padding(self.stream_train, mask_sources=['x'])
        self.stream_train = Mapping(self.stream_train, transpose_stream)

        self.stream_valid = DataStream(dataset=dataset_valid, iteration_scheme=ShuffledScheme(examples=dataset_valid.num_examples, batch_size=self.batch_size))
        self.stream_valid = Padding(self.stream_valid, mask_sources=['x'])
        self.stream_valid = Mapping(self.stream_valid, transpose_stream)

    #when making predictions, replace buffer of data with new test data
    def replaceTestData(self, testNodes, maxNeighbors=1000, maskNames=['x']):
        if self.batchesInferences:
            batch_size = self.batch_size
        else:
            batch_size = 1

        testing, testIDs = encode_data_VarLen(self.G, testNodes, self.attrKey, maxNeighbors, useActualLabs=self.useActualLabs, useInputX2=self.useInputX2, onlyLabs=self.onlyLabs, lastH=self.lastHH, nodeIDs=True)
        dataset_test=IndexableDataset(testing)
        self.stream_test = DataStream(dataset=dataset_test, iteration_scheme=SequentialScheme(examples=dataset_test.num_examples, batch_size=batch_size))
        #add masks, have to do individually to avoid all dimensions must be equal error
        #write own padding transformer, their's sucks ...
        self.stream_test = Padding(self.stream_test, mask_sources=maskNames)
        #transpose them for rnn input
        self.stream_test = Mapping(self.stream_test, self.transpose_streamTest)
        self.num_examples_test = dataset_test.num_examples

        #replace shareddata with test_all data
        self.test_all, names = self.iterateShared(self.stream_test, makeShared=False, name="test")

        #if we are doing test in batches
        if self.batchesInferences:
            for key in self.test_all:
                totalTestBatches = len(self.test_all[key])
                if key!='nodeID':
                    for i in range(0, totalTestBatches):
                        #if test data has more batches, we add more to shared data list
                        #else we just reset
                        if i>= self.totalBatches:
                            newKey=key+'_myinput'
                            self.sharedData[key].append(shared(self.test_all[key][i], name=self.sharedName+'_'+newKey+'_test_'+str(i)))
                        else:
                            self.sharedData[key][i].set_value(self.test_all[key][i], borrow=True)

                    self.sharedBatch[key].set_value(self.sharedData[key][0].get_value(borrow=True), borrow=True)

            self.stream_test_int = IntStream(0, totalTestBatches, 1, 'int_stream')

    #given test nodes, make predictions
    def makePredictions(self, testNodes, maxNeighbors=1000, changeLabel=True, maskNames=['x'], lastH=False):
        self.replaceTestData(testNodes, maxNeighbors, maskNames)

        #get predictions and score

        
        if lastH:
            accuracy,predictions, hiddenRep = self.predictBAE(changeLabel=changeLabel, lastH=lastH)
            return (accuracy,predictions, hiddenRep)
        else:
            accuracy,predictions = self.predictBAE(changeLabel=changeLabel, lastH=lastH)
            return (accuracy, predictions)

    #convert data to shared data so that everything is on the gpu
    def dataToShare(self, makeShared):

        sharedDataTrain, sharedNamesTrain = self.iterateShared(self.stream_train, makeShared=makeShared, name="train")
        sharedDataValid, sharedNamesValid = self.iterateShared(self.stream_valid, makeShared=makeShared, name="valid")
        self.sharedNames = sharedNamesTrain+sharedNamesValid

        #combine shared data
        sharedData={}
        for key in sharedDataTrain:
            sharedData[key]=sharedDataTrain[key]+sharedDataValid[key]
 
        #now create new streams
        totalBatchesTrain = len(sharedDataTrain[key])
        totalBatchesValid = len(sharedDataValid[key])
        self.totalBatches=totalBatchesTrain+totalBatchesValid

        self.stream_train_int = IntStream(0, totalBatchesTrain, 1, 'int_stream')
        self.stream_valid_int = IntStream(totalBatchesTrain, totalBatchesValid, 1, 'int_stream')
        return sharedData

    #iterate over data stream and make into shared data
    def iterateShared(self, stream, makeShared=True, name="train"):
        names=[]
        sharedData = defaultdict(lambda:[])
        for i, batch in enumerate(stream.get_epoch_iterator(as_dict=True)):
            for key in batch:
                newKey=key+'_myinput'
                namePost = self.sharedName+'_'+newKey+'_'+name+"_"+str(i)
                names.append(namePost)
                if makeShared and key!='nodeID':
                    sharedData[key].append(shared(batch[key], name=namePost)) 
                else:
                    sharedData[key].append(batch[key]) 

        return (sharedData, names)

    #predict in batches of data, faster since we utilize gpu but perhaps less accurate
    def predInBatches(self, changeLabel):
        err1=0.0
        err0=0.0
        count1=0
        count0=0
        predictions={}

        epoch_iterator = (self.stream_test_int.get_epoch_iterator(as_dict=True))
        while True:
            try:
                batch = next(epoch_iterator)
                batchInd = batch['int_stream_From']
                #switch batches before running
                for key in self.test_all:
                    if key!='nodeID':
                        self.sharedBatch[key].set_value(self.sharedData[key][batchInd].get_value(borrow=True), borrow=True)

                preds = self.f()
                #print(self.test_all['y'][batchInd].shape)
                batchLen = self.test_all['y'][batchInd].shape[0]
                #iterate through a batch
                for i in range(0, batchLen):
                    nodeID = self.test_all['nodeID'][batchInd][i]
                    actual = self.test_all['y'][batchInd][i][0]
                    pred = preds[i][0]

                    predictions[nodeID]=pred
                    if changeLabel:
                        self.G.node[nodeID]['dynamic_label']=[pred]
                    self.G.node[nodeID]['pred_label']=[pred]

                    if actual==1:
                        err1 += (1-pred)
                        count1 += 1
                    elif actual == 0:
                        err0 += pred
                        count0 += 1

            except StopIteration:
                break  
        return (err1, err0, count1, count0, predictions) 
    
    #predict per example
    def predInExample(self, changeLabel, lastH=False): 
        err1=0.0
        err0=0.0
        count1=0
        count0=0
        predictions={}
        hiddenRep={}
        numExamples = len(self.test_all['nodeID'])
        for i in range(0, numExamples):
            #switch batches before running
            for key in self.test_all:
                if key!='nodeID':
                    self.sharedBatch[key].set_value(self.test_all[key][i], borrow=True)
                else:
                    self.sharedBatch[key] = self.test_all[key][i]

            preds = self.f()


            nodeID = self.test_all['nodeID'][i][0]
            actual = self.test_all['y'][i][0][0]
            pred = preds[0][0]
            predictions[nodeID]=pred

            if lastH:
                lH= self.lastH()[0]
                hiddenRep[nodeID]=lH.tolist()

            if changeLabel:
                self.G.node[nodeID]['dynamic_label']=[pred]
            self.G.node[nodeID]['pred_label']=[pred]

            if actual==1:
                err1 += (1-pred)
                count1 += 1
            elif actual == 0:
                err0 += pred
                count0 += 1

        if lastH:
            return (err1, err0, count1, count0, predictions, hiddenRep)  
        else:
            return (err1, err0, count1, count0, predictions)  

    #grab activations per train or valid set
    def generateHidden(self, name="train"):
        if name=="train":
            tSet = self.stream_train_int
        elif name=="valid":
            tSet = self.stream_valid_int

        hiddenRep={}
        for batch in tSet.get_epoch_iterator(as_dict=True):
            i = batch['int_stream_From']
            #switch batches before running
            for key in self.sharedBatch:
                if key!='nodeID':
                    self.sharedBatch[key].set_value(self.sharedData[key][i].get_value(borrow=True), borrow=True)
                else:
                    self.sharedBatch[key] = self.sharedData[key][i]

            lastH = self.lastH()
            for j in range(0, lastH.shape[0]):
                nodeID = self.sharedBatch['nodeID'][j]
                hiddenRep[nodeID]=lastH[j].tolist()

        return hiddenRep 

    #output BAE scores and predictions    
    def predictBAE(self, changeLabel=True, lastH=False):
        if self.batchesInferences:
            err1, err0, count1, count0, predictions = self.predInBatches(changeLabel)
        else:
            if lastH:
                err1, err0, count1, count0, predictions, hiddenRep = self.predInExample(changeLabel, lastH=lastH)
            else:
                err1, err0, count1, count0, predictions = self.predInExample(changeLabel, lastH=lastH)

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

        if lastH:
            return (BAE, predictions, hiddenRep)
        else:
            return (BAE, predictions)

    # track the best model evaluated on the validation set
    def track_best(self, channel, cg):
        tracker = myTrackBest(channel)
        finishNoimprove = FinishIfNoImprovementEpsilonAfter(channel+'_best_so_far', epochs=self.num_epochs, epsilon=self.epsilon)
        finishAfter = FinishAfter(after_n_epochs=self.max_epochs)

        return [tracker, finishNoimprove, finishAfter]

import code.readData.readData as readData
#test if we can load model and apply it, for loading the model, pass in G, batch_size, attrKey, load_path
def test1(fName):
    percentTest=0.35
    percentValidation=0.2
    load_path = 'models/Rnn_vary_training_A_BAE_amazon_DVD_20000_trial_0_fold_0.pkl'
    G = readData.readDataset("data/", fName) 
    (testNodes, validationNodes, rest) = readData.splitNodes(G.nodes(),percentTest, percentValidation)
    rnn = RelationalRNN(G=G, batch_size=10, attrKey='attr', load_path = load_path)
    accuracy = rnn.makePredictions(testNodes)
    print("accuracy: "+str(accuracy))

if __name__ == "__main__":
    test1("amazon_DVD_20000")

