import numpy as np
import theano
import sys

from code.BlocksModules.SwitchSharedReferences import SwitchSharedReferences, DataStreamMonitoringShared

from blocks.bricks import Linear, Logistic, Rectifier, Tanh
from blocks.bricks.recurrent import LSTM
from blocks.bricks.cost import SquaredError, AbsoluteError
from blocks.initialization import Constant, IsotropicGaussian
from blocks.algorithms import GradientDescent, RMSProp
from blocks.graph import ComputationGraph
from blocks.extensions import Printing
from blocks.extensions.monitoring import TrainingDataMonitoring

from blocks.main_loop import MainLoop
from blocks.model import Model

from RelationalRNN import RelationalRNN
from collections import defaultdict
from theano import shared
from code.BlocksModules.IntStream import *

from blocks.config import config
from blocks.log import BACKENDS

from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks.recurrent import SimpleRecurrent, BaseRecurrent, recurrent
from theano import tensor as T

sys.setrecursionlimit(1000000)

class RNNwMini(BaseRecurrent):
    def __init__(self, dim, mini_dim, summary_dim, **kwargs):
        super(RNNwMini, self).__init__(**kwargs)
        self.dim = dim
        self.mini_dim=mini_dim
        self.summary_dim=summary_dim

        self.recurrent_layer = SimpleRecurrent(
            dim=self.summary_dim, activation=Rectifier(), name='recurrent_layer',
            weights_init=IsotropicGaussian(), biases_init=Constant(0.0))
        self.mini_recurrent_layer = SimpleRecurrent(
            dim=self.mini_dim, activation=Rectifier(), name='mini_recurrent_layer',
            weights_init=IsotropicGaussian(), biases_init=Constant(0.0))
        
        self.mini_to_main = Linear(self.dim+self.mini_dim, self.summary_dim, name='mini_to_main',
                        weights_init=IsotropicGaussian(),
                        biases_init=Constant(0.0))
        self.children = [self.recurrent_layer,
                         self.mini_recurrent_layer, self.mini_to_main]

    @recurrent(sequences=['x', 'xmini'], contexts=[],
               states=['states'],
               outputs=['states'])
    def apply(self, x, xmini, states=None):
        mini_h_all = self.mini_recurrent_layer.apply(
            inputs=xmini, states=None, iterate=True)
        #grab last hidden state
        mini_h = mini_h_all[-1]

        combInput = T.concatenate([x, mini_h], axis=1 )
        combTransform = self.mini_to_main.apply(combInput)

        h = self.recurrent_layer.apply(
            inputs=combTransform, states=states, iterate=False)

        return h

    def get_dim(self, name):
        dim = 1
        if name == 'x':
            dim=self.dim
        elif name == 'states':
            dim=self.summary_dim
        else:
            dim = super(RNNwMini, self).get_dim(name)
        return dim


"""
RelationalRNNwMini is a subset of RelationalRNN
It must provide train function, main_loop for collective inference, and transpose methods for transforming input data
"""
class RelationalRNNwMini(RelationalRNN):

    def __init__(self,G,trainNodes, validationNodes, dim, mini_dim, summary_dim, input_dimx, input_dimxmini, **kwargs):
        RelationalRNN.__init__(self, G=G,trainNodes=trainNodes, validationNodes=validationNodes, **kwargs)
        self.dim=dim
        self.mini_dim=mini_dim
        self.summary_dim=summary_dim
        self.input_dimx=input_dimx
        self.input_dimxmini=input_dimxmini

    #train our q net
    #this part has theano + blocks statements
    def train(self):

        x = self.sharedBatch['x']
        x.name = 'x_myinput'
        xmini = self.sharedBatch['xmini']
        xmini.name = 'xmini_myinput'
        y = self.sharedBatch['y']
        y.name = 'y_myinput'

        # we need to provide data for the LSTM layer of size 4 * ltsm_dim, see
        # LSTM layer documentation for the explanation
        x_to_h = Linear(self.input_dimx, self.dim, name='x_to_h',
                        weights_init=IsotropicGaussian(),
                        biases_init=Constant(0.0))
        xmini_to_h = Linear(self.input_dimxmini, self.mini_dim, name='xmini_to_h',
                        weights_init=IsotropicGaussian(),
                        biases_init=Constant(0.0))

        rnnwmini = RNNwMini(dim=self.dim, mini_dim=self.mini_dim, summary_dim=self.summary_dim)

        h_to_o = Linear(self.summary_dim, 1, name='h_to_o',
                        weights_init=IsotropicGaussian(),
                        biases_init=Constant(0.0))

        x_transform = x_to_h.apply(x)
        xmini_transform = xmini_to_h.apply(xmini)

        h = rnnwmini.apply(x=x_transform, xmini=xmini_transform)

        # only values of hidden units of the last timeframe are used for
        # the classification
        y_hat = h_to_o.apply(h[-1])
        #y_hat = Logistic().apply(y_hat)

        cost = SquaredError().apply(y, y_hat)
        cost.name = 'cost'

        rnnwmini.initialize()
        x_to_h.initialize()
        xmini_to_h.initialize()
        h_to_o.initialize()

        self.f = theano.function(inputs = [], outputs=y_hat)

        #print("self.f === ")
        #print(self.f())
        #print(self.f().shape)
        #print("====")

        self.cg = ComputationGraph(cost)
        m = Model(cost)

        algorithm = GradientDescent(cost=cost, parameters=self.cg.parameters,
                                    step_rule=RMSProp(learning_rate=0.01), on_unused_sources='ignore')
        valid_monitor = DataStreamMonitoringShared(variables=[cost], data_stream=self.stream_valid_int, prefix="valid", sharedBatch=self.sharedBatch, sharedData=self.sharedData)
        train_monitor = TrainingDataMonitoring(variables=[cost], prefix="train",
                                               after_epoch=True)

        sharedVarMonitor = SwitchSharedReferences(self.sharedBatch, self.sharedData)
        tBest = self.track_best('valid_cost', self.cg) 
        self.tracker = tBest[0]
        extensions = [sharedVarMonitor, valid_monitor] + tBest

        if self.debug:
            extensions.append(Printing())

        self.algorithm = algorithm
        self.extensions=extensions
        self.model = m
        self.mainloop = MainLoop(self.algorithm, self.stream_train_int, extensions=self.extensions, model=self.model)
        self.main_loop(True)

    #call this to train again
    #modify the blocks mainloop logic
    def main_loop(self, first):

        log_backend = config.log_backend
        self.mainloop.log = BACKENDS[log_backend]()
        if first:
            self.mainloop.status['training_started'] = False
        else:
            self.mainloop.status['training_started'] = True
        self.mainloop.status['epoch_started'] = False
        self.mainloop.status['epoch_interrupt_received'] = False
        self.mainloop.status['batch_interrupt_received'] = False
        self.mainloop.run()
        #make sure we have the best model
        self.tracker.set_best_model()

    #function empty to prevent from doing extra work
    def indexData(self):
        self.stream_train = self.readSynthetic('train')
        self.stream_valid = self.readSynthetic('valid')
        self.stream_test = self.readSynthetic('test')

    #helper function to read synthetic data
    def readSynthetic(self, name):
        stream = {'x':None,'xmini':None,'y':None}
        stream['x'] = np.load("../experiments/data/RNNwMiniSynthetic_"+name+"_x.npy")
        stream['xmini'] = np.load("../experiments/data/RNNwMiniSynthetic_"+name+"_xmini.npy")
        stream['y'] = np.load("../experiments/data/RNNwMiniSynthetic_"+name+"_y.npy")
        return stream

    #iterate over data stream and make into shared data
    def iterateShared(self, stream, makeShared=True, name="train"):
        names=[]
        sharedData = defaultdict(lambda:[])
        for key in stream:
            numBatches = stream[key].shape[0]
            for i in range(0, numBatches):
                data = stream[key][i]
                if len(stream[key][i].shape)==1:
                    data=np.reshape(data, (data.shape[0], 1))

                newKey=key+'_myinput'
                namePost = self.sharedName+'_'+newKey+'_'+name+"_"+str(i)
                names.append(namePost)
                if makeShared:
                    sharedData[key].append(shared(data, name=namePost)) 
                else:
                    sharedData[key].append(data) 

        return (sharedData, names)

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

    #when making predictions, replace buffer of data with new test data
    #for now, only use test data
    def replaceTestData(self, testNodes, useInputX2=False, maxNeighbors=1000, maskNames=['x']):
        if self.batchesInferences:
            batch_size = self.batch_size
        else:
            batch_size = 1

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


    #output BAE scores and predictions    
    def predictBAE(self, changeLabel=True):
        if self.batchesInferences:
            MSE, predictions = self.predInBatches(changeLabel)

        return (MSE, predictions)


    #predict in batches of data, faster since we utilize gpu but perhaps less accurate
    def predInBatches(self, changeLabel):
        err1=0.0
        err0=0.0
        count1=0
        count0=0
        predictions={}
        sqError=0
        total=0
        epoch_iterator = (self.stream_test_int.get_epoch_iterator(as_dict=True))
        while True:
            try:
                batch = next(epoch_iterator)
                batchInd = batch['int_stream_From']
                #switch batches before running
                for key in self.sharedBatch:
                    self.sharedBatch[key].set_value(self.sharedData[key][batchInd].get_value(borrow=True), borrow=True)

                preds = self.f()
                batchLen = self.test_all['y'][batchInd].shape[0]
                print(self.test_all['y'][batchInd].shape)
                #iterate through a batch
                for i in range(0, batchLen):
                    total+=1
                    actual = self.test_all['y'][batchInd][i][0]
                    pred = preds[i][0]
                    print("actual and pred")
                    print(actual)
                    print(pred)
                    print("====")
                    sqError+=(actual-pred)*(actual-pred)

            except StopIteration:
                break  
        MSE=sqError/total
        return (MSE, predictions) 

