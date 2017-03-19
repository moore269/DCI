import numpy as np
import theano
import sys

from code.BlocksModules.SwitchSharedReferences import SwitchSharedReferences, DataStreamMonitoringShared
from code.readData.encodeData import *

from blocks.bricks import Linear, Logistic, Rectifier, Tanh, MLP
from blocks.bricks.recurrent import LSTM
from blocks.bricks.cost import BinaryCrossEntropy
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

class LSTMwMini(BaseRecurrent):
    def __init__(self, dim, mini_dim, summary_dim, **kwargs):
        super(LSTMwMini, self).__init__(**kwargs)
        self.dim = dim
        self.mini_dim=mini_dim
        self.summary_dim=summary_dim


        self.recurrent_layer = LSTM(
            dim=self.summary_dim, activation=Rectifier(), name='recurrent_layer',
            weights_init=IsotropicGaussian(), biases_init=Constant(0.0))
        self.mini_recurrent_layer = LSTM(
            dim=self.mini_dim, activation=Rectifier(), name='mini_recurrent_layer',
            weights_init=IsotropicGaussian(), biases_init=Constant(0.0))
        
        self.mini_to_main = Linear(self.dim+self.mini_dim, self.summary_dim, name='mini_to_main',
                        weights_init=IsotropicGaussian(),
                        biases_init=Constant(0.0))
        self.mini_to_main2 = Linear(self.summary_dim, self.summary_dim*4, name='mini_to_main2',
                        weights_init=IsotropicGaussian(),
                        biases_init=Constant(0.0))

        self.children = [self.recurrent_layer,
                         self.mini_recurrent_layer, self.mini_to_main, self.mini_to_main2]

    @recurrent(sequences=['x', 'xmini', 'xmask', 'xmini_mask'], contexts=[],
               states=['states', 'cells'],
               outputs=['states', 'cells'])
    def apply(self, x, xmini, xmask=None, xmini_mask=None, states=None, cells=None):
        mini_h_all, c1 = self.mini_recurrent_layer.apply(
            inputs=xmini, mask=xmini_mask, states=None, iterate=True)
        #grab last hidden state
        mini_h = mini_h_all[-1]

        combInput = T.concatenate([x, mini_h], axis=1 )
        combTransform = self.mini_to_main.apply(combInput)
        combTransform2 = self.mini_to_main2.apply(combTransform)

        h, c2 = self.recurrent_layer.apply(
            inputs=combTransform2, mask=xmask, states=states, cells=cells, iterate=False)

        return h, c2

    def get_dim(self, name):
        dim = 1
        if name == 'x':
            dim=self.dim
        elif name in ['states', 'cells']:
            dim=self.summary_dim
        else:
            dim = super(LSTMwMini, self).get_dim(name)
        return dim


"""
RelationalLSTMwMini is a subset of RelationalRNN
It must provide train function, main_loop for collective inference, and transpose methods for transforming input data
"""
class RelationalLSTMwMini(RelationalRNN):

    def __init__(self,G,trainNodes, validationNodes, dim, mini_dim, summary_dim, input_dimx=2, input_dimxmini=2, maxNeighbors2=1000, perturb=False, HogunVarLen=False, **kwargs):
        self.maxNeighbors2 = maxNeighbors2
        self.perturb=perturb
        self.HogunVarLen=HogunVarLen
        RelationalRNN.__init__(self, G=G,trainNodes=trainNodes, validationNodes=validationNodes, **kwargs)
        self.dim=dim
        self.mini_dim=mini_dim
        self.summary_dim=summary_dim
        #self.input_dimx=input_dimx
        #self.input_dimxmini=input_dimxmini

    #train our q net
    #this part has theano + blocks statements
    def train(self):

        #print(self.sharedBatch.keys())
        x = self.sharedBatch['x']
        x.name = 'x_myinput'
        xmask = self.sharedBatch['xmask']
        xmask.name = 'xmask_myinput'
        xmini = self.sharedBatch['xmini']
        xmini.name = 'xmini_myinput'
        xmini_mask = self.sharedBatch['xmini_mask']
        xmini_mask.name = 'xmini_mask_myinput'
        y = self.sharedBatch['y']
        y.name = 'y_myinput'
        xattr = self.sharedBatch['xattr']
        xattr.name = 'xattr_myinput'

        # we need to provide data for the LSTM layer of size 4 * ltsm_dim, see
        # LSTM layer documentation for the explanation
        x_to_h = Linear(self.input_dimx, self.dim, name='x_to_h',
                        weights_init=IsotropicGaussian(),
                        biases_init=Constant(0.0))
        xmini_to_h = Linear(self.input_dimxmini, self.mini_dim*4, name='xmini_to_h',
                        weights_init=IsotropicGaussian(),
                        biases_init=Constant(0.0))

        comb_to_h = Linear(self.input_dimxattr+self.summary_dim, self.input_dimxattr+self.summary_dim, name='comb_to_h',
                        weights_init=IsotropicGaussian(),
                        biases_init=Constant(0.0))

        lstmwmini = LSTMwMini(dim=self.dim, mini_dim=self.mini_dim, summary_dim=self.summary_dim)

        mlp = MLP(activations=[Rectifier()], dims=[self.summary_dim+self.input_dimxattr, self.summary_dim],
            weights_init=IsotropicGaussian(), biases_init=Constant(0.0))

        h_to_o = Linear(self.summary_dim, 1, name='h_to_o',
                        weights_init=IsotropicGaussian(),
                        biases_init=Constant(0.0))

        x_transform = x_to_h.apply(x)
        xmini_transform = xmini_to_h.apply(xmini)

        h, c = lstmwmini.apply(x=x_transform, xmini=xmini_transform, xmask=xmask, xmini_mask=xmini_mask)


        attr_and_rnn = T.concatenate([xattr, h[-1]], axis=1 )

        #self.f = theano.function(inputs = [], outputs=attr_and_rnn)
        #print(self.summary_dim)
        #print(self.input_dimx)
        #print("self.f === ")
        #print(self.f())
        #print(self.f().shape)
        #print("====")
        comb_transform = comb_to_h.apply(attr_and_rnn)
        mlp_transform = mlp.apply(comb_transform)

        # only values of hidden units of the last timeframe are used for
        # the classification
        #y_hat = h_to_o.apply(h[-1])
        y_hat = h_to_o.apply(mlp_transform)
        y_hat = Logistic().apply(y_hat)

        cost = BinaryCrossEntropy().apply(y, y_hat)
        cost.name = 'cost'

        lstmwmini.initialize()
        x_to_h.initialize()
        xmini_to_h.initialize()
        comb_to_h.initialize()
        mlp.initialize()
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
        #self.stream_train_synthetic = self.readSynthetic('train')
        #self.stream_valid_synthetic = self.readSynthetic('valid')
        #self.stream_test = self.readSynthetic('test')
        if not self.HogunVarLen:
            self.stream_train = encode_data_VarLenMini(self.G, self.trainNodes, self.attrKey, self.maxNeighbors, self.maxNeighbors2,
                perturb=self.perturb, usePrevWeights = self.usePrevWeights, useActualLabs=self.useActualLabs, onlyLabs=self.onlyLabs, batch_size=self.batch_size)
            self.stream_valid = encode_data_VarLenMini(self.G, self.validationNodes, self.attrKey, self.maxNeighbors, self.maxNeighbors2,
                perturb=self.perturb, usePrevWeights = self.usePrevWeights, useActualLabs=self.useActualLabs, onlyLabs=self.onlyLabs, batch_size=self.batch_size)
        else:
            self.stream_train = encode_data_VarLenMiniHogun(self.G, self.trainNodes, self.attrKey, self.maxNeighbors, self.maxNeighbors2,
                perturb=self.perturb, usePrevWeights = self.usePrevWeights, useActualLabs=self.useActualLabs, onlyLabs=self.onlyLabs, batch_size=self.batch_size)
            self.stream_valid = encode_data_VarLenMiniHogun(self.G, self.validationNodes, self.attrKey, self.maxNeighbors, self.maxNeighbors2,
                perturb=self.perturb, usePrevWeights = self.usePrevWeights, useActualLabs=self.useActualLabs, onlyLabs=self.onlyLabs, batch_size=self.batch_size)
        
        #reset input dimensions
        self.input_dimx = self.stream_train['x'][0].shape[-1]
        self.input_dimxattr = self.stream_train['xattr'][0].shape[-1]

        self.input_dimxmini = self.stream_train['xmini'][0].shape[-1]

        #self.stream_test = readData.encode_data_VarLenMini(self.G, self.testNodes, self.attrKey, self.maxNeighbors, 
        #    usePrevWeights = self.usePrevWeights, useActualLabs=self.useActualLabs, onlyLabs=self.onlyLabs, batch_size=self.batch_size)

    #x:         numBatches, n1len, batchsize, attr1
    #xmask:     numBatches, n1len, batchsize
    #xmini:     numBatches, n1len, n2len, batchsize, attr2
    #xminimask: numBatches, n1len, n2len, batchsize



    #helper function to read synthetic data
    def readSynthetic(self, name):
        stream = {'x':None,'xmini':None,'y':None}
        stream['x'] = np.load("../experiments/data/RNNwMiniSynthetic_"+name+"_x.npy")
        stream['xmask'] = np.random.randint(0, 2, stream['x'].shape[:3]).astype(theano.config.floatX)
        stream['xmini'] = np.load("../experiments/data/RNNwMiniSynthetic_"+name+"_xmini.npy")
        stream['xmini_mask'] = np.random.randint(0, 2, stream['xmini'].shape[:4]).astype(theano.config.floatX)
        stream['y'] = np.load("../experiments/data/RNNwMiniSynthetic_"+name+"_y.npy")
        return stream

    #iterate over data stream and make into shared data
    def iterateShared(self, stream, makeShared=True, name="train"):
        names=[]
        sharedData = defaultdict(lambda:[])
        for key in stream:
            numBatches = len(stream[key])
            for i in range(0, numBatches):
                data = stream[key][i]
                if key!='nodeID' and len(data.shape)==1:
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
    def replaceTestData(self, testNodes, useInputX2=False, maxNeighbors=1000, maxNeighbors2=1000, maskNames=['x']):
        if self.batchesInferences:
            batch_size = self.batch_size
        else:
            batch_size = 1
        if not self.HogunVarLen:
            stream_test = encode_data_VarLenMini(self.G, testNodes, self.attrKey, maxNeighbors, maxNeighbors2,
                usePrevWeights = self.usePrevWeights, useActualLabs=self.useActualLabs, onlyLabs=self.onlyLabs, batch_size=batch_size)
        else:
            stream_test = encode_data_VarLenMiniHogun(self.G, testNodes, self.attrKey, maxNeighbors, maxNeighbors2,
                usePrevWeights = self.usePrevWeights, useActualLabs=self.useActualLabs, onlyLabs=self.onlyLabs, batch_size=batch_size)
        #replace shareddata with test_all data
        self.test_all, names = self.iterateShared(stream_test, makeShared=False, name="test")

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
    """def predictBAE(self, changeLabel=True):
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
        return (MSE, predictions) """

