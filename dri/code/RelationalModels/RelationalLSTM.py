import numpy as np
import theano
import sys

from code.BlocksModules.SwitchSharedReferences import SwitchSharedReferences, DataStreamMonitoringShared

from blocks.bricks import Linear, Logistic, Rectifier
from blocks.bricks.recurrent import LSTM
from blocks.bricks.cost import BinaryCrossEntropy
from code.BlocksModules.costs import BinaryCrossEntropyProp

from blocks.initialization import Constant, IsotropicGaussian
from blocks.algorithms import GradientDescent, RMSProp
from blocks.graph import ComputationGraph
from blocks.extensions import Printing
from blocks.extensions.monitoring import TrainingDataMonitoring

from blocks.main_loop import MainLoop
from blocks.model import Model

from RelationalRNN import RelationalRNN
from blocks.config import config
from blocks.log import BACKENDS

sys.setrecursionlimit(1000000)

"""
RelationalLSTM is a subset of RelationalRNN
It must provide train function, main_loop for collective inference, and transpose methods for transforming input data
"""
class RelationalLSTM(RelationalRNN):

    def __init__(self,G,trainNodes, validationNodes, **kwargs):
        self.useInputX2=False
        RelationalRNN.__init__(self, G=G,trainNodes=trainNodes, validationNodes=validationNodes, **kwargs)

    #train our neural net
    #this part has theano + blocks statements
    def train(self):

        x = self.sharedBatch['x']
        x.name = 'x_myinput'
        x_mask = self.sharedBatch['x_mask']
        x_mask.name = 'x_mask_myinput'
        y = self.sharedBatch['y']
        y.name = 'y_myinput'

        if self.usePro:
            proportion = self.sharedBatch['pro']
            proportion.name = 'pro'

        # we need to provide data for the LSTM layer of size 4 * ltsm_dim, see
        # LSTM layer documentation for the explanation
        x_to_h = Linear(self.input_dimx1, self.dim * 4, name='x_to_h',
                        weights_init=IsotropicGaussian(),
                        biases_init=Constant(0.0))
        lstm = LSTM(self.dim, name='lstm',
                    weights_init=IsotropicGaussian(),
                    biases_init=Constant(0.0))
        h_to_o = Linear(self.dim, 1, name='h_to_o',
                        weights_init=IsotropicGaussian(),
                        biases_init=Constant(0.0))

        x_transform = x_to_h.apply(x)
        h, c = lstm.apply(x_transform, mask=x_mask)

        # only values of hidden units of the last timeframe are used for
        # the classification
        y_hat = h_to_o.apply(h[-1])
        y_hat = Logistic().apply(y_hat)
        
        if self.usePro:
            cost = BinaryCrossEntropyProp().apply(y, y_hat, proportion)
        else:
            cost = BinaryCrossEntropy().apply(y, y_hat)
        
        cost.name = 'cost'

        lstm.initialize()
        x_to_h.initialize()
        h_to_o.initialize()

        self.f = theano.function(inputs = [], outputs=y_hat)
        self.lastH = theano.function(inputs = [], outputs=h[-1])
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

    #how to transpose input data
    def transpose_stream(self, data):
        return (np.transpose(data[0], axes=(1,0,2)), np.transpose(data[1], axes=(1,0)), data[2], data[3])

    def transpose_streamTest(self, data):
        return (np.transpose(data[0], axes=(1,0,2)), np.transpose(data[1], axes=(1,0)), data[2], data[3])

    #how to transpose input data
    def transpose_streamPro(self, data):
        return (np.transpose(data[0], axes=(1,0,2)), np.transpose(data[1], axes=(1,0)), data[2], data[3], data[4])



