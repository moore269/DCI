from abc import ABCMeta, abstractmethod

import theano
from theano import tensor
from six import add_metaclass
from blocks.bricks.cost import Cost, CostMatrix

from blocks.bricks.base import application, Brick

class BinaryCrossEntropyProp(CostMatrix):
    @application
    def cost_matrix(self, y, y_hat, proportion):
        cost = proportion*tensor.nnet.binary_crossentropy(y_hat, y)
        return cost