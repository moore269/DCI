import numpy
import theano
from theano import tensor as T
from blocks import initialization
from blocks.bricks import Linear, Tanh
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.initialization import Constant, IsotropicGaussian

class RNNWithMini(BaseRecurrent):
    def __init__(self, dim, mini_dim, summary_dim, **kwargs):
        super(RNNWithMini, self).__init__(**kwargs)
        self.dim = dim
        self.mini_dim=mini_dim
        self.summary_dim=summary_dim

        self.recurrent_layer = SimpleRecurrent(
            dim=self.summary_dim, activation=Tanh(), name='recurrent_layer',
            weights_init=IsotropicGaussian(), biases_init=Constant(0.0))
        self.mini_recurrent_layer = SimpleRecurrent(
            dim=self.dim, activation=Tanh(), name='mini_recurrent_layer',
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
    		dim = super(RNNWithMini, self).get_dim(name)
    	return dim

#input2
#seq1len, seq2len, batchsize, attr2
#seq2len, batchsize, attr2
#batchsize, attr2


#input1
#seq1len, batchsize, attr1
#batchsize, attr1

#input1cat
#seq1len, attr1+input2, batchsize


#example
# seqlen, batchsize, attr
# batchsize, attr

#minimum working example
#input1 and input2: numpy.ones((3, 1, 4), dtype=theano.config.floatX), numpy.ones((3, 2, 1, 4), dtype=theano.config.floatX))
#input2 for 1 element of seq1 corresponds to processing seq2len, batchsize, attr2
#this results in [[2 2 2 2]]
#concatenating onto first input results in and adding 0s results in the first state, [[1111 2222]]
#Performing the same process but adding the previoius state results in [[2222 4444]]
#again [[3333 6666]]

x = T.tensor3('x')
xmini = T.tensor4('xmini')

feedback = RNNWithMini(dim=2, mini_dim=2, summary_dim=4)
feedback.initialize()
h = feedback.apply(x=x, xmini=xmini)
f = theano.function([x, xmini], [h])
for states in f(numpy.ones((3, 1, 2), dtype=theano.config.floatX), numpy.ones((3, 2, 1, 2), dtype=theano.config.floatX)):
    print(states.shape)
    print(states)