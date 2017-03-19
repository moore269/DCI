import sys
import os
#add previous folder to path
cwd = os.getcwd(); rIndex = cwd.rfind("/");cwd = cwd[:rIndex];
sys.path.insert(0, cwd)

import numpy as np
import theano
from theano import tensor as T
from code.RelationalModels.RNNwMini import RNNwMini
import math
from random import shuffle

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
#the linear transformation simply copies previous input when summary_dim=8
#when summary_dim<8, it copies first summary_dim elements


#create sets of numbers between [0, 2^j] where j in (0, numSets)
#create batchsize examples per set
#randomize this
#and redistribute into batches of batchsize

def main():
    batchsize=100
    seqlenx = 10
    seqlenxmini=2
    attrxsize=2
    attrxminisize=2
    summary_dim=4
    numSets = 2
    #values denote number of batches
    data_sets={'train':6,'valid':2,'test':2}

    x = T.tensor3('x')
    xmini = T.tensor4('xmini')

    feedback = RNNwMini(dim=attrxsize, mini_dim=attrxminisize, summary_dim=summary_dim)
    feedback.initialize()
    h = feedback.apply(x=x, xmini=xmini)
    endSum = T.sum(h[-1], axis=1)

    f = theano.function([x, xmini], [endSum, h])

    for key in data_sets:
        alldata=[]
        for i in range(0, data_sets[key]):
            for j in range(0, numSets):
                for k in range(0, batchsize):
                    mult = math.pow(2, j)
                    in_x = np.random.random_sample((seqlenx, 1, attrxsize))*mult; in_x = in_x.astype(theano.config.floatX);
                    in_xmini = np.random.random_sample((seqlenx, seqlenxmini, 1, attrxminisize))*mult; in_xmini=in_xmini.astype(theano.config.floatX);

                    outputs = f(in_x, in_xmini)
                    alldata.append((in_x, in_xmini, outputs[0]))

        shuffle(alldata)
        #combine then shuffle
        comb={'x':None, 'xmini':None, 'y':None}
        axes={'x':1, 'xmini':2, 'y':0}
        indices={'x':0, 'xmini':1, 'y':2}
        data={'x':[], 'xmini':[], 'y':[]}
        for i, batches in enumerate(alldata):
            if i!=0 and i%batchsize==0:
                for name in data:
                    data[name].append(comb[name])
                    comb[name]=None
            print(batches[indices['y']])
            for name in indices:
                if comb[name] is None:
                    comb[name] = batches[indices[name]]
                else:
                    comb[name]=np.concatenate((comb[name], batches[indices[name]]), axis=axes[name])

        np.save("../experiments/data/RNNwMiniSynthetic_"+key+"_x", np.array(data['x']))
        np.save("../experiments/data/RNNwMiniSynthetic_"+key+"_xmini", np.array(data['xmini']))
        np.save("../experiments/data/RNNwMiniSynthetic_"+key+"_y", np.array(data['y']))

    print(outputs[0].shape)
    print(outputs[0])
    print(outputs[1].shape)
    print(outputs[1]) 

if __name__ == "__main__":
    main()
