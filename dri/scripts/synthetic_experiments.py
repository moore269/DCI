import sys
import os
#add previous folder to path
cwd = os.getcwd(); rIndex = cwd.rfind("/");cwd = cwd[:rIndex];
sys.path.insert(0, cwd)

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from pylab import *

import networkx as nx
import code.readData.readData as readData
import copy
import code.graph_helper.graph_helper as graph_helper

from operator import add
import numpy as np
import time
from collections import defaultdict

import cPickle as pickle
from config import config

from scipy.stats import pearsonr
from multiprocessing import Pool
import random

# Load config parameters
locals().update(config)

def test_Synthetic(netType, dim, mini_dim, summary_dim, input_dimx, input_dimxmini, gpu):

    actual_save_path="syntheticAddTest_dim_"+str(dim)+"_minidim_"+str(mini_dim)+"_summarydim_"+str(summary_dim)+"_inputdimx_"+str(input_dimx)+"_inputdimxmini_"+str(input_dimxmini)
    if netType=="RNNwMini":
        rnn = RelationalRNNwMini(None, None, None, dim=dim, mini_dim=mini_dim, summary_dim=summary_dim, input_dimx=input_dimx, input_dimxmini=input_dimxmini,
            batch_size=batch_size, num_epochs=num_epochs, save_path=actual_save_path, 
            max_epochs=max_epochs, debug=debug, epsilon=epsilon,batchesInferences=True)   
        rnn.train()
        #DON'T dynamically change test nodes labels
        MSE, curPreds = rnn.makePredictions(None, None)
        print("MSE: "+str(MSE))
        
if __name__ == "__main__":

    #python rnnExperiments.py [rnn] [mem] [prtype] [singlepr] [fName] [bias_self]
    netType=sys.argv[1]
    dim = int(sys.argv[2])
    mini_dim=int(sys.argv[3])
    summary_dim=int(sys.argv[4])
    input_dimx=int(sys.argv[5])
    input_dimxmini=int(sys.argv[6])
    gpu = sys.argv[7]

    # fNames
    # amazon_DVD_20000
    # facebook
    # amazon_Music_10000
    test_Synthetic(netType=netType, dim=dim, mini_dim=mini_dim, summary_dim=summary_dim, input_dimx=input_dimx, input_dimxmini=input_dimxmini, gpu=gpu)
