import os
import sys
#add previous folder to path
cwd = os.getcwd(); rIndex = cwd.rfind("/");cwd = cwd[:rIndex];
sys.path.insert(0, cwd)

import networkx as nx
import code.readData.readData as readData
import copy
import code.graph_helper.graph_helper as graph_helper

from operator import add
import numpy as np
import time
import re
from collections import defaultdict

import cPickle as pickle
from config import config

from scipy.stats import pearsonr
from multiprocessing import Pool
import random
from rnnExperimentHogun import *

# Load config parameters
locals().update(config)

def test_E(accuracyOutputName, fName, memory, mini_dim,netType, no0Deg=True, avgNeighb=False, degree=False, neighbDegree=False, avgNeighbLabel=False, 
    dynamLabel=True, avgPosNegAttr=False, num01s=False, noValid=False, useActualLabs=False, onlyLabs=False, perturb=False, gpu="cpu", localClustering=False, PRType="neg", 
    singlePR=False, bias_self="", testLimit=False, usePrevWeights=True, n0Deg=False, dataAug="none", randInit=False, pageRankOrder=True, usePro=False, lastH=False, lastHStartRan=False,
    i=0, j=0):
    print("i="+str(i))
    print("j="+str(j))
    startTime=time.time()
    prevStr = ""
    if usePrevWeights:
        prevStr="w_"
    if no0Deg:
        prevStr=prevStr+"no0_"

    if debug:
        print('uncomment your values')
        sys.exit(0)
        """
        maxNProp=1
        max_epochs=1
        trials=1
        selectFolds=[3]
        numProcesses=1
        print("in debugging mode")"""
        

    accuracyOutputName=accuracyOutputName+prevStr+netType+"_aug_"+dataAug+"_"+fName+"_mNe_"+str(maxNeighbors)+"_mmNe_"+str(maxNeighbors2)+"_noVl_"+str(bStr(noValid))+"_Mem_"+str(memory)+"_min_"+str(mini_dim) + \
        "_mEp_"+str(max_epochs)+"_mNPro_"+str(maxNProp)+"_trls_"+str(trials)+"_sFlds_"+str(bStr(onlySelectFolds))+ \
        "_PPR_"+str(pageRankOrder) + "_onlyPR_"+str(bStr(singlePR))+"_prT_"+PRType + "_bself_" + bias_self + "_d_"+str(bStr(degree) + \
        "_lim_"+str(bStr(testLimit)))+ "_rinit_"+str(bStr(randInit))+"_p_"+str(bStr(perturb)+"_pro_"+str(bStr(usePro)))+"_lH_"+str(bStr(lastH))+"_lR_"+str(bStr(lastHStartRan))
    print("test Name: "+accuracyOutputName)

    save_path=save_path_prefix+accuracyOutputName

    #if single PR, set to blank on collective runs
    if singlePR:
        attr1='blank'
    else:
        attr1 = attr1d

    experiment1(locals().copy())
    elapsed=time.time()-startTime
    print("total time: "+str(elapsed))

if __name__ == "__main__":

    #python rnnExperiments.py [rnn] [mem] [prtype] [singlepr] [fName] [bias_self]
    netType=sys.argv[1]
    memory = int(sys.argv[2])
    mini_dim = int(sys.argv[3])
    degree = True if not int(sys.argv[4])==0 else False
    fName = sys.argv[5]
    i = int(sys.argv[6])
    j = int(sys.argv[7])
    test_E(accuracyOutputName="", fName=fName, netType=netType, memory=memory, mini_dim=mini_dim, i=i, j=j)