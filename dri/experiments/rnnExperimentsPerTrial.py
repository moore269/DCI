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
from rnnExperiments import *

# Load config parameters
locals().update(config)

def test_E(accuracyOutputName, fName, memory, mini_dim, generateUnordered, avgNeighb, degree, neighbDegree, avgNeighbLabel, 
    dynamLabel, avgPosNegAttr, num01s, noValid, netType, useActualLabs, onlyLabs, perturb=False, gpu="cpu", localClustering=False, PRType="neg", 
    singlePR=False, bias_self="", testLimit=False, usePrevWeights=True, no0Deg=False, dataAug="none", randInit=False, pageRankOrder="F", usePro=False, lastH=False, lastHStartRan=False,
    changeTrainValid=0, i=0, j=0):
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
        "_lim_"+str(bStr(testLimit)))+ "_rinit_"+str(bStr(randInit))+"_p_"+str(bStr(perturb))+"_pro_"+str(bStr(usePro))+"_lH_"+str(bStr(lastH))+"_lR_"+str(bStr(lastHStartRan))+"_CTV_"+str(changeTrainValid)
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
    prtype=sys.argv[4]
    singlepr= True if not int(sys.argv[5])==0 else False
    fName = sys.argv[6]
    bias_self = sys.argv[7]
    degree = True if not int(sys.argv[8])==0 else False
    testLimit=True if not int(sys.argv[9])==0 else False
    usePrevWeights = True if not int(sys.argv[10])==0 else False
    no0Deg = True if not int(sys.argv[11])==0 else False
    dataAug = sys.argv[12]
    randInit = True if not int(sys.argv[13])==0 else False
    pageRankOrder = sys.argv[14]
    perturb = True if not int(sys.argv[15])==0 else False
    usePro = True if not int(sys.argv[16])==0 else False
    lastH = True if not int(sys.argv[17])==0 else False
    lastHStartRan = True if not int(sys.argv[18])==0 else False
    changeTrainValid = int(sys.argv[19])
    gpu = sys.argv[20]
    i = int(sys.argv[21])
    j = int(sys.argv[22])
    test_E(accuracyOutputName="", perturb=perturb, lastH=lastH, lastHStartRan=lastHStartRan, fName=fName, noValid=False, netType=netType, memory=memory, mini_dim=mini_dim, bias_self=bias_self, testLimit=testLimit, gpu=gpu,
        changeTrainValid=changeTrainValid, usePrevWeights = usePrevWeights, generateUnordered = 1, PRType= prtype, singlePR=singlepr, dataAug=dataAug, randInit = randInit, pageRankOrder=pageRankOrder, useActualLabs=False, onlyLabs=False, avgNeighb=False, degree=degree, no0Deg=no0Deg,
        neighbDegree=False, avgNeighbLabel=False, dynamLabel=True, avgPosNegAttr=False, num01s=False, localClustering=False, i=i, j=j, usePro=usePro)