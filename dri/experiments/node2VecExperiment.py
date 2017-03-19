import sys
import os
#add previous folder to path
cwd = os.getcwd(); rIndex = cwd.rfind("/");cwd = cwd[:rIndex];
sys.path.insert(0, cwd)


import networkx as nx
import numpy as np
import time
from collections import defaultdict

from config import config
import code.graph_helper.graph_helper as graph_helper
import code.readData.readData as readData
from multiprocessing import Pool

import multiprocessing, logging

logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.INFO)
logger.warning('doomed')

import random

# Load config parameters
locals().update(config)

def experiment1(variables):
    exec ""
    locals().update(variables)
    from code.RelationalModels.node2vecLR import node2vecLR
    startTime=time.time()
    G = readData.readDataset(dataFolder, fName, no0Deg=True) 
    GFirst=G
    retDict = {}

    rest, validationNodes= readData.readTrial(fName, i, percentValidation)

    #prune out nodes that don't exist in GFirst
    rest = graph_helper.prune0s(GFirst, rest)
    validationNodes = graph_helper.prune0s(GFirst, validationNodes)

    #split into folds
    folds= readData.splitNodeFolds(rest, numFolds)

    #vary training set
    foldsStart=time.time()
    #m is index into array, j is fold index. they are the same if we are looping over all folds

    #add up trainNodes
    trainNodes=[] 
    for k in range(0, j+1):
        trainNodes+=folds[k]

    #add up rest of nodes
    testNodes=[]
    for k in range(j+1, numFolds):
        testNodes+=folds[k]

    actual_save_path = save_path+"_trial_"+str(i)+"_fold_"+str(j)

    lr = node2vecLR(dataFolder+fName, trainNodes, validationNodes, testNodes)
    lr.train()
    accuracyTrain = lr.predictBAE(testSet="train")
    accuracyValid = lr.predictBAE(testSet="valid")
    accuracyTest = lr.predictBAE(testSet="test")
    retDict['accuracyTrain'] = accuracyTrain
    retDict['accuracyValid'] = accuracyValid
    retDict['accuracyTest'] = accuracyTest

    np.save(save_path+"_BAE_Test", np.array(accuracyTest))
    np.save(save_path+"_BAE_Tra", np.array(accuracyTrain))
    np.save(save_path+"_BAE_Val", np.array(accuracyValid))

    elapsed=time.time()-startTime
    print("trial: "+str(i)+", fold: "+str(j)+", time: "+str(elapsed))
    return retDict

def node2vecExperiment(fName):
    startTime=time.time()
    accuracyOutputName = "node2vec_default_params_"+fName
    print("test Name: "+accuracyOutputName)

    save_path=save_path_prefix+accuracyOutputName
    netType="node2vec"
    paramsList = []
    results = []
    for i in range(0, trials):
        for m, j in enumerate(selectFolds):
            localsCopy = locals().copy()
            #debug
            if debug:
                results.append(experiment1(localsCopy))
            paramsList.append(localsCopy)
    if not debug:
        pool = Pool(processes=numProcesses)
        results = pool.map(experiment1, paramsList)

    accTrials2={netType + "_Test":np.zeros((trials, numSelectFolds)), netType + "_Train":np.zeros((trials, numSelectFolds)), netType + "_Valid":np.zeros((trials, numSelectFolds))}

    #now parse all results, assuming they are in order (pool.map preserves this)
    counter = -1
    for i in range(0, trials):
        for m, j in enumerate(selectFolds):
            counter+=1
            retDict = results[counter]
            setNames = ["Train", "Valid", "Test"]
            for name in setNames:
                accTrials2[netType + "_"+name][i][m]=retDict['accuracy'+name]

    #graph and record data
    if onlySelectFolds:
        t=[ round(round(percentValidation, 5)+round((fold+1)*percentBy, 5), 5) for fold in selectFolds]
    else:
        t = arange(round(percentValidation+percentBy, 5), round(percentRest+percentValidation, 5)-0.0001, round(percentBy,5))
    
    graph_helper.plotBAE(accTrials2, accuracyOutputName+"_Plot", t)
    graph_helper.printNPBAE(accTrials2, accuracyOutputName)
    

    #print aggregated results
    strBuilder = ""
    for key in accTrials2:
        baes = []
        mean = np.mean(accTrials2[key], axis=0)
        strBuilder = strBuilder+str(key)+": "+str(mean[0])+", "
    print(strBuilder)
    #print(accTrials2)
    elapsed=time.time()-startTime
    print("total time: "+str(elapsed))


if __name__ == "__main__":
    fName=sys.argv[1]
    node2vecExperiment(fName)