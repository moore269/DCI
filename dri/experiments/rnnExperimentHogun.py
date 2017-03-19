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
import multiprocessing, logging
mpl = multiprocessing.log_to_stderr()
mpl.setLevel(logging.INFO)
import random

# Load config parameters
locals().update(config)

def computeAccuracies(G, curPreds, save_path, name):
    accCount=0; accTotal = 0;
    for key in curPreds:
        accTotal+=1
        pred = 0
        if curPreds[key]>0.5:
            pred = 1
        if G.node[key]['label'][0]==pred:
            accCount+=1
    accuracyActual = float(accCount)/accTotal
    np.save(save_path+"_ACC_"+name, np.array(accuracyActual))
    print("_ACC_"+name+": "+str(accuracyActual))


#train Rnn collectively
def trainRnnCollective(G=None, maxNProp=None, attr1=None, attr2=None, avgNeighbLabel=None, avgPosNegAttr=None, num01s=None, trainNodes=None, 
    validationNodes=None, testNodes=None, memory=None, mini_dim=None, batch_size=None, num_epochs=None, maxNeighbors=None, maxNeighbors2=None, debug=None, max_epochs=None, 
    actual_save_path=None, multipleCollectiveTraining=None, dynamLabel=None, epsilon=None, netType=None, useActualLabs=None, 
    num_layers=None, onlyLabs=None, pageRankOrder=None, PPRs=None, bias_self=None, testLimit=None, usePrevWeights=None, 
    dataAug="none", perturb=False, usePro=False, lastH=False, lastHs=None, lastHStartRan=False, **kwargs):
    from code.RelationalModels.RelationalLSTM import RelationalLSTM
    from code.RelationalModels.RelationalLSTM_2 import RelationalLSTM_2
    from code.RelationalModels.RelationalRNNwMini import RelationalRNNwMini
    from code.RelationalModels.RelationalLSTMwMini import RelationalLSTMwMini
    if maxNProp==0:
        return (None, None, None)

    if lastHStartRan:
        lastHs=None
    #returns mean squared error given two dictionaries
    def MSE(curr, prev):
        mse = 0.0
        for key in curr:
            mse+=(curr[key] - prev[key])**2
        mse=mse/len(curr)
        return mse

    maxRnnAcc=10.0; maxRnnAccValid=10.0; maxRnnAccTrain=10.0;
    accuracies=[]; accuraciesValid=[]
    numEpochs=[]
    bestRnn=None
    bestPredsTrain = {}; bestPredsValid = {}; bestPredsTest = {};
    rnn2=None
    maxK=0
    curPredsPrev=None
    validCounter=0
    loss=[]
    startTime = time.time()
    for k in range(0, maxNProp):
        #stop early if we have no improvement
        validCounter+=1
        if validCounter>=num_epochs:
            break

        #average neighbors, put into attr2
        #graph_helper.averageNeighbors(G, attr2, attr1, avgNeighbLabel, avgPosNegAttr, num01s)
        #don't average neighbors, use their actual predictions
        G = graph_helper.usePreviousPredictions(G, attr2, attr1, dynamLabel=dynamLabel, avgNeighbLabel=avgNeighbLabel, pageRankOrder=pageRankOrder, PPRs=PPRs, maxNeighbors=maxNeighbors, bias_self=bias_self, testLimit=testLimit, lastHs=lastHs, lastH=lastH, dim=memory)

        #only train on first iteration if multipleCollectiveTraining is false
        # or if we want to use previous weights
        #train once
        if k==0 or (multipleCollectiveTraining and not usePrevWeights):
            save_path2=actual_save_path+"_RNNC_"+str(k)
            if netType=="LSTMwMini":
                rnn2 = RelationalLSTMwMini(G, trainNodes, validationNodes, HogunVarLen=True, perturb=perturb, dim=memory, mini_dim=mini_dim, summary_dim=memory+mini_dim, batch_size=batch_size, num_epochs=num_epochs, save_path=save_path2, 
                    max_epochs=max_epochs, maxNeighbors=maxNeighbors, maxNeighbors2=maxNeighbors2, attrKey=attr2, debug=debug, usePrevWeights=usePrevWeights, epsilon=epsilon, 
                    useActualLabs=useActualLabs, onlyLabs=onlyLabs, dataAug=dataAug,pageRankOrder=pageRankOrder, batchesInferences=batchesInferences)  
             
            rnn2.train()

        #use previous weights here
        elif usePrevWeights:
            rnn2.resetData()
            rnn2.main_loop(False)

        if lastH:
            lastHs = rnn2.generateHidden("train")
            lastHs.update(rnn2.generateHidden("valid"))

        print(rnn2.mainloop.status['epochs_done'])
        numEpochs.append(rnn2.mainloop.status['epochs_done'])
        endTime = time.time()
        print(str(k)+" time: "+str(endTime-startTime))
        startTime=time.time()

        #make preds to update pred atrribute in G
        accuracyTrain, curPredsTrain = rnn2.makePredictions(trainNodes, lastH=False)
        accuracyValid, curPredsValid = rnn2.makePredictions(validationNodes, lastH=False) 
        if lastH:
            accuracyTest, curPredsTest, hiddenRep = rnn2.makePredictions(testNodes, lastH=True)
            lastHs.update(hiddenRep)
            #update nodes lastH
            for node in G.nodes():
                G.node[node]['lastH'] = lastHs[node] 

        else:
            accuracyTest, curPredsTest = rnn2.makePredictions(testNodes)

        #see if predictions converge
        if curPredsPrev is not None:
            loss.append(MSE(curPredsTest, curPredsPrev))
        curPredsPrev=curPredsTest

        #rnn2.makePredictions(validationNodes)
        accuracies.append(accuracyTest)
        accuraciesValid.append(accuracyValid)
        if(accuracyValid<maxRnnAccValid):
            
            validCounter=0
            maxRnnAccTrain = accuracyTrain
            maxRnnAccValid=accuracyValid
            maxRnnAcc=accuracyTest

            maxK=k
            bestRnn=rnn2
            bestPredsTrain = curPredsTrain
            bestPredsValid = curPredsValid
            bestPredsTest = curPredsTest

    #save which model is best
    np.save(actual_save_path+"_accValid", np.array(accuraciesValid))
    np.save(actual_save_path+"_accTest", np.array(accuracies))

    #save baes
    np.save(actual_save_path+"_BAE_TestC", np.array(maxRnnAcc))
    np.save(actual_save_path+"_BAE_TraC", np.array(maxRnnAccTrain))
    np.save(actual_save_path+"_BAE_ValC", np.array(maxRnnAccValid))

    #save accuracies
    computeAccuracies(G, bestPredsTrain, actual_save_path, "TraC")
    computeAccuracies(G, bestPredsValid, actual_save_path, "ValC")
    computeAccuracies(G, bestPredsTest, actual_save_path, "TestC")

    #save preds
    np.save(actual_save_path+"_pre_TraC", np.array(bestPredsTrain.items()))
    np.save(actual_save_path+"_pre_ValC", np.array(bestPredsValid.items()))
    np.save(actual_save_path+"_pre_TestC", np.array(bestPredsTest.items()))

    np.save(actual_save_path+"_MSEs", np.array(loss))
    np.save(actual_save_path+"_bestIter", np.array(maxK))
    np.save(actual_save_path+"_numEpochs", np.array(numEpochs))

    best = {'Train_acc': maxRnnAccTrain, 'Valid_acc': maxRnnAccValid, 'Test_acc': maxRnnAcc}
    return (maxRnnAcc, bestRnn, best)

#helper function to return T or F for True or False. This helps to cut down on fileName size
def bStr(boolStr):
    if boolStr:
        return "T"
    else:
        return "F"



def experiment1(variables):

    #this is so that we can pass in all vars and update all locals to these vars
    exec ""
    locals().update(variables)
    if gpu!="cpu":
        import theano.sandbox.cuda
        theano.sandbox.cuda.use(gpu)

    from code.RelationalModels.RelationalLSTM import RelationalLSTM
    from code.RelationalModels.RelationalLSTM_2 import RelationalLSTM_2
    from code.RelationalModels.RelationalRNNwMini import RelationalRNNwMini
    from code.RelationalModels.RelationalLSTMwMini import RelationalLSTMwMini

    from blocks.filter import VariableFilter
    from blocks.roles import PARAMETER

    startTime=time.time()
    G = readData.readHogun(dataFolder, fName)
    GFirst=G

    retDict = {}

    lastHs = None
    PPRs=None
    #read trial from file
    rest, validationNodes= readData.readTrial(dataFolder, fName, i, percentValidation)

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

    #if we don't want to partition into traditional validation set
    #we simply set train set to validation set
    if noValid:
        trainAll=trainNodes+validationNodes
        shuffle(trainAll)
        totalTrain = int(len(trainAll)*0.4)
        trainNodes = trainAll[:totalTrain]
        validationNodes = trainAll[totalTrain:]

    # if we are doing PPR, then change G to the one associated with individual folds/trial
    # otherwise G is always the same
    if pageRankOrder=="for" or pageRankOrder=="back":
        PPRs = pickle.load(open(dataFolder+fName+"_10pr_"+PRType+"_trial_"+str(i%10)+"_fold_"+str(j)+".p", 'rb'))
        G = readData.readHogun(dataFolder, fName)
        if degree:
            graph_helper.transferAttr(GFirst, G, 'degree')

    actual_save_path = save_path+"_trial_"+str(i)+"_fold_"+str(j)
    if not randInit:
        if netType=="LSTMwMini":
            rnn = RelationalLSTMwMini(G, trainNodes, validationNodes, HogunVarLen=True, perturb=perturb, dim=memory, mini_dim=mini_dim, summary_dim=memory+mini_dim, batch_size=batch_size, num_epochs=num_epochs, save_path=actual_save_path, 
                max_epochs=max_epochs, maxNeighbors=maxNeighbors, maxNeighbors2=maxNeighbors2, attrKey=attr1, debug=debug, usePrevWeights=usePrevWeights,epsilon=epsilon, 
                pageRankOrder=pageRankOrder, batchesInferences=batchesInferences)   

        rnn.train()
        if lastH:
            lastHs = rnn.generateHidden("train")
            lastHs.update(rnn.generateHidden("valid"))

        #DON'T dynamically change test nodes labels
        accuracyTrain, curPredsTrain = rnn.makePredictions(trainNodes, maxNeighbors, changeLabel=False, lastH=False)
        retDict['accuracyTrain'] = accuracyTrain

        #DON'T dynamically change test nodes labels
        accuracyValid, curPredsValid = rnn.makePredictions(validationNodes, maxNeighbors, changeLabel=False, lastH=False)
        retDict['accuracyValid'] = accuracyValid

        #dynamically change test nodes labels
        if lastH:
            accuracyTest, curPredsTest, hiddenRep = rnn.makePredictions(testNodes, maxNeighbors, lastH=True)
            lastHs.update(hiddenRep)
        else:
            accuracyTest, curPredsTest = rnn.makePredictions(testNodes, maxNeighbors, lastH=False)
        retDict['accuracyTest'] = accuracyTest

        #save the actual predictions
        np.save(actual_save_path+"_pre_Tra", np.array(curPredsTrain.items()))
        np.save(actual_save_path+"_pre_Val", np.array(curPredsValid.items()))
        np.save(actual_save_path+"_pre_Test", np.array(curPredsTest.items()))

        np.save(actual_save_path+"_BAE_Test", np.array(accuracyTest))
        np.save(actual_save_path+"_BAE_Tra", np.array(accuracyTrain))
        np.save(actual_save_path+"_BAE_Val", np.array(accuracyValid))
    else:
        graph_helper.setLabels(G, trainNodes+validationNodes, testNodes)

    computeAccuracies(G, curPredsTrain, actual_save_path, "Tra")
    computeAccuracies(G, curPredsValid, actual_save_path, "Val")
    computeAccuracies(G, curPredsTest, actual_save_path, "Test")

    #also dynamically change validation nodes if we desire
    #rnn.makePredictions(validationNodes, maxNeighbors)
    localsCopy = globals().copy()
    localsCopy.update(locals())
    #localsCopy = locals().copy()
    test_bae, rnn2, best = trainRnnCollective(**localsCopy)

    
    #if randInit, replace with actual collective performance
    if randInit:
        retDict['accuracyTrain'] = best['Train_acc']
        retDict['accuracyValid'] = best['Valid_acc']
        retDict['accuracyTest'] = best['Test_acc']

    print("test_bae: "+str(test_bae))
    retDict['accuracyTest_C'] = test_bae
    elapsed=time.time()-startTime

    print("trial: "+str(i)+", fold: "+str(j)+", time: "+str(elapsed))
    return retDict


def test_D(accuracyOutputName, fName, memory, mini_dim,netType, no0Deg=True, avgNeighb=False, degree=False, neighbDegree=False, avgNeighbLabel=False, 
    dynamLabel=True, avgPosNegAttr=False, num01s=False, noValid=False, useActualLabs=False, onlyLabs=False, perturb=False, gpu="cpu", localClustering=False, PRType="neg", 
    singlePR=False, bias_self="", testLimit=False, usePrevWeights=True, n0Deg=False, dataAug="none", randInit=False, pageRankOrder=True, usePro=False, lastH=False, lastHStartRan=False):
    startTime=time.time()


    #maxNProp=5
    maxNProp=0
    prevStr = ""
    if usePrevWeights:
        prevStr="w_"
        maxNProp=100
        #maxNProp=1
    if no0Deg:
        prevStr=prevStr+"no0_"

    if debug:
        """print('uncomment your values')
        sys.exit(0)
        """
        maxNProp=0
        max_epochs=200
        trials=1
        selectFolds=[15]
        numProcesses=1
        print("in debugging mode")

    #create unique experiment name for output file names
    accuracyOutputName=accuracyOutputName+prevStr+netType+"_aug_"+dataAug+"_"+fName+"_mNe_"+str(maxNeighbors)+"_mmNe_"+str(maxNeighbors2)+"_noVl_"+str(bStr(noValid))+"_Mem_"+str(memory)+"_min_"+str(mini_dim) + \
        "_mEp_"+str(max_epochs)+"_mNPro_"+str(maxNProp)+"_trls_"+str(trials)+"_sFlds_"+str(bStr(onlySelectFolds))+ \
        "_PPR_"+str(pageRankOrder) + "_onlyPR_"+str(bStr(singlePR))+"_prT_"+PRType + "_bself_" + bias_self + "_d_"+str(bStr(degree) + \
        "_lim_"+str(bStr(testLimit)))+ "_rinit_"+str(bStr(randInit))+"_p_"+str(bStr(perturb)+"_pro_"+str(bStr(usePro)))+"_lH_"+str(bStr(lastH))+"_lR_"+str(bStr(lastHStartRan))
    print("test Name: "+accuracyOutputName)

    save_path=save_path_prefix+accuracyOutputName

    print(selectFolds)
    #if single PR, set to blank on collective runs
    if singlePR:
        attr1='blank'
    else:
        attr1 = attr1d

    #i=0; m=0; j=1;

    #create input data only varying i, m, and j
    paramsList = []
    results = []
    for i in range(0, trials):
        for m, j in enumerate(selectFolds):
            localsCopy = locals().copy()
            #debug
            if debug:
                results.append(experiment1(localsCopy))
            paramsList.append(localsCopy)

            #debugging
            #experiment1(localsCopy)

    #run in parallel with max number of processes
    if not debug:
        pool = Pool(processes=numProcesses)
        results = pool.map(experiment1, paramsList)
    
    #read data to keep record of the original network
    G = readData.readHogun(dataFolder, fName)
    GFirst=G
    accTrials2={netType + "_Test":np.zeros((trials, numSelectFolds)), netType + "_Train":np.zeros((trials, numSelectFolds)), netType + "_Valid":np.zeros((trials, numSelectFolds)), netType + "_Test_C":np.zeros((trials, numSelectFolds))}

    #now parse all results, assuming they are in order (pool.map preserves this)
    counter = -1
    for i in range(0, trials):
        for m, j in enumerate(selectFolds):
            counter+=1
            retDict = results[counter]
            setNames = ["Train", "Valid", "Test", "Test_C"]
            for name in setNames:
                accTrials2[netType + "_"+name][i][m]=retDict['accuracy'+name]

    #graph and record data
    if onlySelectFolds:
        t=[ round(round(percentValidation, 5)+round((fold+1)*percentBy, 5), 5) for fold in selectFolds]
    else:
        t = arange(round(percentValidation+percentBy, 5), round(percentRest+percentValidation, 5)-0.0001, round(percentBy,5))
    
    #graph_helper.plotBAE(accTrials2, accuracyOutputName+"_Plot", t)
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

    #python rnnExperiments.py [rnn] [mem] [prtype] [singlepr] [fName] [bias_self]
    netType=sys.argv[1]
    memory = int(sys.argv[2])
    mini_dim = int(sys.argv[3])
    degree = True if not int(sys.argv[4])==0 else False
    fName = sys.argv[5]

    # fNames
    # amazon_DVD_20000
    # facebook
    # amazon_Music_10000
    test_D(accuracyOutputName="", fName=fName, netType=netType, memory=memory, mini_dim=mini_dim)

    
