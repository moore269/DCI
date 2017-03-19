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
    if accCount==0 and accTotal==0:
        accuracyActual =0
    else:
        accuracyActual = float(accCount)/accTotal
    np.save(save_path+"_ACC_"+name, np.array(accuracyActual))
    print("_ACC_"+name+": "+str(accuracyActual))


#train Rnn collectively
def trainRnnCollective(G=None, maxNProp=None, attr1=None, attr2=None, avgNeighbLabel=None, avgPosNegAttr=None, num01s=None, trainNodes=None, 
    validationNodes=None, validationNodes2=None, testNodes=None, memory=None, mini_dim=None, batch_size=None, num_epochs=None, maxNeighbors=None, maxNeighbors2=None, debug=None, max_epochs=None, 
    actual_save_path=None, multipleCollectiveTraining=None, dynamLabel=None, epsilon=None, netType=None, useActualLabs=None, 
    num_layers=None, onlyLabs=None, pageRankOrder=None, PPRs=None, bias_self=None, testLimit=None, usePrevWeights=None, 
    dataAug="none", perturb=False, usePro=False, lastH=False, lastHs=None, lastHStartRan=False, changeTrainValid=0, **kwargs):
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

    maxRnnAcc=10.0; maxRnnAccValid=10.0; maxRnnAccValid2=10.0; maxRnnAccTrain=10.0;
    accuracies=[]; accuraciesValid=[] ; accuraciesValid2=[] ; accuraciesTrain=[] ;
    numEpochs=[]
    bestRnn=None
    bestPredsTrain = {}; bestPredsValid = {}; bestPredsValid2 = {}; bestPredsTest = {};
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
        G = graph_helper.usePreviousPredictions(G, attr2, attr1, dynamLabel=dynamLabel, avgNeighbLabel=avgNeighbLabel, pageRankOrder=pageRankOrder, PPRs=PPRs, maxNeighbors=maxNeighbors, bias_self=bias_self, testLimit=testLimit, lastHs=lastHs, lastH=lastH, dim=memory, changeTrainValid=changeTrainValid)

        #only train on first iteration if multipleCollectiveTraining is false
        # or if we want to use previous weights
        #train once
        if k==0 or (multipleCollectiveTraining and not usePrevWeights):
            save_path2=actual_save_path+"_RNNC_"+str(k)
            if netType=="LSTM":
                rnn2 = RelationalLSTM(G, trainNodes, validationNodes, dim=memory, batch_size=batch_size, num_epochs=num_epochs, save_path=save_path2, 
                    max_epochs=max_epochs, maxNeighbors=maxNeighbors, attrKey=attr2, debug=debug, usePrevWeights=usePrevWeights, epsilon=epsilon, 
                    useActualLabs=useActualLabs, onlyLabs=onlyLabs, dataAug=dataAug,pageRankOrder=pageRankOrder, batchesInferences=batchesInferences, usePro=usePro, lastH=lastH)
            elif netType=="LSTM2":
                rnn2 = RelationalLSTM_2(G, trainNodes, validationNodes, dim=memory, summary_dim=memory, batch_size=batch_size, num_epochs=num_epochs, save_path=save_path2, 
                    max_epochs=max_epochs, maxNeighbors=maxNeighbors, attrKey=attr2, debug=debug, usePrevWeights=usePrevWeights, epsilon=epsilon, 
                    useActualLabs=useActualLabs, onlyLabs=onlyLabs, dataAug=dataAug,pageRankOrder=pageRankOrder, batchesInferences=batchesInferences)
            elif netType=="RNNwMini":
                rnn2 = RelationalRNNwMini(G, trainNodes, validationNodes, perturb=perturb, dim=dim, mini_dim=mini_dim, summary_dim=memory+mini_dim, batch_size=batch_size, num_epochs=num_epochs, save_path=save_path2, 
                    max_epochs=max_epochs, maxNeighbors=maxNeighbors, attrKey=attr2, debug=debug, usePrevWeights=usePrevWeights, epsilon=epsilon, 
                    useActualLabs=useActualLabs, onlyLabs=onlyLabs, dataAug=dataAug,pageRankOrder=pageRankOrder, batchesInferences=batchesInferences)  
            elif netType=="LSTMwMini":
                rnn2 = RelationalLSTMwMini(G, trainNodes, validationNodes, perturb=perturb, dim=memory, mini_dim=mini_dim, summary_dim=memory+mini_dim, batch_size=batch_size, num_epochs=num_epochs, save_path=save_path2, 
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
        #print("train: "+bStr(False if changeTrainValid>0 else True))
        #print("valid: "+bStr(False if changeTrainValid>1 else True))
        #make preds to update pred atrribute in G
        accuracyTrain, curPredsTrain = rnn2.makePredictions(trainNodes, changeLabel= False if changeTrainValid>0 else True, lastH=False)
        accuracyValid, curPredsValid = rnn2.makePredictions(validationNodes, changeLabel= False if changeTrainValid>1 else True, lastH=False) 
        accuracyValid2, curPredsValid2 = rnn2.makePredictions(validationNodes2, lastH=False) 

        if lastH:
            if "swap" in dataAug:
                #iterate through all nodes to get hidden states
                tempT, tempPred, hiddenRepT = rnn2.makePredictions(trainNodes, changeLabel= False, lastH=True)
                tempV, tempPred, hiddenRepV = rnn2.makePredictions(validationNodes, changeLabel= False, lastH=True)
                lastHs.update(hiddenRepT)
                lastHs.update(hiddenRepV)

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
        accuraciesValid2.append(accuracyValid2)
        accuraciesTrain.append(accuracyTrain)
        if(accuracyValid<maxRnnAccValid):
            
            validCounter=0
            maxRnnAccTrain = accuracyTrain
            maxRnnAccValid=accuracyValid
            maxRnnAccValid2=accuracyValid2
            maxRnnAcc=accuracyTest

            maxK=k
            bestRnn=rnn2
            bestPredsTrain = curPredsTrain
            bestPredsValid = curPredsValid
            bestPredsValid2 = curPredsValid2
            bestPredsTest = curPredsTest

    #save which model is best
    np.save(actual_save_path+"_accTrain", np.array(accuraciesTrain))
    np.save(actual_save_path+"_accValid", np.array(accuraciesValid))
    np.save(actual_save_path+"_accValid2", np.array(accuraciesValid2))
    np.save(actual_save_path+"_accTest", np.array(accuracies))

    #save baes
    np.save(actual_save_path+"_BAE_TestC", np.array(maxRnnAcc))
    np.save(actual_save_path+"_BAE_TraC", np.array(maxRnnAccTrain))
    np.save(actual_save_path+"_BAE_ValC", np.array(maxRnnAccValid))
    np.save(actual_save_path+"_BAE_Val2C", np.array(maxRnnAccValid2))

    #save accuracies
    computeAccuracies(G, bestPredsTrain, actual_save_path, "TraC")
    computeAccuracies(G, bestPredsValid, actual_save_path, "ValC")
    computeAccuracies(G, bestPredsValid2, actual_save_path, "Val2C")
    computeAccuracies(G, bestPredsTest, actual_save_path, "TestC")

    #save preds
    np.save(actual_save_path+"_pre_TraC", np.array(bestPredsTrain.items()))
    np.save(actual_save_path+"_pre_ValC", np.array(bestPredsValid.items()))
    np.save(actual_save_path+"_pre_Val2C", np.array(bestPredsValid2.items()))
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
    from code.RelationalModels.RelationalLRAVG import RelationalLRAVG
    from blocks.filter import VariableFilter
    from blocks.roles import PARAMETER

    startTime=time.time()
    G = readData.readDataset(dataFolder, fName, sampleAttrs=sampleAttrs, averageNeighborAttr=avgNeighb, degree=degree, neighbDegree=neighbDegree, localClustering=localClustering, no0Deg=no0Deg) 
    GFirst=G

    retDict = {}

    lastHs = None
    PPRs=None
    #read trial from file

    nodeData = readData.readTrial(dataFolder, fName, i, percentValidation, changeTrainValid)
    validationNodes2 = []
    if changeTrainValid>3:
        rest = nodeData[0] ; validationNodes = nodeData[1] ; validationNodes2 = nodeData[2] ;
    else:
        rest = nodeData[0] ; validationNodes = nodeData[1] ;

    #prune out nodes that don't exist in GFirst
    rest = graph_helper.prune0s(GFirst, rest)
    validationNodes = graph_helper.prune0s(GFirst, validationNodes)
    validationNodes2 = graph_helper.prune0s(GFirst, validationNodes2)

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
        PPRs = pickle.load(open(dataFolder+fName.replace("amazon_Music_64500", "amazon_Music_7500")+"_10pr_"+PRType+"_trial_"+str(i%10)+"_fold_"+str(j)+".p", 'rb'))
        G = readData.readDataset(dataFolder, fName, sampleAttrs=sampleAttrs, averageNeighborAttr=avgNeighb, degree=False, neighbDegree=neighbDegree, localClustering=localClustering, pageRankOrder=pageRankOrder, PPRs=PPRs, maxNeighbors=maxNeighbors, bias_self=bias_self, trainNodes=trainNodes+validationNodes, testNodes=testNodes, testLimit=testLimit,
            no0Deg=no0Deg)
        if degree:
            graph_helper.transferAttr(GFirst, G, 'degree')

    actual_save_path = save_path+"_trial_"+str(i)+"_fold_"+str(j)
    if not randInit:
        if netType=="LSTM":
            rnn = RelationalLSTM(G, trainNodes, validationNodes, dim=memory, batch_size=batch_size, num_epochs=num_epochs, save_path=actual_save_path, 
                max_epochs=max_epochs, maxNeighbors=maxNeighbors, attrKey=attr1, debug=debug, usePrevWeights=usePrevWeights,epsilon=epsilon, 
                pageRankOrder=pageRankOrder, batchesInferences=batchesInferences, usePro=usePro)
        elif netType=="LSTM2":
            rnn = RelationalLSTM_2(G, trainNodes, validationNodes, dim=memory, summary_dim=memory, batch_size=batch_size, num_epochs=num_epochs, save_path=actual_save_path, 
                max_epochs=max_epochs, maxNeighbors=maxNeighbors, attrKey=attr1, debug=debug, usePrevWeights=usePrevWeights,epsilon=epsilon, 
                pageRankOrder=pageRankOrder, batchesInferences=batchesInferences, usePro=usePro)
        elif netType=="RNNwMini":
            rnn = RelationalRNNwMini(G, trainNodes, validationNodes, perturb=perturb, dim=memory, mini_dim=mini_dim, summary_dim=memory+mini_dim, batch_size=batch_size, num_epochs=num_epochs, save_path=actual_save_path, 
                max_epochs=max_epochs, maxNeighbors=maxNeighbors, attrKey=attr1, debug=debug, usePrevWeights=usePrevWeights,epsilon=epsilon, 
                pageRankOrder=pageRankOrder, batchesInferences=batchesInferences)   
        elif netType=="LSTMwMini":
            rnn = RelationalLSTMwMini(G, trainNodes, validationNodes, perturb=perturb, dim=memory, mini_dim=mini_dim, summary_dim=memory+mini_dim, batch_size=batch_size, num_epochs=num_epochs, save_path=actual_save_path, 
                max_epochs=max_epochs, maxNeighbors=maxNeighbors, maxNeighbors2=maxNeighbors2, attrKey=attr1, debug=debug, usePrevWeights=usePrevWeights,epsilon=epsilon, 
                pageRankOrder=pageRankOrder, batchesInferences=batchesInferences)   
        elif "LRAVG" in netType:
            rnn = RelationalLRAVG(G, netType=netType.replace("LRAVG", ""), trainNodes=trainNodes, validationNodes=validationNodes, testNodes=testNodes)
        rnn.train()
        if lastH:
            lastHs = rnn.generateHidden("train")
            lastHs.update(rnn.generateHidden("valid"))

        #DON'T dynamically change test nodes labels
        accuracyTrain, curPredsTrain = rnn.makePredictions(trainNodes, maxNeighbors, changeLabel=False if changeTrainValid > -1 else True, lastH=False)
        retDict['accuracyTrain'] = accuracyTrain

        #DON'T dynamically change test nodes labels
        accuracyValid, curPredsValid = rnn.makePredictions(validationNodes, maxNeighbors, changeLabel=False if changeTrainValid > -1 else True, lastH=False)
        retDict['accuracyValid'] = accuracyValid
        

        accuracyValid2, curPredsValid2 = rnn.makePredictions(validationNodes2, maxNeighbors, lastH=False)
        retDict['accuracyValid2'] = accuracyValid2


        #dynamically change test nodes labels
        if lastH:
            if "swap" in dataAug:
                #iterate through all nodes to get hidden states
                tempT, tempPred, hiddenRepT = rnn.makePredictions(trainNodes, maxNeighbors, changeLabel= False, lastH=True)
                tempV, tempPred, hiddenRepV = rnn.makePredictions(validationNodes, maxNeighbors, changeLabel= False, lastH=True)
                lastHs.update(hiddenRepT)
                lastHs.update(hiddenRepV)

            accuracyTest, curPredsTest, hiddenRep = rnn.makePredictions(testNodes, maxNeighbors, lastH=True)
            lastHs.update(hiddenRep)
        else:
            accuracyTest, curPredsTest = rnn.makePredictions(testNodes, maxNeighbors, lastH=False)
        retDict['accuracyTest'] = accuracyTest

        #save the actual predictions
        np.save(actual_save_path+"_pre_Tra", np.array(curPredsTrain.items()))
        np.save(actual_save_path+"_pre_Val", np.array(curPredsValid.items()))
        np.save(actual_save_path+"_pre_Val2", np.array(curPredsValid2.items()))
        np.save(actual_save_path+"_pre_Test", np.array(curPredsTest.items()))

        np.save(actual_save_path+"_BAE_Test", np.array(accuracyTest))
        np.save(actual_save_path+"_BAE_Tra", np.array(accuracyTrain))
        np.save(actual_save_path+"_BAE_Val", np.array(accuracyValid))
        np.save(actual_save_path+"_BAE_Val2", np.array(accuracyValid2))
        print("BAE_Tra: "+str(accuracyTrain))
        print("BAE_Val: "+str(accuracyValid))
        print("BAE_Val2: "+str(accuracyValid2))
        print("BAE_Test: "+str(accuracyTest))

        computeAccuracies(G, curPredsTrain, actual_save_path, "Tra")
        computeAccuracies(G, curPredsValid, actual_save_path, "Val")
        computeAccuracies(G, curPredsValid2, actual_save_path, "Val2")
        computeAccuracies(G, curPredsTest, actual_save_path, "Test")
    else:
        graph_helper.setLabels(G, trainNodes, validationNodes+validationNodes2, testNodes, changeTrainValid)




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


def test_D(accuracyOutputName, fName, memory, mini_dim, avgNeighb, degree, neighbDegree, avgNeighbLabel, 
    dynamLabel, avgPosNegAttr, num01s, noValid, netType, useActualLabs, onlyLabs, perturb=False, gpu="cpu", localClustering=False, PRType="neg", 
    singlePR=False, bias_self="", testLimit=False, usePrevWeights=False, n0Deg=False, dataAug="none", randInit=False, pageRankOrder=True, usePro=False, lastH=False, changeTrainValid=0):
    startTime=time.time()


    #maxNProp=5
    maxNProp=0
    prevStr = ""
    if usePrevWeights:
        prevStr="w_"
        #maxNProp=100
        maxNProp=1
    if no0Deg:
        prevStr=prevStr+"no0_"

    if debug:
        print('uncomment your values')
        sys.exit(0)
        """
        maxNProp=0
        max_epochs=200
        trials=1
        selectFolds=[15]
        numProcesses=1
        print("in debugging mode")"""

    #create unique experiment name for output file names
    accuracyOutputName=accuracyOutputName+prevStr+netType+"_aug_"+dataAug+"_"+fName+"_mNe_"+str(maxNeighbors)+"_mmNe_"+str(maxNeighbors2)+"_noVl_"+str(bStr(noValid))+"_Mem_"+str(memory)+"_min_"+str(mini_dim) + \
        "_mEp_"+str(max_epochs)+"_mNPro_"+str(maxNProp)+"_trls_"+str(trials)+"_sFlds_"+str(bStr(onlySelectFolds))+ \
        "_PPR_"+str(pageRankOrder) + "_onlyPR_"+str(bStr(singlePR))+"_prT_"+PRType + "_bself_" + bias_self + "_d_"+str(bStr(degree) + \
        "_lim_"+str(bStr(testLimit)))+ "_rinit_"+str(bStr(randInit))+"_p_"+str(bStr(perturb)+"_pro_"+str(bStr(usePro)))+"_lH_"+str(bStr(lastH))+"_lR_"+str(bStr(lastHStartRan))+"_CTV_"+str(changeTrainValid)
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
    G = readData.readDataset(dataFolder, fName, sampleAttrs=sampleAttrs, averageNeighborAttr=avgNeighb, degree=degree, neighbDegree=neighbDegree, localClustering=localClustering, no0Deg=no0Deg) 
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
#changeTrainValid means multiple things
#-1 means to propagate predictions of train,valid,test
#0 means to propagate labels of train,valid on first iteration, propagate predictions thereafter
#1 means to propagate labels of train and propagate predictions on validation,test
#2 means to propagate labels of train and valid and predictions of test
#3 means to propagate labels of train and valid and predictions of test, actual propagation is the actual prediction (ie >=0.5 then 1, else 0)
#4 means to propagate labels of train and valid and predictions of test, except cut validation into 3% and 12%, use 3% as pseudotesting
#5 means to propagate labels of train and valid and predictions of test, actual propagation is the actual prediction (ie >=0.5 then 1, else 0)
#  except cut validation into 3% and 12%, use 3% as pseudotesting
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

    # fNames
    # amazon_DVD_20000
    # facebook
    # amazon_Music_10000
    test_D(accuracyOutputName="", fName=fName, noValid=False, netType=netType, memory=memory, mini_dim=mini_dim, bias_self=bias_self, testLimit=testLimit, gpu=gpu, perturb=perturb, lastH=lastH, lastHStartRan=lastHStartRan,
        usePrevWeights = usePrevWeights, PRType= prtype, singlePR=singlepr, dataAug=dataAug, randInit = randInit, pageRankOrder=pageRankOrder, useActualLabs=False, onlyLabs=False, avgNeighb=False, degree=degree, n0Deg=no0Deg,
        changeTrainValid=changeTrainValid, neighbDegree=False, avgNeighbLabel=False, dynamLabel=True, avgPosNegAttr=False, num01s=False, localClustering=False, usePro=usePro)

    
