import sys
import os
#add previous folder to path
cwd = os.getcwd(); rIndex = cwd.rfind("/");cwd = cwd[:rIndex];
sys.path.insert(0, cwd)

import numpy as numpy
from networkx.algorithms.link_analysis import pagerank_scipy, pagerank_numpy
import code.readData.readData as readData
import copy
import sys
import math
from collections import defaultdict
import cPickle as pickle
import operator
import time
import os

#weightMult is for numerical stability, default=4 to have whole integers
def personalizeTransMat(G, isTrain, weightMult, label):
    for n1, n2 in G.edges_iter():
        #if we know both end vertices
        # both good - 4
        # 1 good - 2
        # 0 good - 1
        if isTrain[n1] and isTrain[n2]:
            if label==G.node[n1]['label'][0] and label==G.node[n2]['label'][0]:
                G[n1][n2]['weight']=G[n1][n2]['weight']*weightMult*4
            elif label==G.node[n1]['label'][0] or label==G.node[n2]['label'][0]:
                G[n1][n2]['weight']=G[n1][n2]['weight']*weightMult*2
            else:
                G[n1][n2]['weight']=G[n1][n2]['weight']*weightMult
        #if we only know 1 vertice
        # 1 good - 3
        # 0 good - 1.5
        elif isTrain[n1]:
            if label==G.node[n1]['label'][0]:
                G[n1][n2]['weight']=G[n1][n2]['weight']*weightMult*3
            else:
                G[n1][n2]['weight']=G[n1][n2]['weight']*weightMult*1.5
        elif isTrain[n2]:
            if label==G.node[n2]['label'][0]:
                G[n1][n2]['weight']=G[n1][n2]['weight']*weightMult*3
            else:
                G[n1][n2]['weight']=G[n1][n2]['weight']*weightMult*1.5
        #if we know none
        # none - 2.25
        else:
            G[n1][n2]['weight']=G[n1][n2]['weight']*weightMult*2.25

def computePersonalizedPR(G, trainNodes, testNodes, alpha=0.85, label=None, weightMult=4.0, debug=True, saveFullPR=False):
    Gorig=G
    #dictionary for known nodes and unknown nodes
    isTrain = defaultdict(lambda:False)
    for node in trainNodes:
        isTrain[node]=True

    defaultVector={}
    for node in G.nodes():
        defaultVector[node]=0.0  

    #else, we want to favor either positive or negative labeled nodes
    if label!=None and label!='similar':
        #if positive label, then we give preference to positive labeled nodes
        if label=="pos":
            label=1.0
        elif label=="neg":
            label=0.0
        personalizeTransMat(G, isTrain, weightMult, label)

    origLabel = label
    PRs={}
    top10PRs={}
    print(len(testNodes))
    startTime=time.time()

    if debug:
        Gnodes=G.nodes()[0:2]
    else:
        Gnodes=G.nodes()
    #run pagerank for each vector where teleportation always starts from candidate node
    for i, node in enumerate(Gnodes):
        vector=copy.deepcopy(defaultVector)
        vector[node]=1.0

        #only have to change transition matrix if this changes per node
        if origLabel=='similar':
            G = copy.deepcopy(Gorig)
            label=G.node[node]['label'][0]
            personalizeTransMat(G, isTrain, weightMult, label)

        prNode=pagerank_scipy(G, alpha=alpha, personalization=vector)
        if saveFullPR:
            PRs[node] = prNode

        sortedPR = sorted(prNode.items(), key=operator.itemgetter(1), reverse=True)
        top10PRs[node]=sortedPR[0:100]
        if (i%100==0):
            endTime=time.time()
            #print("1 took "+str(endTime-startTime))
            startTime=time.time()
       #print("i="+str(i)+", done")
    return (PRs, top10PRs)

#runs 3 personalized page ranks, but only gets 10 first nodes in test set
def unitTest1():
    fName="facebook"
    numFolds=10
    maxFolds=8
    trial=0
    Gorig = readData.readDataset("data/"+fName+".edges","data/"+fName+".attr","data/"+fName+".lab") 
    rest, validationNodes= readData.readTrial(fName, trial, 0.15)
    folds= readData.splitNodeFolds(rest, numFolds)

    trainNodes=[]
    testNodes=[]
    for i in range(0, maxFolds):
        trainNodes+=folds[i]
    trainNodes+=validationNodes
    for i in range(maxFolds, numFolds):
        testNodes+=folds[i]

    testNodes=testNodes[0:10]
    Gpos=copy.deepcopy(Gorig)
    testPRpos=computePersonalizedPR(Gpos, trainNodes, testNodes, label='pos')
    Gneg=copy.deepcopy(Gorig)
    testPRneg=computePersonalizedPR(Gneg, trainNodes, testNodes, label='neg')
    Gnn=copy.deepcopy(Gorig)
    testPRneutral=computePersonalizedPR(Gnn, trainNodes, testNodes)
    Gsimilar = copy.deepcopy(Gorig)
    testPRsimilar=computePersonalizedPR(Gsimilar, trainNodes, testNodes, label='similar')

    for node in testPRpos.keys():
        sorted_PRpos = sorted(testPRpos[node].items(), key=operator.itemgetter(1))
        sorted_PRneg = sorted(testPRneg[node].items(), key=operator.itemgetter(1))
        sorted_PRneutral = sorted(testPRneutral[node].items(), key=operator.itemgetter(1))
        sorted_PRsimilar = sorted(testPRsimilar[node].items(), key=operator.itemgetter(1))

        norm2pneut=0.0
        norm2nn=0.0
        norm2pneg=0.0
        norm2psimilar=0.0
        for key in testPRpos[node].keys():
            norm2pneut+=math.pow(testPRpos[node][key] - testPRneutral[node][key], 2)
            norm2nn+=math.pow(testPRneg[node][key] - testPRneutral[node][key], 2)
            norm2pneg+=math.pow(testPRpos[node][key] - testPRneg[node][key], 2)
            norm2psimilar+=math.pow(testPRpos[node][key] - testPRsimilar[node][key], 2)
        norm2pneut=math.sqrt(norm2pneut)
        norm2nn=math.sqrt(norm2nn)
        norm2pneg=math.sqrt(norm2pneg)
        norm2psimilar=math.sqrt(norm2psimilar)
        print("norm2pneut: "+str(norm2pneut))
        print("norm2nn: "+str(norm2nn))
        print("norm2pneg: "+str(norm2pneg))
        print("norm2psimilar: "+str(norm2psimilar))

    pickle.dump( testPRpos, open( "data/"+fName+"_pos_trial_"+str(trial)+"_fold_"+str(maxFolds)+".p", "wb" ) )
    testPRpos2=pickle.load( open( "data/"+fName+"_pos_trial_"+str(trial)+"_fold_"+str(maxFolds)+".p", "rb" ) )
    print("here")

def savePPR(fName, dataFolder, trial, debug=True):
    numFolds=17
    Gorig = readData.readDataset(dataFolder, fName) 

    startOverall=time.time()

    rest, validationNodes= readData.readTrial(fName, trial, 0.15)
    folds= readData.splitNodeFolds(rest, numFolds)

    for fold in range(0, numFolds-1):
        startFold=time.time()
        trainNodes=[]
        testNodes=[]
        for i in range(0, fold+1):
            trainNodes+=folds[i]
        trainNodes+=validationNodes
        for i in range(fold+1, numFolds):
            testNodes+=folds[i]

        testNodes=testNodes
        #copy and then run page rank
        G=copy.deepcopy(Gorig)
        pr, top10pr=computePersonalizedPR(G, trainNodes, testNodes, label='pos', debug=debug)
        pickle.dump( pr, open( dataFolder+fName+"_fullpr_pos_trial_"+str(trial)+"_fold_"+str(fold)+".p", "wb" ) ) 
        pickle.dump( top10pr, open( dataFolder+fName+"_10pr_pos_trial_"+str(trial)+"_fold_"+str(fold)+".p", "wb" ) ) 

        G=copy.deepcopy(Gorig)
        pr, top10pr=computePersonalizedPR(G, trainNodes, testNodes, label='neg', debug=debug)
        pickle.dump( pr, open( dataFolder+fName+"_fullpr_neg_trial_"+str(trial)+"_fold_"+str(fold)+".p", "wb" ) )
        pickle.dump( top10pr, open( dataFolder+fName+"_10pr_neg_trial_"+str(trial)+"_fold_"+str(fold)+".p", "wb" ) )

        G=copy.deepcopy(Gorig)
        pr, top10pr=computePersonalizedPR(G, trainNodes, testNodes, debug=debug)
        pickle.dump( pr, open( dataFolder+fName+"_fullpr_neutral_trial_"+str(trial)+"_fold_"+str(fold)+".p", "wb" ) ) 
        pickle.dump( top10pr, open( dataFolder+fName+"_10pr_neutral_trial_"+str(trial)+"_fold_"+str(fold)+".p", "wb" ) ) 

        endFold=time.time()
        print("Fold "+str(fold)+": "+str(endFold-startFold))

    endOverall=time.time()
    print("AllTime: "+str(endOverall-startOverall))

def savePPRtype(fName, dataFolder, trial, fold, prType, debug=True):
    numFolds=17
    Gorig = readData.readDataset(dataFolder, fName)

    startOverall=time.time()

    rest, validationNodes= readData.readTrial(fName, trial, 0.15)
    folds= readData.splitNodeFolds(rest, numFolds)

    startFold=time.time()
    trainNodes=[]
    testNodes=[]
    for i in range(0, fold+1):
        trainNodes+=folds[i]
    trainNodes+=validationNodes
    for i in range(fold+1, numFolds):
        testNodes+=folds[i]

    testNodes=testNodes
    #copy and then run page rank
    G=copy.deepcopy(Gorig)
    if prType=='pos':  
        pr, top10pr=computePersonalizedPR(G, trainNodes, testNodes, label='pos', debug=debug)
        pickle.dump( pr, open( dataFolder+fName+"_fullpr_pos_trial_"+str(trial)+"_fold_"+str(fold)+".p", "wb" ) ) 
        pickle.dump( top10pr, open( dataFolder+fName+"_10pr_pos_trial_"+str(trial)+"_fold_"+str(fold)+".p", "wb" ) ) 
    elif prType=='neg':
        pr, top10pr=computePersonalizedPR(G, trainNodes, testNodes, label='neg', debug=debug)
        pickle.dump( pr, open( dataFolder+fName+"_fullpr_neg_trial_"+str(trial)+"_fold_"+str(fold)+".p", "wb" ) )
        pickle.dump( top10pr, open( dataFolder+fName+"_10pr_neg_trial_"+str(trial)+"_fold_"+str(fold)+".p", "wb" ) )
    elif prType=='neutral':
        pr, top10pr=computePersonalizedPR(G, trainNodes, testNodes, debug=debug)
        #pickle.dump( pr, open( dataFolder+fName+"_fullpr_neutral_trial_"+str(trial)+"_fold_"+str(fold)+".p", "wb" ) ) 
        pickle.dump( top10pr, open( dataFolder+fName+"_10pr_neutral_trial_"+str(trial)+"_fold_"+str(fold)+".p", "wb" ) ) 

    endFold=time.time()
    print("Trial " + str(trial) + " Fold "+str(fold)+": "+str(endFold-startFold))


if __name__ == "__main__":
    #unitTest1()

    #are we on the cluster computers?
    scratch=os.environ.get('RCAC_SCRATCH')
    if not scratch ==None:
        dataFolder=scratch+"/GraphDeepLearning/data/"
        debug=False
    else:
        dataFolder="data/"
        debug=True

    name = sys.argv[1]
    trial=int(sys.argv[2])
    fold=int(sys.argv[3])
    prType=sys.argv[4]

    if fold==-1:
        for fold in xrange(0, 17):
            savePPRtype(name, dataFolder, trial, fold, prType, debug=debug)
    else:
        savePPRtype(name, dataFolder, trial, fold, prType, debug=debug)