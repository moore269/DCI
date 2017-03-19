import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from pylab import *

import networkx as nx
import readData
import copy
import graph_helper
from RelationalRNN import RelationalRNN
from operator import add
import numpy as np
import time
import re
from collections import defaultdict
import sys
import os
import glob
import re
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from sets import Set
from graph_helper import *

#original that seemed unstable
def monitorValidationAndTrain(path, name):
    numFolds=17
    percentValidation=0.15
    fName=""
    if "amazon_DVD_20000" in name:
        fName="amazon_DVD_20000" 
    elif "facebook" in name:
        fName="facebook"
    elif "amazon_Music_10000" in name:
        fName="amazon_Music_10000"
    G = readData.readDataset("data/"+fName+".edges","data/"+fName+".attr","data/"+fName+".lab" ) 

    candidateFiles=glob.glob(path+"/"+name+"*.pkl")

    for cName in candidateFiles:
        m=re.search('trial_([0-9]+)_fold_([0-9]+)', cName)
        trial=int(m.groups(0)[0])
        fold=int(m.groups(0)[1])
        #read trial from file
        rest, validationNodes= readData.readTrial(fName, trial, percentValidation)
        #split into folds
        folds= readData.splitNodeFolds(rest, numFolds)
        #add up trainNodes
        trainNodes=[] 
        for k in range(0, fold+1):
            trainNodes+=folds[k]

        #add up rest of nodes
        testNodes=[]
        for k in range(fold+1, numFolds):
            testNodes+=folds[k]

        #show how we do
        if "RNNC" not in cName:
            print("cName: "+cName)
            rnn = RelationalRNN(G=G, batch_size=10, attrKey='attr', load_path = cName)
            time.sleep(10) 
            #bae=rnn.makePredictions(trainNodes)
            #print("BAEtrain: "+str(bae))
            #time.sleep(5) 
            #bae=rnn.makePredictions(validationNodes)
            #print("BAEvalidation: "+str(bae))
            #time.sleep(5) 
            for j in range(1, 10):
                bae = rnn.makePredictions(testNodes)
                print("BAEtest: "+str(bae))
            #time.sleep(5) 

def averageModelsPred(path, trial, fold, netTypes, part1N, part2N, dataSet):
    averagePreds={}
    for i, netType in enumerate(netTypes):
        name = part1N+netType+"_"+dataSet+part2N+netType+"_"
        preds = getPreds(path, trial, fold, name)
        if i==0:
            #allocate dictionary to do averaging
            for trainvalidtest in preds:
                averagePreds[trainvalidtest]={}
                for id in preds[trainvalidtest]:
                    averagePreds[trainvalidtest][id]=0.0


        for trainvalidtest in preds:
            for id in preds[trainvalidtest]:
                averagePreds[trainvalidtest][id]+=preds[trainvalidtest][id]

    for trainvalidtest in averagePreds:
        for id in averagePreds[trainvalidtest]:
            averagePreds[trainvalidtest][id]=averagePreds[trainvalidtest][id]/len(netTypes)

    return averagePreds


def predictBAE(testing, G):
    err1=0.0
    err0=0.0
    count1=0
    count0=0

    #go through batches & record predictions
    for i, sample in enumerate(testing):
        nodeID = testing[i][0]
        pred = testing[i][1]
        actual = G.node[nodeID]['label'][0]

        if actual==1:
            err1 += (1-pred)
            count1 += 1
        elif actual == 0:
            err0 += pred
            count0 += 1

    divideBy=0
    if count1!=0:
        err1=err1/count1
        divideBy+=1
    if count0!=0:
        err0=err0/count0
        divideBy+=1

    BAE=(err1+err0)/divideBy

    return BAE

#returns error ob, the types ordered, and a list of ordered pairs for use in calculating pearson correlation
#error ob: [errorType][train valid, etc][degree or All][tuple of ID and BAE]
def getErrorDegreeOb(dataFolder, path, trial, fold, part1N="", part2N="", dataSet=""):
    G = readData.readDataset(dataFolder, dataSet)
    types=["AverageModels", "LSTM", "LSTM2", "BIDLSTM", "StackLSTM"]
    noAvgTypes = [model for model in types if not model == "AverageModels"]
    orderedTypes=sorted(types)

    idList=defaultdict(lambda:Set())
    #get the id, BAE dictionary for each train, valid, test & loop through each set
    errorType = {}
    for i, netType in enumerate(types):
        name = part1N+netType+"_"+dataSet+part2N+netType+"_"
        if netType=="AverageModels":
            predOb= averageModelsPred(path, trial, fold, noAvgTypes, part1N, part2N, dataSet)
        else:
            predOb = getPreds(path, trial, fold, name)
        if i==0:
            for trainvalidtest in predOb:
                for id in sorted(predOb[trainvalidtest]):
                    idList[trainvalidtest].add(id)

        errorSetType = {}
        #error Set Type, keys: 'Train', 'Valid', 'Test', 'Test_C'
        #error Ob, keys: degree or All, values: [bae, #nodes]
        for setType in predOb:
            degreePreds=defaultdict(lambda:[])
            AllPreds=[]
            preds=predOb[setType]
            errorOb={}
            #group by degrees
            for id in sorted(preds):
                degreePreds[G.node[id]['degree']].append([id, preds[id]])

            #group all
            for id in sorted(preds):
                AllPreds.append([id, preds[id]])

            #print(setType)
            for key in sorted(degreePreds):
                bae = predictBAE(degreePreds[key], G)
                errorOb[key] = [bae, len(degreePreds[key]), degreePreds[key]]
                #print(str(key) + ", "+str(bae)+", "+str(len(degreePreds[key])))

            bae = predictBAE(AllPreds, G)
            errorOb['All'] = [bae, len(AllPreds), AllPreds]
            #print("All, "+str(bae)+", "+str(len(AllPreds)))
            errorSetType[setType]=errorOb
        errorType[netType]=errorSetType
    return (errorType, orderedTypes, idList)

def analyzePostMortemDegrees(path, trial, fold, part1N="", part2N="", dataSet=""):
    errorType, orderedTypes, idList = getErrorDegreeOb(path, trial, fold, part1N, part2N, dataSet)
    #construct a list of ordered pairs for use in calculating pearson correlation
    orderedPairs = []
    for i in range(1, len(orderedTypes)):
        for j in range(i+1, len(orderedTypes)):
            orderedPairs.append(orderedTypes[i]+"_"+orderedTypes[j])

    for setType in errorType[orderedTypes[0]]:
        print("=========="+setType+"==========")
        print("degree\t"+"\t".join(sorted(errorType.keys()))+"\t# nodes\t" + "\t".join(orderedPairs))
        errorObKeys=errorType[orderedTypes[0]][setType].keys()
        for degree in sorted(errorObKeys):
            buildStr = str(degree)
            for netType in orderedTypes:
                buildStr=buildStr + ", " + str(errorType[netType][setType][degree][0])
            buildStr = buildStr + ", " + str(errorType[netType][setType][degree][1])

            allPairs={}
            #generate all pairs for pearson correlation coefficient
            for i in range(1, len(orderedTypes)):
                for j in range(i+1, len(orderedTypes)):
                    first = np.array(errorType[orderedTypes[i]][setType][degree][2])[:,1]
                    second = np.array(errorType[orderedTypes[j]][setType][degree][2])[:,1]
                    buildStr= buildStr + ", " + str(pearsonr(first, second)[0])

            print(buildStr)

#error ob: [errorType][train valid, etc][degree or All][tuple of ID and BAE]
#to do across trials, use parameter trial2
#to do across tests, use parameter part1N2 to specify the form of the file to read
def analyzePostMortemDegreesAcrossTrials(dataFolder, path, trial1, fold, part1N, part2N, trial2=-1, dataSet="", part1N2=""):

    if part1N2=="":
        errorType1, orderedTypes, idList1 = getErrorDegreeOb(dataFolder, path, trial1, fold, part1N, part2N, dataSet)
        errorType2, orderedTypes, idList2 = getErrorDegreeOb(dataFolder, path, trial2, fold, part1N, part2N, dataSet)
    else:
        errorType1, orderedTypes, idList1 = getErrorDegreeOb(dataFolder, path, trial1, fold, part1N, part2N, dataSet)
        errorType2, orderedTypes, idList2 = getErrorDegreeOb(dataFolder, path, trial1, fold, part1N2, part2N, dataSet)
    orderedPairs=[orderType+"t1_t2" for orderType in orderedTypes ]

    for setType in errorType1[orderedTypes[0]]:
        print("=========="+setType+"==========")
        print("degree\t"+"\t".join(sorted(errorType1.keys()))+"\t# nodes\t" + "\t".join(orderedPairs))
        degrees1=Set(errorType1[orderedTypes[0]][setType].keys())
        degrees2=Set(errorType2[orderedTypes[0]][setType].keys())
        degrees = sorted(degrees1 & degrees2)
        for degree in degrees:
            buildStr = str(degree)
            for netType in orderedTypes:
                buildStr=buildStr + ", " + str(errorType1[netType][setType][degree][0])
            buildStr = buildStr + ", " + str(errorType1[netType][setType][degree][1])

            allPairs={}
            #generate all pairs for pearson correlation coefficient
            for i in range(0, len(orderedTypes)):
                trial1Pairs=[baepair for baepair in errorType1[orderedTypes[i]][setType][degree][2] if baepair[0] in idList2[setType]]
                trial2Pairs=[baepair for baepair in errorType2[orderedTypes[i]][setType][degree][2] if baepair[0] in idList1[setType]]
                if len(trial1Pairs)!=0:
                    first = np.array(trial1Pairs)[:,1]
                    second = np.array(trial2Pairs)[:,1]
                    buildStr= buildStr + ", " + str(pearsonr(first, second)[0])
                else:
                    buildStr=buildStr+ ", nan"

            print(buildStr)

def getBAEs(predOb, G):
    errorSetType = {}
    #error Set Type, keys: 'Train', 'Valid', 'Test', 'Test_C'
    #error Ob, keys: degree or All, values: [bae, #nodes]
    for setType in predOb:
        degreePreds=defaultdict(lambda:[])
        AllPreds=[]
        preds=predOb[setType]
        errorOb={}
        #group by degrees
        for id in sorted(preds):
            degreePreds[G.node[id]['degree']].append([id, preds[id]])

        #group all
        for id in sorted(preds):
            AllPreds.append([id, preds[id]])

        #print(setType)
        for key in sorted(degreePreds):
            bae = predictBAE(degreePreds[key], G)
            errorOb[key] = [bae, len(degreePreds[key]), degreePreds[key]]
            #print(str(key) + ", "+str(bae)+", "+str(len(degreePreds[key])))

        bae = predictBAE(AllPreds, G)
        errorOb['All'] = [bae, len(AllPreds), AllPreds]
        #print("All, "+str(bae)+", "+str(len(AllPreds)))
        errorSetType[setType]=errorOb
    return errorSetType

def averagePreds(predObs, setType, id):
    sumPreds=0.0
    for predOb in predObs:
        sumPreds=sumPreds+predOb[setType][id]
    return sumPreds/len(predObs)

def majorityVote(predObs, setType, id):
    posCount=0
    negCount=0
    for predOb in predObs:
        if predOb[setType][id]>=0.5:
            posCount+=1
        elif predOb[setType][id]<0.5:
            negCount+=1
    if posCount>negCount:
        return 1.0
    elif negCount>posCount:
        return 0.0
    else:
        return 0.5
        #print("please input an odd number of predictors")
        #sys.exit(0)

    return 1.0

def getBAECombined(predObs, G, combineFunction):
    errorSetType = {}
    #error Set Type, keys: 'Train', 'Valid', 'Test', 'Test_C'
    #error Ob, keys: degree or All, values: [bae, #nodes]
    predOb1 = predObs[0]
    for setType in predOb1:
        degreePreds=defaultdict(lambda:[])
        AllPreds=[]
        preds1=predObs[0][setType]
        predsC = {}
        for id in sorted(preds1):
            predsC[id] = combineFunction(predObs, setType, id)

        errorOb={}
        #group by degrees
        for id in sorted(predsC):
            degreePreds[G.node[id]['degree']].append([id, predsC[id]])

        #group all
        for id in sorted(predsC):
            AllPreds.append([id, predsC[id]])

        #print(setType)
        for key in sorted(degreePreds):
            bae = predictBAE(degreePreds[key], G)
            errorOb[key] = [bae, len(degreePreds[key]), degreePreds[key]]
            #print(str(key) + ", "+str(bae)+", "+str(len(degreePreds[key])))

        bae = predictBAE(AllPreds, G)
        errorOb['All'] = [bae, len(AllPreds), AllPreds]
        #print("All, "+str(bae)+", "+str(len(AllPreds)))
        errorSetType[setType]=errorOb
    return errorSetType 



#returns error ob, the types ordered, and a list of ordered pairs for use in calculating pearson correlation
#error ob: [errorType][train valid, etc][degree or All][tuple of ID and BAE]
def ensemblePosNegs(dataFolder, path, pathOrig, dataSet="", degree=True, Mem=10, testLimit=False, ensemblePredsFunction=majorityVote):
    defaultFile = "LSTM_amazon_DVD_20000_noVal_F_Mem_10_Ex_1_MultTra_T_maxEpoch_200_maxNProp_5_trials_10_selectFolds_T_avgNeigLab_F_dynamLab_T_useActLabs_F_onlyLabs_F_PPR_T_onlyPR_F_prtype_neutral_biasself_pos_d_F_lim_F_sd_3_predictions_LSTM_"
    #defaultFile2 = "_netType_LSTM_amazon_DVD_20000_noValid_F_Mem_10_Ex_1_MultTrain_T_maxEpochs_300_maxNProp_10_trials_5_selectFolds_T_avgNeighbLabel_F_dynamLabel_T_useActualLabs_F_onlyLabs_F_PPR_T_singlePR_F_prtype_neutral_d_T_predictions_LSTM_"
    defaultFile2 = "LSTM_amazon_DVD_20000_noVal_F_Mem_10_Ex_1_MultTra_T_maxEpoch_200_maxNProp_5_trials_10_selectFolds_T_avgNeigLab_F_dynamLab_T_useActLabs_F_onlyLabs_F_PPR_T_onlyPR_F_prtype_neutral_biasself_none_d_F_lim_F_sd_3_predictions_LSTM_"
    namePos = defaultFile.replace("amazon_DVD_20000", dataSet).replace("_d_F", "_d_" + bStr(degree)).replace("_Mem_10", "_Mem_"+ str(Mem)).replace("_lim_F", "_lim_"+bStr(testLimit))
    nameNeg = namePos.replace("biasself_pos", "biasself_neg")
    nameOrig = defaultFile2.replace("amazon_DVD_20000", dataSet).replace("_d_F", "_d_" + bStr(degree)).replace("_Mem_10", "_Mem_"+ str(Mem))
    namePosNeg = defaultFile2.replace("LSTM", "LSTMPAR").replace("biasself_none", "biasself_posneg").replace("amazon_DVD_20000", dataSet).replace("_d_F", "_d_" + bStr(degree)).replace("_Mem_10", "_Mem_"+ str(Mem))

    G = readData.readDataset(dataFolder, dataSet)

    idList=defaultdict(lambda:Set())
    folds = [1, 3, 5, 7, 9, 11, 13, 15]
    trials = xrange(0, 10)
    if dataSet=="amazon_DVD_20000":
        predKeys = ['Pos', 'Neg', 'Orig', 'C']
    else:
        predKeys = ['Pos', 'Neg', 'Orig', 'PosNeg', 'C']
    setKeys = ['Test', 'Train', 'Valid', 'Test_C']

    #instantiate dataset
    data={}
    for setType in setKeys:
        data[setType]={}
        for key in predKeys:
            data[setType][key] = np.zeros((len(folds), len(trials)))
    dataSum={}

    #gather data
    for fold in range(0, len(folds)):
        dataSum[fold]={setKeys[0]: defaultdict(lambda: 0.0), setKeys[1]: defaultdict(lambda: 0.0), setKeys[2]: defaultdict(lambda: 0.0), setKeys[3]: defaultdict(lambda: 0.0)}

        for trial in trials:
            #get the id, BAE dictionary for each train, valid, test & loop through each set
            #print("trial: "+str(trial)+", fold: "+str(fold))
            predObPos = getPreds(path, trial, fold, namePos)
            predObNeg = getPreds(path, trial, fold, nameNeg)
            predObOrig = getPreds(pathOrig, trial, fold, nameOrig)
            if dataSet!="amazon_DVD_20000":
                predObPosNeg = getPreds(path, trial, fold, namePosNeg)

            baePos = getBAEs(predObPos, G)
            baeNeg = getBAEs(predObNeg, G)
            baeOrig = getBAEs(predObOrig, G)
            if dataSet!="amazon_DVD_20000":
                baePosNeg = getBAEs(predObPosNeg, G)

            baeC = getBAECombined([predObPos, predObNeg, predObOrig], G, ensemblePredsFunction)

            if dataSet=="amazon_DVD_20000":
                baeObs = {predKeys[0]: baePos, predKeys[1]: baeNeg, predKeys[2]: baeOrig, predKeys[3]: baeC}
            else:
                baeObs = {predKeys[0]: baePos, predKeys[1]: baeNeg, predKeys[2]: baeOrig, predKeys[3]: baePosNeg, predKeys[4]: baeC}
            for setType in setKeys:
                for key in baeObs:
                    error = baeObs[key][setType]['All'][0]
                    data[setType][key][fold][trial] = error
                    dataSum[fold][setType][key] += error
    
    strTitle = "\t"
    for key in predKeys:
        strTitle=strTitle+key+"\t\t"
    strTitle = strTitle+"C_stds"
    print(strTitle)

    for fold, realFold in enumerate(folds):
        print("Fold: "+str(realFold))
        for setType in baePos:
            strRow = setType
            for key in predKeys:
                error = dataSum[fold][setType][key]/len(trials)
                strRow = strRow+"\t"+str(error)
            strRow = strRow + "\t"+str(np.std(data[setType]['C'][fold]))
            print(strRow)


#returns error ob, the types ordered, and a list of ordered pairs for use in calculating pearson correlation
#error ob: [errorType][train valid, etc][degree or All][tuple of ID and BAE]
def analyzeCorPosNegs(dataFolder, path, pathOrig, trial, dataSet="", degree=True, Mem=10, testLimit=False, ensemblePredsFunction=majorityVote):
    defaultFile = "LSTM_amazon_DVD_20000_noVal_F_Mem_10_Ex_1_MultTra_T_maxEpoch_200_maxNProp_5_trials_10_selectFolds_T_avgNeigLab_F_dynamLab_T_useActLabs_F_onlyLabs_F_PPR_T_onlyPR_F_prtype_neutral_biasself_pos_d_F_lim_F_sd_3_predictions_LSTM_"
    #defaultFile2 = "_netType_LSTM_amazon_DVD_20000_noValid_F_Mem_10_Ex_1_MultTrain_T_maxEpochs_300_maxNProp_10_trials_5_selectFolds_T_avgNeighbLabel_F_dynamLabel_T_useActualLabs_F_onlyLabs_F_PPR_T_singlePR_F_prtype_neutral_d_T_predictions_LSTM_"
    defaultFile2 = "LSTM_amazon_DVD_20000_noVal_F_Mem_10_Ex_1_MultTra_T_maxEpoch_200_maxNProp_5_trials_10_selectFolds_T_avgNeigLab_F_dynamLab_T_useActLabs_F_onlyLabs_F_PPR_T_onlyPR_F_prtype_neutral_biasself_none_d_F_lim_F_sd_3_predictions_LSTM_"
    namePos = defaultFile.replace("amazon_DVD_20000", dataSet).replace("_d_F", "_d_" + bStr(degree)).replace("_Mem_10", "_Mem_"+ str(Mem)).replace("_lim_F", "_lim_"+bStr(testLimit))
    nameNeg = namePos.replace("biasself_pos", "biasself_neg")
    nameOrig = defaultFile2.replace("amazon_DVD_20000", dataSet).replace("_d_F", "_d_" + bStr(degree)).replace("_Mem_10", "_Mem_"+ str(Mem))
    namePosNeg = defaultFile2.replace("LSTM", "LSTMPAR").replace("biasself_none", "biasself_posneg").replace("amazon_DVD_20000", dataSet).replace("_d_F", "_d_" + bStr(degree)).replace("_Mem_10", "_Mem_"+ str(Mem))

    G = readData.readDataset(dataFolder, dataSet)

    idList=defaultdict(lambda:Set())
    folds = [1, 3, 5, 7, 9, 11, 13, 15]

    #select which folds you want
    selectedFolds = [1, 13]


    trials = xrange(0, 10)

    #instantiate dataset
    data = {}
    dataSum={}

    #gather data
    for fold, realFold in enumerate(folds):
        if realFold not in selectedFolds:
            continue
        
        data[fold]={}
        dataSum[fold]={}
        data[fold][trial]={}
        #get the id, BAE dictionary for each train, valid, test & loop through each set
        #print("trial: "+str(trial)+", fold: "+str(fold))
        predObPos = getPreds(path, trial, fold, namePos)
        predObNeg = getPreds(path, trial, fold, nameNeg)
        predObOrig = getPreds(pathOrig, trial, fold, nameOrig)
        predObPosNeg = getPreds(path, trial, fold, namePosNeg)

        baePos = getBAEs(predObPos, G)
        baeNeg = getBAEs(predObNeg, G)
        baeOrig = getBAEs(predObOrig, G)
        baePosNeg = getBAEs(predObPosNeg, G)

        baeC = getBAECombined([predObPos, predObNeg, predObOrig, predObPosNeg], G, ensemblePredsFunction)
        for setType in baePos:
            print("Set\tFold: "+str(realFold)+" Pos_Neg\tPos_O\t\tNeg_O\t\tPosNeg_O\tPos_PosNeg\tNeg_PosNeg\tPos_C\t\tNeg_C\t\tO_C\t\tPosNeg_C")
            for deg in baePos[setType]:

                data[fold][trial][setType]={}
                dataSum[fold][setType]=defaultdict(lambda: 0.0)
                predsPos = baePos[setType][deg][2]
                predsNeg = baeNeg[setType][deg][2]
                predsOrig = baeOrig[setType][deg][2]
                predsPosNeg = baePosNeg[setType][deg][2]
                predsC = baeC[setType][deg][2]
                
                #pearson correlation inputs
                predsPosP = np.array(predsPos)[:,1]
                predsNegP = np.array(predsNeg)[:,1]
                predsOrigP = np.array(predsOrig)[:,1]
                predsPosNegP = np.array(predsPosNeg)[:,1]
                predsCP = np.array(predsC)[:,1]

                if deg =='All':
                    plotPreds(path, setType+ "_data_"+dataSet+ "_Trial_"+str(trial)+"_Fold_"+str(realFold) + "_pos_neg", predsPosP, predsNegP, "positive", "negative")
                    plotPreds(path, setType+ "_data_"+dataSet+ "_Trial_"+str(trial)+"_Fold_"+str(realFold) + "_pos_C", predsPosP, predsCP, "positive", "combined")
                    plotPreds(path, setType+ "_data_"+dataSet+ "_Trial_"+str(trial)+"_Fold_"+str(realFold) + "_neg_C", predsNegP, predsCP, "negative", "combined")

                    plotPreds(path, setType+ "_data_"+dataSet+ "_Trial_"+str(trial)+"_Fold_"+str(realFold) + "_pos_Orig", predsPosP, predsOrigP, "positive", "orig")
                    plotPreds(path, setType+ "_data_"+dataSet+ "_Trial_"+str(trial)+"_Fold_"+str(realFold) + "_neg_Orig", predsNegP, predsOrigP, "negative", "orig")
                    plotPreds(path, setType+ "_data_"+dataSet+ "_Trial_"+str(trial)+"_Fold_"+str(realFold) + "_pos_posneg", predsPosP, predsPosNegP, "positive", "posneg")
                    plotPreds(path, setType+ "_data_"+dataSet+ "_Trial_"+str(trial)+"_Fold_"+str(realFold) + "_neg_posneg", predsNegP, predsPosNegP, "negative", "posneg")
                    plotPreds(path, setType+ "_data_"+dataSet+ "_Trial_"+str(trial)+"_Fold_"+str(realFold) + "_posneg_Orig", predsPosNegP, predsOrigP, "posneg", "orig")
                    plotPreds(path, setType+ "_data_"+dataSet+ "_Trial_"+str(trial)+"_Fold_"+str(realFold) + "_posneg_Orig", predsPosNegP, predsOrigP, "posneg", "orig")

                predStr = setType+"\t" + str(deg) + "\t" + str(pearsonr(predsPosP, predsNegP)[0])+"\t" + str(pearsonr(predsPosP, predsOrigP)[0]) + "\t" + str(pearsonr(predsNegP, predsOrigP)[0])+\
                "\t" + str(pearsonr(predsPosNegP, predsOrigP)[0])+"\t"+str(pearsonr(predsPosP, predsPosNegP)[0])+"\t"+str(pearsonr(predsNegP, predsPosNegP)[0])+\
                "\t" + str(pearsonr(predsPosP, predsCP)[0])+"\t"+str(pearsonr(predsNegP, predsCP)[0])+"\t"+str(pearsonr(predsOrigP, predsCP)[0]) + "\t"+str(pearsonr(predsPosNegP, predsCP)[0]) 
                    
                print(predStr)

        predsPos = np.array(baePos['Test_C']['All'][2])[:,1]
        predsNeg = np.array(baeNeg['Test_C']['All'][2])[:,1]
        counts = countLabels(predsPos, predsNeg)
        print("Test_C Counts\t"+str(counts[0])+"\t"+str(counts[1])+"\t"+str(counts[2])+"\t"+str(counts[3]))

#returns error ob, the types ordered, and a list of ordered pairs for use in calculating pearson correlation
#error ob: [errorType][train valid, etc][degree or All][tuple of ID and BAE]
def analyzeCorPosNegs2(dataFolder, path, pathOrig, pathJoel, trial, dataSet="", degree=True, Mem=10, testLimit=False, ensemblePredsFunction=majorityVote):
    defaultFile = "LSTM_amazon_DVD_20000_noVal_F_Mem_10_Ex_1_MultTra_T_maxEpoch_200_maxNProp_5_trials_10_selectFolds_T_avgNeigLab_F_dynamLab_T_useActLabs_F_onlyLabs_F_PPR_T_onlyPR_F_prtype_neutral_biasself_pos_d_F_lim_F_sd_3_predictions_LSTM_"
    #defaultFile2 = "_netType_LSTM_amazon_DVD_20000_noValid_F_Mem_10_Ex_1_MultTrain_T_maxEpochs_300_maxNProp_10_trials_5_selectFolds_T_avgNeighbLabel_F_dynamLabel_T_useActualLabs_F_onlyLabs_F_PPR_T_singlePR_F_prtype_neutral_d_T_predictions_LSTM_"
    defaultFile2 = "LSTM_amazon_DVD_20000_noVal_F_Mem_10_Ex_1_MultTra_T_maxEpoch_200_maxNProp_5_trials_10_selectFolds_T_avgNeigLab_F_dynamLab_T_useActLabs_F_onlyLabs_F_PPR_T_onlyPR_F_prtype_neutral_biasself_none_d_F_lim_F_sd_3_predictions_LSTM_"
    nameOrig = defaultFile2.replace("amazon_DVD_20000", dataSet).replace("_d_F", "_d_" + bStr(degree)).replace("_Mem_10", "_Mem_"+ str(Mem))

    G = readData.readDataset(dataFolder, dataSet)

    idList=defaultdict(lambda:Set())
    folds = [1, 3, 5, 7, 9, 11, 13, 15]

    #select which folds you want
    selectedFolds = folds


    trials = xrange(0, 10)

    #instantiate dataset
    data = {}
    dataSum={}
    for trial in trials:
        print(trial)
        #gather data
        for fold, realFold in enumerate(folds):
            if realFold not in selectedFolds:
                continue
            
            data[fold]={}
            dataSum[fold]={}
            data[fold][trial]={}
            #get the id, BAE dictionary for each train, valid, test & loop through each set
            #print("trial: "+str(trial)+", fold: "+str(fold))
            predObOrigPre = getPreds(pathOrig, trial, fold, nameOrig)
            predObJoel = readJoelResults(pathJoel, dataSet, trial, realFold)

            #make sure we only use preds that Joel uses (non-0 deg nodes) on test set
            predObOrig = {'Train':{}, 'Valid':{}, 'Test':{}, 'Test_C':{}}
            TVTsets = ['Train', 'Valid', 'Test']
            for key in predObJoel['Test_C']:
                predObOrig['Test_C'][key] = predObOrigPre['Test_C'][key]
            for setType in TVTsets:
                for key in predObOrigPre[setType]:
                    predObOrig[setType][key] = predObOrigPre[setType][key]

            baeOrig = getBAEs(predObOrig, G)
            baeJoel = getBAEs(predObJoel, G)
            print('Fold: '+str(realFold)+', BAE: '+str(baeJoel['Test_C']['All'][0]))

            #baeC = getBAECombined([predObPos, predObNeg, predObOrig, predObPosNeg], G, ensemblePredsFunction)
            setType='Test_C'
            print("Set\tFold: "+str(realFold)+" Orig_Joel")
            for deg in baeOrig[setType]:

                data[fold][trial][setType]={}
                dataSum[fold][setType]=defaultdict(lambda: 0.0)
                predsOrig = baeOrig[setType][deg][2]
                predsJoel = baeJoel[setType][deg][2]
                #pearson correlation inputs
                predsOrigP = np.array(predsOrig)[:,1]
                predsJoelP = np.array(predsJoel)[:,1]

                if deg =='All':
                    plotPreds(path, setType+ "_data_"+dataSet+ "_Trial_"+str(trial)+"_Fold_"+str(realFold) + "_Orig_Joel", predsOrigP, predsJoelP, "Original", "Joel")

                predStr = setType+"\t" + str(deg) + "\t" + str(pearsonr(predsOrigP, predsJoelP)[0])
                    
                #print(predStr)  

def readJoelResults(pathJoel, dataSet, trial, fold):
    fName = pathJoel+dataSet+"/"+dataSet+"_trial_"+str(trial)+"_fold_"+str(fold)+"_PL-EM (CAL).txt"
    f = open(fName, 'r')
    setType='Test_C'
    predictions = {setType:{}}
    for i, line in enumerate(f):
        if i!=0:
            fields = line.replace("\n", "").split(",")
            id=int(fields[0])
            prob = float(fields[1])
            predictions[setType][id] = prob
    return predictions

def countLabels(preds1, preds2):
    countPosPos = 0
    countPosNeg = 0
    countNegPos = 0
    countNegNeg = 0
    for i in range(0, len(preds1)):
        pred1 = preds1[i]
        pred2 = preds2[i]
        if pred1 >= 0.5 and pred2 >= 0.5:
            countPosPos+=1
        elif pred1 >= 0.5 and pred2 < 0.5:
            countPosNeg+=1
        elif pred1 < 0.5 and pred2 >= 0.5:
            countNegPos+=1
        elif pred1 < 0.5 and pred2 < 0.5:
            countNegNeg+=1
        else:
            print("pred needs to be real valued")
            sys.exit(0)

    return (countPosPos, countPosNeg, countNegPos, countNegNeg)


#plot predictions against each other
#x-axis - preds1 predictions
#y-axis - preds2 predictions
def plotPreds(path, name, preds1, preds2, preds1Name, preds2Name):
    fig = plt.figure()
    scatter(preds1, preds2)
    xlabel(preds1Name)
    ylabel(preds2Name)
    title(name)
    savefig(path+"output/"+name+".png")
    close(fig)

#helper function to return T or F for True or False. This helps to cut down on fileName size
def bStr(boolStr):
    if boolStr:
        return "T"
    else:
        return "F"


if __name__ == "__main__":
    #monitorValidationAndTrain("models", 'Rnn_vary_training_A_BAE_amazon_DVD_20000_')
    #monitorValidationAndTrain("models", "Rnn_vary_training_A_All_BAE_train_NoAvg_facebook_noValid_False_Mem_10_Ex_10_MultTrain_True_maxEpochs_300_maxNProp_10_trials_5_selectFolds_True_avgNeighbLabel_True_dynamLabel_True_")
    #monitorValidationAndTrain("models", "Rnn_vary_training_A_BAE_train_NoAvg_amazon_DVD_20000_noValid_False_Mem_10_Ex_10_MultTrain_True_maxEpochs_300_maxNProp_10_trials_5_selectFolds_True_avgNeighbLabel_True_dynamLabel_True_")
    dataSet="facebook"
    #dataSet="amazon_DVD_20000"
    #dataSet="amazon_Music_10000"

    #ensemblePosNegs(dataFolder="data/", path="output/output_Feb_8_2016/", pathOrig = "output/output_Feb_25_original/", dataSet=dataSet)
    #analyzeCorPosNegs(dataFolder="data/", path="output/output_Feb_8_2016/", pathOrig = "output/output_Feb_25_original/", trial=2, dataSet=dataSet)
    
    #ensemblePosNegs(dataFolder="data/", path="output/output_Mar_1/", pathOrig = "output/output_Feb_25_original/", dataSet=dataSet)
    #analyzeCorPosNegs(dataFolder="data/", path="output/output_Mar_1/", pathOrig = "output/output_Feb_25_original/", trial=2, dataSet=dataSet)
    analyzeCorPosNegs2(dataFolder="data/", path="output/output_Mar_1/", pathOrig = "output/output_Feb_25_original/", pathJoel="JoelOutput/", trial=2, dataSet=dataSet)


