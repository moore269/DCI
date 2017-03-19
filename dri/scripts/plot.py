import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import math
from pylab import *
from maxEntInf import *
import numpy as np
import os.path
import sys
# Load config parameters
nodeType=os.environ.get('RCAC_SCRATCH')
if nodeType==None:
    JoelFolder = "/Users/macbookair/Documents/Joel/"
else:
    JoelFolder =nodeType+"/Joel/"
#trials = 5
selectFolds = [1, 3, 5, 7, 9, 11, 13, 15]
#selectFolds = [3, 7, 15]
#selectFolds = [6]

def plotAcc(dataY, dataStd, titleName, rnnMeans=None, rnnStds=None, stderr=True, trials=range(0, 10)):


    folds = selectFolds
    x = np.array(folds)*0.05+0.15
    print(x)
    axes = gca()
    axes.set_xlim([0.15, 0.95])
    ax = subplot(111)

    #blue, green, red, cyan, magenta, yellow, black:
    #ax.set_color_cycle(['cyan', 'magenta', 'black', 'yellow'])
    #ax.set_color_cycle(['blue', 'green', 'red', 'cyan', 'black', 'magenta', 'yellow'])
    #plt.ylim((0.3, 0.55))
    for key in dataY:
        if key!='trial':
            y = np.array(dataY[key])
            std = np.array(dataStd[key])

            print(key)
            print('y: '+'\t'.join(map(str, dataY[key])))
            print('std: '+'\t'.join(map(str, dataStd[key])))
            if stderr:
                std = std / math.sqrt(len(trials))

            ax.errorbar(x, y, yerr=std, label=key.replace("PL-EM (CAL)", "PL-EM-M"))

    #if we have original rnn input
    if rnnMeans!=None and rnnStds!=None:
        for key in rnnMeans:
            x = np.array(selectFolds)*0.05+0.15

            print(key)
            print('y: '+'\t'.join(map(str, rnnMeans[key])))
            print('std: '+'\t'.join(map(str, rnnStds[key])))
            std = rnnStds[key]
            if stderr:
                std = rnnStds[key] / math.sqrt(len(trials))
            ax.errorbar(x, rnnMeans[key], yerr=std, label=key)

    xlabel('Train Set Proportion')
    ylabel('BAE')
    box=ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, fancybox=True, shadow=True)
    grid(True)
    #title(titleName)
    savefig(titleName+".pdf")
    close()

def readData(filePath):
    f = open(filePath, 'r')
    data = {}
    fieldKeys = {}
    for i, line in enumerate(f):
        fields = line.replace("\n", "").split("\t")
        if i==0:
            for j, field in enumerate(fields):
                if "Trials" in field:
                    field = "trial"
                data[field]=[]
                fieldKeys[j]=field
        else:
            for j, field in enumerate(fields):
                data[fieldKeys[j]].append(float(field))

    foldsDone = data['trial']
    foldIndices = {}
    #record indices
    for fold in selectFolds:
        for i, foldDone in enumerate(foldsDone):
            if fold == foldDone:
                foldIndices[fold]=i

    for key in data:
        if key!='trial':
            newVals = []
            for fInd, fold in enumerate(selectFolds):
                newVals.append(data[key][foldIndices[fold]])
            data[key] = newVals


    return data

#read in our original input in .np format
#return mean and std
def readNP(names, filePaths, trials):
    means = {}
    stds = {}
    for i, path in enumerate(filePaths):
        x = np.zeros((len(trials), len(selectFolds)))
        for tInd, trial in enumerate(trials):
            for fInd, fold in enumerate(selectFolds):
                newPath = path.replace("trial_0", "trial_"+str(trial)).replace("fold_0", "fold_"+str(fold))
                if "MEF" in names[i]:
                    baeIndex = newPath.find("BAE")
                    try:
                        x[tInd][fInd] = maxEntInf(newPath[:baeIndex]+"pre_")
                    except:
                        print("NP: trial: "+str(trial)+", fold: "+str(fold))
                        print(newPath[:baeIndex]+"pre_") #pass
                else:
                    try:
                        x[tInd][fInd] = np.load(newPath)
                        #passed=True
                    except:
                        print("NP: trial: "+str(trial)+", fold: "+str(fold))
                        print(newPath) #pass

        means[names[i]] = np.mean(x, axis=0)
        stds[names[i]] = np.std(x, axis=0)
    return (means, stds)

def readBest(names, filePaths, trials):
    means = {}
    stds = {}
    for i, path in enumerate(filePaths):
        x = np.zeros((len(trials), len(selectFolds)))
        for tInd, trial in enumerate(trials):
            for fInd, fold in enumerate(selectFolds):
                newPathV1 = path[0].replace("trial_0", "trial_"+str(trial)).replace("fold_0", "fold_"+str(fold)).replace("TestC", "Val2C")
                newPathV2 = path[1].replace("trial_0", "trial_"+str(trial)).replace("fold_0", "fold_"+str(fold)).replace("TestC", "Val2C")

                if "MEF" in names[i]:
                    baeIndex = newPath.find("BAE")
                    try:
                        c1 = maxEntInf(newPathV1[:baeIndex]+"pre_")
                        c2 = maxEntInf(newPathV2[:baeIndex]+"pre_")
                    except:
                        print("NP: trial: "+str(trial)+", fold: "+str(fold))
                        print(newPathV1[:baeIndex]+"pre_") #pass
                        print(newPathV2[:baeIndex]+"pre_") #pass
                else:
                    try:
                        c1 = np.load(newPathV1)
                        c2 = np.load(newPathV2)
                        #passed=True
                    except:
                        print("NP: trial: "+str(trial)+", fold: "+str(fold))
                        print(newPathV1) #pass
                        print(newPathV2) #pass
                #prefer the smaller validation set
                if c1 <= c2:
                    newPath = path[0].replace("trial_0", "trial_"+str(trial)).replace("fold_0", "fold_"+str(fold))
                else:
                    newPath = path[1].replace("trial_0", "trial_"+str(trial)).replace("fold_0", "fold_"+str(fold))
                #now finally set it
                try:
                    x[tInd][fInd]= np.load(newPath)
                except:
                    print("NP: trial: "+str(trial)+", fold: "+str(fold))
                    print(newPath) #pass


        means[names[i]] = np.mean(x, axis=0)
        stds[names[i]] = np.std(x, axis=0)
    return (means, stds)

def readMultiple(dataName,namesJoel):
    dataY = readData( JoelFolder+"Outputs/ScalableClassification/"+dataName+"Orig/baeloss_mean.txt")
    dataStd = readData(JoelFolder+"Outputs/ScalableClassification/"+dataName+"Orig/baeloss_stddev.txt")
    dataYnv = readData( JoelFolder+"Outputs/ScalableClassification/"+dataName+"N2V/baeloss_mean.txt")
    dataStdnv = readData(JoelFolder+"Outputs/ScalableClassification/"+dataName+"N2V/baeloss_stddev.txt")
    dataYN2V={}
    dataStdN2V = {}
    #add keys with N2V postfix
    for key in dataYnv:
        dataYN2V[key+"+N2V"] = dataYnv[key]
    for key in dataStdnv:
        dataStdN2V[key+"+N2V"] = dataStdnv[key]
    dataY.update(dataYN2V)
    dataStd.update(dataStdN2V)

    #only include provided models to plot
    if namesJoel!=None:
        dataY2 = {}
        dataStd2 = {}
        for key in namesJoel:
            dataY2[key] = dataY[key]
            dataStd2[key] = dataStd[key]
        dataY2['trial'] = dataY['trial']
        dataStd2['trial'] = dataStd['trial']
        dataY = dataY2
        dataStd = dataStd2
    return (dataY, dataStd)

def readNPAll(names, fileNames, trials=range(0, 10)):
    #separate single vs double comparison based names
    fNames = [] ; fNames2 = [] ;
    origNames = [] ; origNames2 = [] ;
    for i, fName in enumerate(fileNames):
        if len(fName)==1:
            fNames.append(fName[0])
            origNames.append(names[i])
        else:
            fNames2.append(fName)
            origNames2.append(names[i])
    rnnMeans, rnnStds = readNP(origNames, fNames, trials)

    #now read and pick one with better validation set
    rnnMeansBest, rnnStdsBest = readBest(origNames2, fNames2, trials)
    rnnMeans.update(rnnMeansBest)
    rnnStds.update(rnnStdsBest)
    return (rnnMeans, rnnStds)

def plotOrig(dataName, names, fileNames, namesJoel=['trial'], JoelType="Orig", trials = range(0, 10), outPostFix=""):
    dataY, dataStd = readMultiple(dataName, namesJoel) 
    rnnMeans, rnnStds = readNPAll(names, fileNames, trials)

    plotAcc(dataY, dataStd, "OutputPlots/"+dataName+outPostFix, rnnMeans=rnnMeans, rnnStds=rnnStds, trials=trials)

#if namesMe is a list, then output all comparisons
#if namesMe is a list of tuples (pairs), then output comparisons between each of the tuples
def BAEsPairwise(dataName, names, fileNames, namesJoel=['trial'], namesMe = [], trials = 10):
    dataY, dataStd = readMultiple(dataName, namesJoel)

    #only include provided models to plot
    if namesJoel!=None:
        dataY2 = {}
        dataStd2 = {}
        for key in namesJoel:
            dataY2[key] = dataY[key]
            dataStd2[key] = dataStd[key]
        dataY = dataY2
        dataStd = dataStd2

    rnnMeans, rnnStds = readNPAll(names, fileNames, trials)

    if len(namesMe)!=0 and len(namesMe[0])==1:
        rnnMeansWithKeys = {} ; rnnStdsWithKeys = {} ;
        for key in namesMe:
            rnnMeansWithKeys[key] = rnnMeans[key]
            rnnStdsWithKeys[key] = rnnStds[key]


    print("Data: "+dataName)
    if len(namesMe)!=0 and len(namesMe[0])==2:
        for keys in namesMe:
            baeGain(rnnMeans[keys[0]], rnnMeans[keys[1]], keys[0], keys[1])


    for rnnKey in rnnMeans:
        for joelKey in dataY:
            if joelKey!='trial':
                baeGain(dataY[joelKey], rnnMeans[rnnKey], joelKey, rnnKey)

        if len(namesMe)!=0 and len(namesMe[0])==1:
            for rnnKey2 in rnnMeansWithKeys:
                if rnnKey!=rnnKey2:
                    baeGain(rnnMeans[rnnKey2], rnnMeans[rnnKey], rnnKey2, rnnKey)



def baeGain(worse, better, worseKey, betterKey):
    baeGain=[]
    for i in range(0, len(better)):
        baeGain.append(float((worse[i] - better[i]))/worse[i])
    baeGainStr = ','.join([str(j) for j in baeGain])
    print(betterKey+"--"+worseKey+" : "+baeGainStr)   


#DCI-S, DCI, and DRI, DCI-R, DCI-A,
#DCI-D, and DCI-10
def main2(dataSet, JoelType="Orig"):
    trials = 10
    prefix = "tests/modelsComb/"
    rinit=["F"]
    orig = "w_no0_LSTM_aug_none_"+dataSet+"_noVl_F_Mem_10_Ex_1_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_F_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_sd_3_rinit_F_trial_0_fold_0_BAE_TestC.npy"
    aug = orig 
    pprNames = ["for", "back", "F"]
    #pprNames=["F"]
    #names = ["none", "attr", "attrdown",  "attrup",  "swap"]
    #names = ["attr", "swap", "attrDouble", "swapDouble"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = [ "swap", "swapDouble", "swapTriple", "attr", "attrDouble", "attrTriple"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = ["attr", "attrDouble", "attrTriple", "attrQuad"]
    #namesJoel = ["PL-EM (CAL)", 'CL-EM (10)', 'PL-EM', 'RLR', 'trial']
    #namesJoel = ["PL-EM (CAL)",'PL-EM', 'LR', 'LP', 'trial']
    namesJoel = ['trial']
    filePaths = []
    for rin in rinit:
        for name in pprNames:
            fPath = prefix+aug.replace("PPR_F", "PPR_"+name)
            filePaths.append(fPath)

    last=["/scratch/snyder/m/moore269/GraphDeepLearning3/GraphDeepLearning/pylearn2/experiments/tests/models/"+aug.replace("aug_none", "aug_swapdown"), prefix+aug.replace("TestC", "Test").replace("mNPro_100", "mNPro_0"), prefix+aug.replace("rinit_F", "rinit_T"), "tests/models/"+aug]
    filePaths = filePaths+last
    #names2 = ["GRNN-C", "GRNN-C-Swap", "GRNN-C-R", "GRNN-C-Swap-R"]
    names2 = ["DCI-A", "DCI-D", "DCI", "DCI-S", "DRI", "DCI-R", "DCI-10"]
    #filePaths=filePaths[2:]+filePaths[0:2]
    #names2 = names2[2:]+names2[0:2]
    plotOrig(dataSet, names2, filePaths, JoelType=JoelType, namesJoel=namesJoel, trials=trials)
    BAEsPairwise(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials)

#PL-EM, PL-EM-M, LR, LP
#DCI-S, DCI
def main1(dataSet, JoelType="Orig"):
    trials = 10
    prefix = "/scratch/snyder/m/moore269/GraphDeepLearning3/GraphDeepLearning/pylearn2/experiments/tests/models/"
    rinit=["F"]
    orig = "w_no0_LSTM_aug_none_"+dataSet+"_noVl_F_Mem_10_Ex_1_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_F_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_sd_3_rinit_F_trial_0_fold_0_BAE_TestC.npy"
    aug = orig 
    #pprNames=["F"]
    names = ["none", "swapdown"]
    #names = ["attr", "swap", "attrDouble", "swapDouble"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = [ "swap", "swapDouble", "swapTriple", "attr", "attrDouble", "attrTriple"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = ["attr", "attrDouble", "attrTriple", "attrQuad"]
    #namesJoel = ["PL-EM (CAL)", 'CL-EM (10)', 'PL-EM', 'RLR', 'trial']
    namesJoel = ["PL-EM (CAL)",'PL-EM', 'LR', 'LP', 'trial']
    #namesJoel = ['trial']
    filePaths = []
    for rin in rinit:
        for name in names:
            fPath = prefix+aug.replace("aug_none", "aug_"+name)
            filePaths.append(fPath)

    names2 = ["DCI", "DCI-S"]
    plotOrig(dataSet, names2, filePaths, JoelType=JoelType, namesJoel=namesJoel, trials=trials)
    BAEsPairwise(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials)



#PL-EM, PL-EM-M
#DCI-10, DCI-S-10
def mainPatents(dataSet, JoelType="Orig"):
    trials = 5
    prefix = "models/"
    rinit=["F"]
    orig = "w_no0_LSTM_aug_none_"+dataSet+"_noVl_F_Mem_10_Ex_1_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_F_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_sd_3_rinit_F_trial_0_fold_0_BAE_TestC.npy"
    aug = orig 
    names = ["none", "swap"]
    #names = ["none", "attr", "attrdown",  "attrup",  "swap"]
    #names = ["attr", "swap", "attrDouble", "swapDouble"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = [ "swap", "swapDouble", "swapTriple", "attr", "attrDouble", "attrTriple"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = ["attr", "attrDouble", "attrTriple", "attrQuad"]
    #namesJoel = ["PL-EM (CAL)", 'CL-EM (10)', 'PL-EM', 'RLR', 'trial']
    namesJoel = ["PL-EM (CAL)",'PL-EM', 'trial']
    filePaths = []
    for rin in rinit:
        for name in names:
            fPath = prefix+aug.replace("aug_none", "aug_"+name).replace("rinit_F", "rinit_"+rin)
            filePaths.append(fPath)

    names2 = ["DCI-10", "DCI-S-10"]
    plotOrig(dataSet, names2, filePaths, JoelType=JoelType, namesJoel=namesJoel, trials=trials)
    BAEsPairwise(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials)


#DCI-S, DCI, and DRI, DCI-R, DCI-A,
#DCI-D, and DCI-10
def compareMini(dataSet, JoelType="Orig"):
    trials = 10
    prefix = nodeType+"/GraphDeepLearning2/GraphDeepLearning/pylearn2/experiments/tests/modelsComb/"
    prefix10 = nodeType+"/GraphDeepLearning2/GraphDeepLearning/pylearn2/experiments/tests/models/"
    prefixSwap = nodeType+"/GraphDeepLearning3/GraphDeepLearning/pylearn2/experiments/tests/models/"
    prefixMini = nodeType + "/GraphDeepLearningMini/GraphDeepLearning/dri/experiments/models/"

    rinit=["F"]
    orig = "w_no0_LSTM_aug_none_"+dataSet+"_noVl_F_Mem_10_Ex_1_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_F_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_sd_3_rinit_F_trial_0_fold_0_BAE_TestC.npy"
    aug = orig 
    origMini = "w_no0_LSTMwMini_aug_none_"+dataSet+"_noVl_F_Mem_5_min_5_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_trial_0_fold_0_BAE_TestC.npy"
    #pprNames=["F"]
    #names = ["none", "attr", "attrdown",  "attrup",  "swap"]
    #names = ["attr", "swap", "attrDouble", "swapDouble"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = [ "swap", "swapDouble", "swapTriple", "attr", "attrDouble", "attrTriple"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = ["attr", "attrDouble", "attrTriple", "attrQuad"]
    #namesJoel = ["PL-EM (CAL)", 'CL-EM (10)', 'PL-EM', 'RLR', 'trial']
    #namesJoel = ["PL-EM (CAL)",'PL-EM', 'LR', 'LP', 'trial']
    namesJoel = ['trial']


    filePaths=[prefix+aug, prefixSwap+aug.replace("aug_none", "aug_swapdown"), prefix+aug.replace("TestC", "Test").replace("mNPro_100", "mNPro_0"), prefix10+aug,
    prefixMini+origMini, prefixMini+origMini.replace("TestC", "Test")]
    
    #names2 = ["GRNN-C", "GRNN-C-Swap", "GRNN-C-R", "GRNN-C-Swap-R"]
    names2 = ["DCI", "DCI-S", "DRI", "DCI-10", "DCI-mini", "DRI-mini"]
    #filePaths=filePaths[2:]+filePaths[0:2]
    #names2 = names2[2:]+names2[0:2]
    plotOrig(dataSet, names2, filePaths, JoelType=JoelType, namesJoel=namesJoel, trials=trials)
    BAEsPairwise(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials)

#mini variants
#DRI
def compareMinisDRI(dataSet, JoelType="Orig"):
    trials = 10
    prefix = nodeType+"/GraphDeepLearning2/GraphDeepLearning/pylearn2/experiments/tests/modelsComb/"
    prefixSwap = nodeType+"/GraphDeepLearning3/GraphDeepLearning/pylearn2/experiments/tests/models/"
    prefixMini = nodeType + "/GraphDeepLearningMini/GraphDeepLearning/dri/experiments/models/"

    rinit=["F"]
    orig = "w_no0_LSTM_aug_none_"+dataSet+"_noVl_F_Mem_10_Ex_1_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_F_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_sd_3_rinit_F_trial_0_fold_0_BAE_TestC.npy"
    aug = orig 
    origMini = "w_no0_LSTMwMini_aug_none_"+dataSet+"_noVl_F_Mem_5_min_5_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_trial_0_fold_0_BAE_TestC.npy"

    origMini2 = "w_no0_LSTMwMini_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_5_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_trial_0_fold_0_BAE_TestC.npy"
    #pprNames=["F"]
    #names = ["none", "attr", "attrdown",  "attrup",  "swap"]
    #names = ["attr", "swap", "attrDouble", "swapDouble"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = [ "swap", "swapDouble", "swapTriple", "attr", "attrDouble", "attrTriple"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = ["attr", "attrDouble", "attrTriple", "attrQuad"]
    #namesJoel = ["PL-EM (CAL)", 'CL-EM (10)', 'PL-EM', 'RLR', 'trial']
    #namesJoel = ["PL-EM (CAL)",'PL-EM', 'LR', 'LP', 'trial']
    namesJoel = ['trial']


    filePaths=[prefix+aug.replace("TestC", "Test").replace("mNPro_100", "mNPro_0"),
    prefixMini+origMini.replace("TestC", "Test"),
    prefixMini+origMini2.replace("TestC", "Test"),
    prefixMini+origMini2.replace("LSTMwMini", "LSTM2").replace("TestC", "Test"),
    prefixMini+origMini2.replace("p_F", "p_T").replace("TestC", "Test")
    ]
    
    #names2 = ["GRNN-C", "GRNN-C-Swap", "GRNN-C-R", "GRNN-C-Swap-R"]
    names2 = ["DRI", "DRI-m",
    "DRI-m-1000",
    "DRI-2",
    "DRI-m-p-1000"
    ]
    #filePaths=filePaths[2:]+filePaths[0:2]
    #names2 = names2[2:]+names2[0:2]
    plotOrig(dataSet, names2, filePaths, JoelType=JoelType, namesJoel=namesJoel, trials=trials, outPostFix="DRI")
    BAEsPairwise(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials)

#mini variants
#DCI
def compareMinisDCI(dataSet, JoelType="Orig"):
    trials = 10
    prefix = nodeType+"/GraphDeepLearning2/GraphDeepLearning/pylearn2/experiments/tests/modelsComb/"
    prefixSwap = nodeType+"/GraphDeepLearning3/GraphDeepLearning/pylearn2/experiments/tests/models/"
    prefixMini = nodeType + "/GraphDeepLearningMini/GraphDeepLearning/dri/experiments/models/"

    rinit=["F"]
    orig = "w_no0_LSTM_aug_none_"+dataSet+"_noVl_F_Mem_10_Ex_1_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_F_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_sd_3_rinit_F_trial_0_fold_0_BAE_TestC.npy"
    aug = orig 
    origMini = "w_no0_LSTMwMini_aug_none_"+dataSet+"_noVl_F_Mem_5_min_5_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_trial_0_fold_0_BAE_TestC.npy"

    origMini2 = "w_no0_LSTMwMini_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_5_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_trial_0_fold_0_BAE_TestC.npy"
    #pprNames=["F"]
    #names = ["none", "attr", "attrdown",  "attrup",  "swap"]
    #names = ["attr", "swap", "attrDouble", "swapDouble"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = [ "swap", "swapDouble", "swapTriple", "attr", "attrDouble", "attrTriple"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = ["attr", "attrDouble", "attrTriple", "attrQuad"]
    #namesJoel = ["PL-EM (CAL)", 'CL-EM (10)', 'PL-EM', 'RLR', 'trial']
    #namesJoel = ["PL-EM (CAL)",'PL-EM', 'LR', 'LP', 'trial']
    namesJoel = ['trial']


    filePaths=[prefix+aug, prefixSwap+aug.replace("aug_none", "aug_swapdown"), 
    prefixMini+origMini, 
    prefixMini+origMini2, 
    prefixMini+origMini2.replace("LSTMwMini", "LSTM2"), 
    prefixMini+origMini2.replace("p_F", "p_T")
    ]
    
    #names2 = ["GRNN-C", "GRNN-C-Swap", "GRNN-C-R", "GRNN-C-Swap-R"]
    names2 = ["DCI", "DCI-S", "DCI-m",
    "DCI-m-1000", 
    "DCI-2", 
    "DCI-m-p-1000"
    ]
    #filePaths=filePaths[2:]+filePaths[0:2]
    #names2 = names2[2:]+names2[0:2]
    plotOrig(dataSet, names2, filePaths, JoelType=JoelType, namesJoel=namesJoel, trials=trials, outPostFix="DCI")
    BAEsPairwise(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials)

    compareMinisDRI(dataSet)


#mini variants
#DRI
def compareMinisDimDRI(dataSet, JoelType="Orig"):
    trials = 10
    prefix = nodeType+"/GraphDeepLearning2/GraphDeepLearning/pylearn2/experiments/tests/modelsComb/"
    prefixSwap = nodeType+"/GraphDeepLearning3/GraphDeepLearning/pylearn2/experiments/tests/models/"
    prefixMini = nodeType + "/GraphDeepLearningMini/GraphDeepLearning/dri/experiments/models/"

    rinit=["F"]
    orig = "w_no0_LSTM_aug_none_"+dataSet+"_noVl_F_Mem_10_Ex_1_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_F_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_sd_3_rinit_F_trial_0_fold_0_BAE_TestC.npy"
    aug = orig 
    origMini = "w_no0_LSTMwMini_aug_none_"+dataSet+"_noVl_F_Mem_5_min_5_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_trial_0_fold_0_BAE_TestC.npy"

    origMini2 = "w_no0_LSTMwMini_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_5_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_trial_0_fold_0_BAE_TestC.npy"
    #pprNames=["F"]
    #names = ["none", "attr", "attrdown",  "attrup",  "swap"]
    #names = ["attr", "swap", "attrDouble", "swapDouble"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = [ "swap", "swapDouble", "swapTriple", "attr", "attrDouble", "attrTriple"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = ["attr", "attrDouble", "attrTriple", "attrQuad"]
    #namesJoel = ["PL-EM (CAL)", 'CL-EM (10)', 'PL-EM', 'RLR', 'trial']
    #namesJoel = ["PL-EM (CAL)",'PL-EM', 'LR', 'LP', 'trial']
    namesJoel = ['trial']


    filePaths=[prefix+aug.replace("TestC", "Test").replace("mNPro_100", "mNPro_0"),
    prefixMini+origMini.replace("TestC", "Test"),
    prefixMini+origMini2.replace("TestC", "Test"),
    prefixMini+origMini2.replace("TestC", "Test").replace("Mem_5","Mem_10"), 
    prefixMini+origMini2.replace("TestC", "Test").replace("Mem_5","Mem_20"), 
    prefixMini+origMini2.replace("LSTMwMini", "LSTM2").replace("TestC", "Test"),
    prefixMini+origMini2.replace("p_F", "p_T").replace("TestC", "Test")
    ]
    
    #names2 = ["GRNN-C", "GRNN-C-Swap", "GRNN-C-R", "GRNN-C-Swap-R"]
    names2 = ["DRI", "DRI-m",
    "DRI-m-1000",
    "DRI-m-1000-d-10",
    "DRI-m-1000-d-20",
    "DRI-2",
    "DRI-m-p-1000"
    ]
    #filePaths=filePaths[2:]+filePaths[0:2]
    #names2 = names2[2:]+names2[0:2]
    plotOrig(dataSet, names2, filePaths, JoelType=JoelType, namesJoel=namesJoel, trials=trials, outPostFix="DRI")
    BAEsPairwise(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials)



#mini variants
#DCI
def compareMinisDimDCI(dataSet, JoelType="Orig"):
    trials = 10
    prefix = nodeType+"/GraphDeepLearning2/GraphDeepLearning/pylearn2/experiments/tests/modelsComb/"
    prefixSwap = nodeType+"/GraphDeepLearning3/GraphDeepLearning/pylearn2/experiments/tests/models/"
    prefixMini = nodeType + "/GraphDeepLearningMini/GraphDeepLearning/dri/experiments/models/"

    rinit=["F"]
    orig = "w_no0_LSTM_aug_none_"+dataSet+"_noVl_F_Mem_10_Ex_1_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_F_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_sd_3_rinit_F_trial_0_fold_0_BAE_TestC.npy"
    aug = orig 
    origMini = "w_no0_LSTMwMini_aug_none_"+dataSet+"_noVl_F_Mem_5_min_5_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_trial_0_fold_0_BAE_TestC.npy"

    origMini2 = "w_no0_LSTMwMini_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_5_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_trial_0_fold_0_BAE_TestC.npy"
    #pprNames=["F"]
    #names = ["none", "attr", "attrdown",  "attrup",  "swap"]
    #names = ["attr", "swap", "attrDouble", "swapDouble"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = [ "swap", "swapDouble", "swapTriple", "attr", "attrDouble", "attrTriple"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = ["attr", "attrDouble", "attrTriple", "attrQuad"]
    #namesJoel = ["PL-EM (CAL)", 'CL-EM (10)', 'PL-EM', 'RLR', 'trial']
    #namesJoel = ["PL-EM (CAL)",'PL-EM', 'LR', 'LP', 'trial']
    namesJoel = ['trial']


    filePaths=[prefix+aug, prefixSwap+aug.replace("aug_none", "aug_swapdown"), 
    prefixMini+origMini, 
    prefixMini+origMini2, 
    prefixMini+origMini2.replace("Mem_5","Mem_10"), 
    prefixMini+origMini2.replace("Mem_5","Mem_20"), 
    prefixMini+origMini2.replace("LSTMwMini", "LSTM2"), 
    prefixMini+origMini2.replace("p_F", "p_T")
    ]
    
    #names2 = ["GRNN-C", "GRNN-C-Swap", "GRNN-C-R", "GRNN-C-Swap-R"]
    names2 = ["DCI", "DCI-S", "DCI-m",
    "DCI-m-1000", 
    "DCI-m-1000-d-10",
    "DCI-m-1000-d-20",
    "DCI-2", 
    "DCI-m-p-1000"
    ]
    #filePaths=filePaths[2:]+filePaths[0:2]
    #names2 = names2[2:]+names2[0:2]
    plotOrig(dataSet, names2, filePaths, JoelType=JoelType, namesJoel=namesJoel, trials=trials, outPostFix="DCI")
    BAEsPairwise(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials)

    compareMinisDimDRI(dataSet)


#DCI-S, DCI, and DRI, DCI-R, DCI-A,
#DCI-D, and DCI-10
def comparePro(dataSet, JoelType="Orig"):
    trials = 10
    prefix = nodeType+"/GraphDeepLearning2/GraphDeepLearning/pylearn2/experiments/tests/modelsComb/"
    prefixSwap = nodeType+"/GraphDeepLearning3/GraphDeepLearning/pylearn2/experiments/tests/models/"
    prefixMini = nodeType + "/GraphDeepLearningMini/GraphDeepLearning/dri/experiments/models/"

    rinit=["F"]
    orig = "w_no0_LSTM_aug_none_"+dataSet+"_noVl_F_Mem_10_Ex_1_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_F_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_sd_3_rinit_F_trial_0_fold_0_BAE_TestC.npy"
    aug = orig 
    origMini = "w_no0_LSTMwMini_aug_none_"+dataSet+"_noVl_F_Mem_10_min_5_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_trial_0_fold_0_BAE_TestC.npy"
    if "patents" in dataSet:
        trials = 5
        prefix = nodeType+"/GraphDeepLearning2/GraphDeepLearning/pylearn2/experiments/tests/models/"
        prefixSwap = prefix
        origMiniPro = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_10_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_T_lH_F_lR_F_trial_0_fold_0_BAE_TestC.npy"
    elif dataSet!="facebook" and dataSet!="IMDB_5" and "amazon_DVD" not in dataSet:
        origMiniPro = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_T_lH_F_lR_F_trial_0_fold_0_BAE_TestC.npy"
    else:
        origMiniPro = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_T_trial_0_fold_0_BAE_TestC.npy"
    origMiniW = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_T_trial_0_fold_0_BAE_TestC.npy"
    origMiniWR= "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_T_lR_T_trial_0_fold_0_BAE_TestC.npy"
    #pprNames=["F"]
    #names = ["none", "attr", "attrdown",  "attrup",  "swap"]
    #names = ["attr", "swap", "attrDouble", "swapDouble"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = [ "swap", "swapDouble", "swapTriple", "attr", "attrDouble", "attrTriple"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = ["attr", "attrDouble", "attrTriple", "attrQuad"]
    #namesJoel = ["PL-EM (CAL)", 'CL-EM (10)', 'PL-EM', 'RLR', 'trial']
    #namesJoel = ["PL-EM (CAL)",'PL-EM', 'LR', 'LP', 'trial']
    namesJoel = ['trial']


    filePaths=[prefix+aug, prefixSwap+aug.replace("aug_none", "aug_swapdown"), prefix+aug.replace("TestC", "Test").replace("mNPro_100", "mNPro_0"),
    prefixMini+origMiniPro, prefixMini+origMiniPro.replace("TestC", "Test"),
    prefixMini+origMiniW, prefixMini+origMiniWR]
    
    #names2 = ["GRNN-C", "GRNN-C-Swap", "GRNN-C-R", "GRNN-C-Swap-R"]
    names2 = ["DCI", "DCI-S", "DRI", "DCI-P", "DRI-P", "DCI-W", "DCI-WR"]
    #filePaths=filePaths[2:]+filePaths[0:2]
    #names2 = names2[2:]+names2[0:2]
    plotOrig(dataSet, names2, filePaths, JoelType=JoelType, namesJoel=namesJoel, trials=trials)
    #BAEsPairwise(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials)

def compareMaxEntInf(dataSet, JoelType="Orig"):
    trials = 10
    prefix = nodeType+"/GraphDeepLearning2/GraphDeepLearning/pylearn2/experiments/tests/modelsComb/"
    prefixSwap = nodeType+"/GraphDeepLearning3/GraphDeepLearning/pylearn2/experiments/tests/models/"
    prefixMini = nodeType + "/GraphDeepLearningMini/GraphDeepLearning/dri/experiments/models/"

    rinit=["F"]
    orig = "w_no0_LSTM_aug_none_"+dataSet+"_noVl_F_Mem_10_Ex_1_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_F_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_sd_3_rinit_F_trial_0_fold_0_BAE_TestC.npy"
    aug = orig 
    origMini = "w_no0_LSTMwMini_aug_none_"+dataSet+"_noVl_F_Mem_10_min_5_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_trial_0_fold_0_BAE_TestC.npy"
    if "patents" in dataSet:
        trials = 5
        prefix = nodeType+"/GraphDeepLearning2/GraphDeepLearning/pylearn2/experiments/tests/models/"
        prefixSwap = prefix
        origMiniPro = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_10_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_T_lH_F_lR_F_trial_0_fold_0_BAE_TestC.npy"
    elif dataSet!="facebook" and dataSet!="IMDB_5" and "amazon_DVD" not in dataSet:
        origMiniPro = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_T_lH_F_lR_F_trial_0_fold_0_BAE_TestC.npy"
    else:
        origMiniPro = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_T_trial_0_fold_0_BAE_TestC.npy"
    origMiniW = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_T_trial_0_fold_0_BAE_TestC.npy"
    origMiniWR = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_T_lR_T_trial_0_fold_0_BAE_TestC.npy"
    #pprNames=["F"]
    #names = ["none", "attr", "attrdown",  "attrup",  "swap"]
    #names = ["attr", "swap", "attrDouble", "swapDouble"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = [ "swap", "swapDouble", "swapTriple", "attr", "attrDouble", "attrTriple"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = ["attr", "attrDouble", "attrTriple", "attrQuad"]
    #namesJoel = ["PL-EM (CAL)", 'CL-EM (10)', 'PL-EM', 'RLR', 'trial']
    #namesJoel = ["PL-EM (CAL)",'PL-EM', 'LR', 'LP', 'trial']
    namesJoel = ['trial']


    filePaths=[prefix+aug, prefix+aug, 
    prefixSwap+aug.replace("aug_none", "aug_swapdown"), prefixSwap+aug.replace("aug_none", "aug_swapdown"),
    prefixMini+origMiniPro, prefixMini+origMiniPro
    ]
    
    #names2 = ["GRNN-C", "GRNN-C-Swap", "GRNN-C-R", "GRNN-C-Swap-R"]
    names2 = ["DCI", "DCI-MEF", "DCI-S", "DCI-S-MEF", "DCI-P", "DCI-P-MEF"]
    #filePaths=filePaths[2:]+filePaths[0:2]
    #names2 = names2[2:]+names2[0:2]
    plotOrig(dataSet, names2, filePaths, JoelType=JoelType, namesJoel=namesJoel, trials=trials, outPostFix="MEF")
    #BAEsPairwise(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials)   

def compareACCBAEHogun(dataSet, JoelType=None):
    trials = 10
    prefix = nodeType + "/GraphDeepLearningMini/GraphDeepLearning/dri/experiments/models/"

    rinit=["F"]
    origMini = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_trial_0_fold_6_ACC_TestC.npy"
    origMiniBAE = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_trial_0_fold_6_BAE_TestC.npy"
    #pprNames=["F"]
    #names = ["none", "attr", "attrdown",  "attrup",  "swap"]
    #names = ["attr", "swap", "attrDouble", "swapDouble"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = [ "swap", "swapDouble", "swapTriple", "attr", "attrDouble", "attrTriple"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = ["attr", "attrDouble", "attrTriple", "attrQuad"]
    #namesJoel = ["PL-EM (CAL)", 'CL-EM (10)', 'PL-EM', 'RLR', 'trial']
    #namesJoel = ["PL-EM (CAL)",'PL-EM', 'LR', 'LP', 'trial']
    namesJoel = None

    filePaths=[prefix+origMini, prefix+origMiniBAE,
    prefix+origMini.replace("aug_none", "aug_swapdown"), prefix+origMiniBAE.replace("aug_none", "aug_swapdown")]
    
    #names2 = ["GRNN-C", "GRNN-C-Swap", "GRNN-C-R", "GRNN-C-Swap-R"]
    names2 = ["ACC", "BAE", "ACC_Swap", "BAE_Swap"]
    #filePaths=filePaths[2:]+filePaths[0:2]
    #names2 = names2[2:]+names2[0:2]
    plotOrig(dataSet, names2, filePaths, JoelType=JoelType, namesJoel=namesJoel, trials=trials)

#DCI-S, DCI, and DRI, DCI-R, DCI-A,
#DCI-D, and DCI-10
def compareDRIBAG(dataSet, JoelType="Orig"):
    trials = range(0, 10)
    prefix = nodeType+"/GraphDeepLearning2/GraphDeepLearning/pylearn2/experiments/tests/modelsComb/"
    prefixSwap = nodeType+"/GraphDeepLearning3/GraphDeepLearning/pylearn2/experiments/tests/models/"
    prefixMini = nodeType + "/GraphDeepLearningMini/GraphDeepLearning/dri/experiments/models/"

    rinit=["F"]
    orig = "w_no0_LSTM_aug_none_"+dataSet+"_noVl_F_Mem_10_Ex_1_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_F_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_sd_3_rinit_F_trial_0_fold_0_BAE_TestC.npy"
    aug = orig 
    origMini = "w_no0_LSTMwMini_aug_none_"+dataSet+"_noVl_F_Mem_10_min_5_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_trial_0_fold_0_BAE_TestC.npy"
    if "patents" in dataSet:
        trials = range(0, 5)
        prefix = nodeType+"/GraphDeepLearning2/GraphDeepLearning/pylearn2/experiments/tests/models/"
        prefixSwap = prefix
        origMiniPro = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_10_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_T_lH_F_lR_F_trial_0_fold_0_BAE_TestC.npy"
    elif dataSet!="facebook" and dataSet!="IMDB_5" and "amazon_DVD" not in dataSet:
        origMiniPro = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_T_lH_F_lR_F_trial_0_fold_0_BAE_TestC.npy"
    else:
        origMiniPro = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_T_trial_0_fold_0_BAE_TestC.npy"
    origMiniW = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_T_trial_0_fold_0_BAE_TestC.npy"
    origMiniWR= "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_T_lR_T_trial_0_fold_0_BAE_TestC.npy"
    origMiniLRAVG = "w_no0_LRAVG_aug_none_"+dataSet+"_mNe_10_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_0_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_trial_0_fold_0_BAE_Test.npy"
    #netTypes=["LRAVG", "LRAVGaverage", "LRAVGaverageOrig"]
    #pprNames=["F"]
    #names = ["none", "attr", "attrdown",  "attrup",  "swap"]
    #names = ["attr", "swap", "attrDouble", "swapDouble"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = [ "swap", "swapDouble", "swapTriple", "attr", "attrDouble", "attrTriple"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = ["attr", "attrDouble", "attrTriple", "attrQuad"]
    #namesJoel = ["PL-EM (CAL)", 'CL-EM (10)', 'PL-EM', 'RLR', 'trial']
    #namesJoel = ["PL-EM (CAL)",'PL-EM', 'LR', 'LP', 'trial']
    namesJoel = ['trial']


    filePaths=[[prefix+aug.replace("TestC", "Test").replace("mNPro_100", "mNPro_0")], 
    [prefixMini+origMiniPro.replace("TestC", "Test")],
    [prefixMini+origMiniLRAVG],
    [prefixMini+origMiniLRAVG.replace("LRAVG", "LRAVGaverage")],
    [prefixMini+origMiniLRAVG.replace("LRAVG", "LRAVGaverageOrig")]
    ]
    
    #names2 = ["GRNN-C", "GRNN-C-Swap", "GRNN-C-R", "GRNN-C-Swap-R"]
    names2 = ["DRI", "DRI-P", "LR", "LRAVG All", "LRAVG + Orig"]
    #filePaths=filePaths[2:]+filePaths[0:2]
    #names2 = names2[2:]+names2[0:2]
    plotOrig(dataSet, names2, filePaths, JoelType=JoelType, namesJoel=namesJoel, trials=trials, outPostFix="LRAVG")
    #BAEsPairwise(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials)

def mainN2V(dataSet, JohnType="", dataType="TestC", extraCTV="5"):
    trials = range(0, 10)

    prefixRNCC = nodeType + "/GraphDeepLearningMiniRNCC/GraphDeepLearning/dri/experiments/models/"
    origRNCC= "w_no0_LSTM2_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_1_mNPro_200_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_T_lR_F_CTV_3_trial_0_fold_0_BAE_"+dataType+".npy"
    prefixMini = nodeType + "/GraphDeepLearningMini/GraphDeepLearning/dri/experiments/models/"
    prefixMini10 = nodeType + "/GraphDeepLearningMini10/GraphDeepLearning/dri/experiments/models/"
    namesJoel = ["PL-EM (CAL)", 'LR', 'LP', "PL-EM (CAL)+N2V", "LR+N2V"]
    names2 = ["DCI", "RNCC"]
    if "patents" in dataSet:
        trials = range(0, 10)
        origRNCC=origRNCC.replace("_mNe_10000", "_mNe_10")
        namesJoel = ["PL-EM (CAL)", "PL-EM (CAL)+N2V"]
        prefixMini=prefixMini10
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_10_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_"+extraCTV+"_trial_0_fold_0_BAE_"+dataType+".npy"
    elif dataSet!="facebook" and dataSet!="IMDB_5" and "amazon_DVD" not in dataSet:
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_"+extraCTV+"_trial_0_fold_0_BAE_"+dataType+".npy"
    else:
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_"+extraCTV+"_trial_0_fold_0_BAE_"+dataType+".npy"


    filePaths = [[prefixMini+origMiniCTV.replace("aug_none", "aug_swapdown"), 
    prefixMini+origMiniCTV.replace("pro_F", "pro_T")],
    [prefixRNCC+origRNCC]]
    plotOrig(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials, outPostFix=dataType)
    BAEsPairwise(dataSet, names2, filePaths, namesJoel=namesJoel, namesMe=[("RNCC", "DCI")], trials=trials)

def compareCTV(dataSet, JoelType="Orig"):
    trials = range(0, 10)
    prefix = nodeType+"/GraphDeepLearning2/GraphDeepLearning/pylearn2/experiments/tests/modelsComb/"
    prefixSwap = nodeType+"/GraphDeepLearning3/GraphDeepLearning/pylearn2/experiments/tests/models/"
    prefixMini = nodeType + "/GraphDeepLearningMini/GraphDeepLearning/dri/experiments/models/"

    rinit=["F"]
    orig = "w_no0_LSTM_aug_none_"+dataSet+"_noVl_F_Mem_10_Ex_1_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_F_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_sd_3_rinit_F_trial_0_fold_0_BAE_TestC.npy"
    aug = orig 
    origMini = "w_no0_LSTMwMini_aug_none_"+dataSet+"_noVl_F_Mem_10_min_5_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_trial_0_fold_0_BAE_TestC.npy"

    if "patents" in dataSet:
        prefix = nodeType+"/GraphDeepLearning2/GraphDeepLearning/pylearn2/experiments/tests/models/"
        prefixSwap = prefix
        origMiniPro = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_10_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_T_lH_F_lR_F_trial_0_fold_0_BAE_TestC.npy"
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_0_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_0_trial_0_fold_0_BAE_TestC.npy"
    elif dataSet!="facebook" and dataSet!="IMDB_5" and "amazon_DVD" not in dataSet:
        origMiniPro = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_T_lH_F_lR_F_trial_0_fold_0_BAE_TestC.npy"
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_0_trial_0_fold_0_BAE_TestC.npy"
    else:
        origMiniPro = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_T_trial_0_fold_0_BAE_TestC.npy"
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_0_trial_0_fold_0_BAE_TestC.npy"
    origMiniW = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_T_trial_0_fold_0_BAE_TestC.npy"
    origMiniWR= "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_T_lR_T_trial_0_fold_0_BAE_TestC.npy"

    #netTypes=["LRAVG", "LRAVGaverage", "LRAVGaverageOrig"]
    #pprNames=["F"]
    #names = ["none", "attr", "attrdown",  "attrup",  "swap"]
    #names = ["attr", "swap", "attrDouble", "swapDouble"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = [ "swap", "swapDouble", "swapTriple", "attr", "attrDouble", "attrTriple"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = ["attr", "attrDouble", "attrTriple", "attrQuad"]
    #namesJoel = ["PL-EM (CAL)", 'CL-EM (10)', 'PL-EM', 'RLR', 'trial']
    #namesJoel = ["PL-EM (CAL)",'PL-EM', 'LR', 'LP', 'trial']
    namesJoel = ['trial']


    filePaths=[[prefix+aug], 
    [prefixSwap+aug.replace("aug_none", "aug_swapdown")], 
    [prefixMini+origMiniCTV.replace("CTV_0", "CTV_2")],
    [prefixMini+origMiniCTV.replace("CTV_0", "CTV_1")],
    [prefixMini+origMiniCTV.replace("CTV_0", "CTV_2").replace("aug_none", "aug_swapdown")],
    [prefixMini+origMiniCTV.replace("CTV_0", "CTV_1").replace("aug_none", "aug_swapdown")]
    ]
    
    #names2 = ["GRNN-C", "GRNN-C-Swap", "GRNN-C-R", "GRNN-C-Swap-R"]
    names2 = ["DCI-0", "DCI-S-0", "DCI-2", "DCI-1", "DCI-S-2", "DCI-S-1"]
    #filePaths=filePaths[2:]+filePaths[0:2]
    #names2 = names2[2:]+names2[0:2]
    plotOrig(dataSet, names2, filePaths, JoelType=JoelType, namesJoel=namesJoel, trials=trials, outPostFix="CTV")


#DCI-S, DCI, and DRI, DCI-R, DCI-A,
#DCI-D, and DCI-10
def DCIVariants(dataSet, JoelType="Orig", extraCTV="3"):

    trials = range(0, 10)
    prefixMini = nodeType + "/GraphDeepLearningMini/GraphDeepLearning/dri/experiments/models/"
    prefixMini10 = nodeType + "/GraphDeepLearningMini10_2/GraphDeepLearning/dri/experiments/models/"

    if "patents" in dataSet:
        prefixMini=nodeType + "/GraphDeepLearningMini10/GraphDeepLearning/dri/experiments/models/"
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_10_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_"+extraCTV+"_trial_0_fold_0_BAE_TestC.npy"
    elif dataSet!="facebook" and dataSet!="IMDB_5" and "amazon_DVD" not in dataSet:
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_"+extraCTV+"_trial_0_fold_0_BAE_TestC.npy"
    else:
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_"+extraCTV+"_trial_0_fold_0_BAE_TestC.npy"


    filePaths = [[prefixMini+origMiniCTV],
    [prefixMini+origMiniCTV.replace("CTV_"+extraCTV, "CTV_5")],
    [prefixMini+origMiniCTV.replace("PPR_0", "PPR_for")],
    [prefixMini+origMiniCTV.replace("PPR_0", "PPR_back")],
    [prefixMini+origMiniCTV.replace("BAE_TestC", "BAE_Test")],
    [prefixMini10+origMiniCTV.replace("mNe_10000", "mNe_10")],
    [prefixMini+origMiniCTV.replace("CTV_"+extraCTV, "CTV_2")]
    ]
    #names2 = ["GRNN-C", "GRNN-C-Swap", "GRNN-C-R", "GRNN-C-Swap-R"]
    names2 = ["DCI-WSB", "DCI", "DCI-A", "DCI-D", "DRI-WSB", "DCI-WSB-10", "DCI-ST"]
    #filePaths=filePaths[2:]+filePaths[0:2]
    #names2 = names2[2:]+names2[0:2]
    plotOrig(dataSet, names2, filePaths, JoelType=JoelType, trials=trials)
    BAEsPairwise(dataSet, names2, filePaths, namesMe=[("DCI-ST", "DCI-WSB")], trials=trials)

def DCIVariantsRandom(dataSet, JoelType="Orig", extraCTV="3"):

    trials = range(0, 10)
    prefixMini = nodeType + "/GraphDeepLearningMini/GraphDeepLearning/dri/experiments/models/"
    prefixMini10 = nodeType + "/GraphDeepLearningMini10_2/GraphDeepLearning/dri/experiments/models/"

    if "patents" in dataSet:
        prefixMini=nodeType + "/GraphDeepLearningMini10/GraphDeepLearning/dri/experiments/models/"
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_10_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_"+extraCTV+"_trial_0_fold_0_BAE_Test.npy"
    elif dataSet!="facebook" and dataSet!="IMDB_5" and "amazon_DVD" not in dataSet:
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_"+extraCTV+"_trial_0_fold_0_BAE_TestC.npy"
    else:
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_"+extraCTV+"_trial_0_fold_0_BAE_TestC.npy"

    filePaths = [[prefixMini+origMiniCTV],
    [prefixMini+origMiniCTV.replace("rinit_F", "rinit_T")],
    [prefixMini+origMiniCTV.replace("aug_none", "aug_swapdown")],
    [prefixMini+origMiniCTV.replace("aug_none", "aug_swapdown").replace("rinit_F", "rinit_T")],
    [prefixMini+origMiniCTV.replace("pro_F", "pro_T")],
    [prefixMini+origMiniCTV.replace("pro_F", "pro_T").replace("rinit_F", "rinit_T")]
    ]
    #names2 = ["GRNN-C", "GRNN-C-Swap", "GRNN-C-R", "GRNN-C-Swap-R"]
    names2 = ["DCI-WSB", "DCI-WSB-R", "DCI-S", "DCI-S-R", "DCI-B", "DCI-B-R"]
    #filePaths=filePaths[2:]+filePaths[0:2]
    #names2 = names2[2:]+names2[0:2]
    plotOrig(dataSet, names2, filePaths, JoelType=JoelType, trials=trials, outPostFix="rinit")
    BAEsPairwise(dataSet, names2, filePaths, namesMe=[("DCI-R", "DCI-WSB"), ("DCI-S-R", "DCI-S"), ("DCI-B-R", "DCI-B")], trials=trials)


def DCIvalidPreds(dataSet, JohnType="", dataType="TestC", extraCTV=1):
    extraCTV=str(extraCTV)
    trials = range(0, 10)
    prefixMini = nodeType + "/GraphDeepLearningMini/GraphDeepLearning/dri/experiments/models/"
    prefixMini10 = nodeType + "/GraphDeepLearningMini10_2/GraphDeepLearning/dri/experiments/models/"
    namesJoel = []

    if "patents" in dataSet:
        prefixMini=nodeType + "/GraphDeepLearningMini10/GraphDeepLearning/dri/experiments/models/"
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_10_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_3_trial_0_fold_0_BAE_"+dataType+".npy"
    elif dataSet!="facebook" and dataSet!="IMDB_5" and "amazon_DVD" not in dataSet:
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_3_trial_0_fold_0_BAE_"+dataType+".npy"
    else:
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_3_trial_0_fold_0_BAE_"+dataType+".npy"

    if "Val" in dataType:
        names2 = ["DCI-CTV-"+extraCTV, "DCI-S-CTV-"+extraCTV, "DCI-B-CTV-"+extraCTV]
        filePaths = [[prefixMini+origMiniCTV.replace("CTV_3", "CTV_"+extraCTV)],
        [prefixMini+origMiniCTV.replace("CTV_3", "CTV_"+extraCTV).replace("aug_none", "aug_swapdown")],
        [prefixMini+origMiniCTV.replace("CTV_3", "CTV_"+extraCTV).replace("pro_F", "pro_T")]
        ]
    else:
        names2 = ["DCI", "DCI-CTV-"+extraCTV, "DCI-S", "DCI-S-CTV-"+extraCTV, "DCI-B", "DCI-B-CTV-"+extraCTV]
        filePaths = [[prefixMini+origMiniCTV], 
        [prefixMini+origMiniCTV.replace("CTV_3", "CTV_"+extraCTV)],
        [prefixMini+origMiniCTV.replace("aug_none", "aug_swapdown")], 
        [prefixMini+origMiniCTV.replace("CTV_3", "CTV_"+extraCTV).replace("aug_none", "aug_swapdown")],
        [prefixMini+origMiniCTV.replace("pro_F", "pro_T")], 
        [prefixMini+origMiniCTV.replace("CTV_3", "CTV_"+extraCTV).replace("pro_F", "pro_T")]
        ]
    plotOrig(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials, outPostFix=dataType)

#hidden reps
#DCI-S, DCI, and DRI, DCI-R, DCI-A,
#DCI-D, and DCI-10
def comparePro2(dataSet, dataType="TestC"):
    trials = range(0, 10)
    prefixMini = nodeType + "/GraphDeepLearningMini/GraphDeepLearning/dri/experiments/models/"
    prefixMini10 = nodeType + "/GraphDeepLearningMini10_2/GraphDeepLearning/dri/experiments/models/"
    namesJoel = []
    if "patents" in dataSet:
        prefixMini=nodeType + "/GraphDeepLearningMini10/GraphDeepLearning/dri/experiments/models/"
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_10_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_2_trial_0_fold_0_BAE_"+dataType+".npy"
    elif dataSet!="facebook" and dataSet!="IMDB_5" and "amazon_DVD" not in dataSet:
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_2_trial_0_fold_0_BAE_"+dataType+".npy"
    else:
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_2_trial_0_fold_0_BAE_"+dataType+".npy"
    namesJoel = ['trial']


    filePaths=[[prefixMini+origMiniCTV], 
    [prefixMini+origMiniCTV.replace("lH_F", "lH_T")],
    [prefixMini+origMiniCTV.replace("lH_F", "lH_T").replace("lR_F", "lR_T")]
    ]
    
    #names2 = ["GRNN-C", "GRNN-C-Swap", "GRNN-C-R", "GRNN-C-Swap-R"]
    names2 = ["DCI-WSB", "DCI-W", "DCI-WR"]
    #filePaths=filePaths[2:]+filePaths[0:2]
    #names2 = names2[2:]+names2[0:2]
    plotOrig(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials, outPostFix=dataType)
    BAEsPairwise(dataSet, names2, filePaths, namesJoel=namesJoel, namesMe=[("DCI-W", "DCI-WSB"), ("DCI-WR", "DCI-WSB")], trials=trials)

def compareMaxEntInf2(dataSet, dataType="TestC", extraCTV="3"):
    trials = range(0, 10)
    prefixMini = nodeType + "/GraphDeepLearningMini/GraphDeepLearning/dri/experiments/models/"
    prefixMini10 = nodeType + "/GraphDeepLearningMini10_2/GraphDeepLearning/dri/experiments/models/"
    namesJoel = []
    names2 = ["DCI-WSB", "DCI-WSB-MEF", "DCI-S", "DCI-S-MEF", "DCI-B", "DCI-B-MEF"]
    if "patents" in dataSet:
        names2 = ["DCI-10-WSB", "DCI-10-WSB-MEF", "DCI-10-S", "DCI-10-S-MEF", "DCI-10-B", "DCI-10-B-MEF"]
        prefixMini=nodeType + "/GraphDeepLearningMini10/GraphDeepLearning/dri/experiments/models/"
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_10_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_"+extraCTV+"_trial_0_fold_0_BAE_"+dataType+".npy"
    elif dataSet!="facebook" and dataSet!="IMDB_5" and "amazon_DVD" not in dataSet:
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_"+extraCTV+"_trial_0_fold_0_BAE_"+dataType+".npy"
    else:
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_"+extraCTV+"_trial_0_fold_0_BAE_"+dataType+".npy"
    namesJoel = ['trial']


    filePaths=[[prefixMini+origMiniCTV],
    [prefixMini+origMiniCTV], 
    [prefixMini+origMiniCTV.replace("aug_none", "aug_swapdown")],
    [prefixMini+origMiniCTV.replace("aug_none", "aug_swapdown")],
    [prefixMini+origMiniCTV.replace("pro_F", "pro_T")],
    [prefixMini+origMiniCTV.replace("pro_F", "pro_T")]
    ]
    
    #names2 = ["GRNN-C", "GRNN-C-Swap", "GRNN-C-R", "GRNN-C-Swap-R"]
    #filePaths=filePaths[2:]+filePaths[0:2]
    #names2 = names2[2:]+names2[0:2]
    plotOrig(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials, outPostFix="MEF")
    BAEsPairwise(dataSet, names2, filePaths, namesJoel=namesJoel, namesMe=[(names2[1], names2[0]), (names2[3], names2[2]), (names2[5], names2[4])], trials=trials)   

#DCI-S, DCI, and DRI, DCI-R, DCI-A,
#DCI-D, and DCI-10
def DCINS(dataSet, dataType="TestC"):

    trials = range(0, 10)
    prefixMini = nodeType + "/GraphDeepLearningMini/GraphDeepLearning/dri/experiments/models/"
    prefixMini10 = nodeType + "/GraphDeepLearningMini10_2/GraphDeepLearning/dri/experiments/models/"

    if "patents" in dataSet:
        prefixMini=nodeType + "/GraphDeepLearningMini10/GraphDeepLearning/dri/experiments/models/"
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_10_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_2_trial_0_fold_0_BAE_"+dataType+".npy"
    elif dataSet!="facebook" and dataSet!="IMDB_5" and "amazon_DVD" not in dataSet:
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_2_trial_0_fold_0_BAE_"+dataType+".npy"
    else:
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_2_trial_0_fold_0_BAE_"+dataType+".npy"


    filePaths = [[prefixMini+origMiniCTV],
    [prefixMini+origMiniCTV.replace("CTV_2", "CTV_3")],
    [prefixMini+origMiniCTV.replace("CTV_2", "CTV_3").replace("aug_none", "aug_swapdown")],
    [prefixMini+origMiniCTV.replace("CTV_2", "CTV_3").replace("pro_F", "pro_T")],
    [prefixMini+origMiniCTV.replace("aug_none", "aug_swapdown")],
    [prefixMini+origMiniCTV.replace("pro_F", "pro_T")]
    ]
    #names2 = ["GRNN-C", "GRNN-C-Swap", "GRNN-C-R", "GRNN-C-Swap-R"]
    names2 = ["DCI", "DCI-NS", "DCI-NS-Swap", "DCI-NS-B", "DCI-S", "DCI-B"]
    #filePaths=filePaths[2:]+filePaths[0:2]
    #names2 = names2[2:]+names2[0:2]
    plotOrig(dataSet, names2, filePaths,  trials=trials, outPostFix=dataType)

def compareRNCC(dataSet, dataType="TestC", extraCTV=3):
    prefixRNCC = nodeType + "/GraphDeepLearningMiniRNCC/GraphDeepLearning/dri/experiments/models/"
    origRNCC= "w_no0_LSTM2_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_1_mNPro_200_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_T_lR_F_CTV_3_trial_0_fold_0_BAE_"+dataType+".npy"

    trials = range(0, 10)
    prefixMini = nodeType + "/GraphDeepLearningMini/GraphDeepLearning/dri/experiments/models/"
    prefixMini10 = nodeType + "/GraphDeepLearningMini10_2/GraphDeepLearning/dri/experiments/models/"

    if "patents" in dataSet:
        prefixMini=nodeType + "/GraphDeepLearningMini10/GraphDeepLearning/dri/experiments/models/"
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_10_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_"+str(extraCTV)+"_trial_0_fold_0_BAE_"+dataType+".npy"
    elif dataSet!="facebook" and dataSet!="IMDB_5" and "amazon_DVD" not in dataSet:
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_"+str(extraCTV)+"_trial_0_fold_0_BAE_"+dataType+".npy"
    else:
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_"+str(extraCTV)+"_trial_0_fold_0_BAE_"+dataType+".npy"

    filePaths = [[prefixRNCC+origRNCC], 
    [prefixRNCC+origRNCC.replace("aug_none", "aug_swapdown")],
    [prefixRNCC+origRNCC.replace("pro_F", "pro_T")],
    [prefixMini+origMiniCTV],
    [prefixMini+origMiniCTV.replace("aug_none", "aug_swapdown")],
    [prefixMini+origMiniCTV.replace("pro_F", "pro_T")]
    ]
    names2 = ["RNCC", "RNCC-S", "RNCC-B", "DCI-WSB", "DCI-S", "DCI-B"]
    plotOrig(dataSet, names2, filePaths, trials=trials, outPostFix=dataType)

def combineModelsV2(dataSet, dataType="TestC", extraCTV=5):
    trials = range(0, 10)
    prefixMini = nodeType + "/GraphDeepLearningMini/GraphDeepLearning/dri/experiments/models/"
    prefixMini10 = nodeType + "/GraphDeepLearningMini10_2/GraphDeepLearning/dri/experiments/models/"

    if "patents" in dataSet:
        prefixMini=nodeType + "/GraphDeepLearningMini10/GraphDeepLearning/dri/experiments/models/"
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_10_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_"+str(extraCTV)+"_trial_0_fold_0_BAE_"+dataType+".npy"
    elif dataSet!="facebook" and dataSet!="IMDB_5" and "amazon_DVD" not in dataSet:
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_"+str(extraCTV)+"_trial_0_fold_0_BAE_"+dataType+".npy"
    else:
        origMiniCTV = "w_no0_LSTM_aug_none_"+dataSet+"_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_CTV_"+str(extraCTV)+"_trial_0_fold_0_BAE_"+dataType+".npy"

    filePaths =[[prefixMini+origMiniCTV],
    [prefixMini+origMiniCTV.replace("aug_none", "aug_swapdown")],
    [prefixMini+origMiniCTV.replace("pro_F", "pro_T")],
    [prefixMini+origMiniCTV.replace("aug_none", "aug_swapdown"), prefixMini+origMiniCTV.replace("pro_F", "pro_T")]
    ]

    names2 = ["DCI-WSB", "DCI-S", "DCI-B", "DCI"]
    plotOrig(dataSet, names2, filePaths, trials=trials, outPostFix=dataType)


if __name__ == "__main__":
    #mainPatents("patents_computers_50attr", JoelType="N2V")
    #comparePro(sys.argv[1])
    #compareMaxEntInf(sys.argv[1])
    #compareDRIBAG(sys.argv[1])
    #compareMinisDCI(sys.argv[1])
    #compareACCBAEHogun(sys.argv[1])
    #main1("amazon_DVD_7500", JoelType="N2V")
    #main1("facebook", JoelType="N2V")
    #main1("amazon_DVD_20000", JoelType="N2V")
    #main1("amazon_Music_64500", JoelType="N2V")
    #main1("amazon_Music_7500", JoelType="N2V")
    #main1("IMDB_5", JoelType="N2V")
    #main2("amazon_DVD_7500", JoelType="N2V")
    #main2("facebook", JoelType="N2V")
    #main2("IMDB_5", JoelType="N2V")
    #mainN2V(sys.argv[1])
    #mainN2V(sys.argv[1], dataType="ValC")
    #compareCTV(sys.argv[1])
    #DCIVariants(sys.argv[1])
    #mainN2V(sys.argv[1])
    #mainN2V(sys.argv[1], dataType="ValC")
    #DCIVariantsRandom(sys.argv[1])
    #DCIvalidPreds(sys.argv[1])
    #DCIvalidPreds(sys.argv[1], dataType="ValC")
    #comparePro2(sys.argv[1])
    #compareMaxEntInf2(sys.argv[1])
    #compareDRIBAG(sys.argv[1])
    combineModelsV2(sys.argv[1])
    #DCIvalidPreds(sys.argv[1], extraCTV=5)
    #DCIvalidPreds(sys.argv[1], dataType="Val2C", extraCTV=5)
    #DCINS(sys.argv[1])
    #DCINS(sys.argv[1], dataType="ValC")
    #compareRNCC(sys.argv[1], extraCTV=3)
    #combineModelsV2(sys.argv[1])
     
