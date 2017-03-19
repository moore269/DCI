import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import math
from pylab import *
import numpy as np
import os.path

# Load config parameters
nodeType=os.environ.get('RCAC_SCRATCH')
if nodeType==None:
    JoelFolder = "/Users/macbookair/Documents/Joel/"
else:
    JoelFolder =nodeType+"/Joel/"
#trials = 5
selectFolds = [1, 3, 5, 7, 9, 11, 13, 15]

def plotAcc(dataY, dataStd, titleName, rnnMeans=None, rnnStds=None, stderr=True, trials=10):
    folds = dataY['trial']
    x = np.array(folds)*0.05+0.15
    print(x)
    axes = gca()
    axes.set_xlim([0.15, 0.95])
    ax = subplot(111)
    for key in dataY:
        if key!='trial':
            y = np.array(dataY[key])
            std = np.array(dataStd[key])

            print(key)
            print('y: '+'\t'.join(map(str, dataY[key])))
            print('std: '+'\t'.join(map(str, dataStd[key])))
            if stderr:
                std = std / math.sqrt(trials)
            ax.errorbar(x, y, yerr=std, label=key.replace("PL-EM (CAL)", "PL-EM"))

    #if we have original rnn input
    if rnnMeans!=None and rnnStds!=None:
        for key in rnnMeans:
            #determine which selectFolds
            if len(rnnMeans[key])==3:
                folds = [3, 7, 15]
            elif len(rnnMeans[key])==8:
                folds = [1, 3, 5, 7, 9, 11, 13, 15]
            else:
                print("length of folds does not equal prespecified")
                sys.exit(0)
            x = np.array(folds)*0.05+0.15

            print(key)
            print('y: '+'\t'.join(map(str, rnnMeans[key])))
            print('std: '+'\t'.join(map(str, rnnStds[key])))
            std = rnnStds[key]
            if stderr:
                std = rnnStds[key] / math.sqrt(trials)
            ax.errorbar(x, rnnMeans[key], yerr=std, label=key)

    xlabel('Train Set Proportion')
    ylabel('BAE')
    box=ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, fancybox=True, shadow=True)
    grid(True)
    #title(titleName)
    savefig(titleName+".png")
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
    return data

#read in our original input in .np format
#return mean and std
def readNP(names, filePaths, trials):
    means = {}
    stds = {}
    for i, path in enumerate(filePaths):
        x = np.zeros((trials, len(selectFolds)))
        for trial in range(0, trials):
            for fInd, fold in enumerate(selectFolds):
                newPath = path.replace("trial_0", "trial_"+str(trial)).replace("fold_0", "fold_"+str(fold))
                #passed=False
                try:
                    x[trial][fInd] = np.load(newPath)
                    #passed=True
                except:
                    print("NP: trial: "+str(trial)+", fold: "+str(fold))
                    print(newPath) #pass
                #try:
                #    newPath = path.replace("trial_0", "trial_"+str(10)).replace("fold_0", "fold_"+str(fold))
                #    x[trial][fInd] = np.load(newPath)
                #    passed=True
                #except:
                #    pass
                #if not passed:
                #    print(newPath)
                #    print("not valid trial: "+str(trial)+", fold: "+str(fold))
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

def plotOrig(dataName, names, fileNames, namesJoel=None, trials = 10):
    dataY, dataStd = readMultiple(dataName, namesJoel) 
    rnnMeans, rnnStds = readNP(names, fileNames, trials)
    plotAcc(dataY, dataStd, "OutputPlots/"+dataName, rnnMeans=rnnMeans, rnnStds=rnnStds, trials=trials)

def BAEsPairwise(dataName, names, fileNames, namesJoel=None, trials = 10):
    dataY, dataStd = readMultiple(dataName, namesJoel)
 
    rnnMeans, rnnStds = readNP(names, fileNames, trials)
    print("Data: "+dataName)
    for rnnKey in rnnMeans:
        for joelKey in dataY:
            if joelKey!='trial':
                baeGain=[]
                for i in range(0, len(rnnMeans[rnnKey])):
                    baeGain.append(float((dataY[joelKey][i] - rnnMeans[rnnKey][i]))/dataY[joelKey][i])
                baeGainStr = ','.join([str(j) for j in baeGain])
                print(rnnKey+"--"+joelKey+" : "+baeGainStr)




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
def main1(dataSet, JohnType=""):
    trials = 10
    prefix = "/scratch/snyder/m/moore269/GraphDeepLearning3"+JohnType+"/GraphDeepLearning/pylearn2/experiments/tests/models/"
    rinit=["F"]
    orig = JohnType+"w_no0_LSTM_aug_none_"+dataSet+"_noVl_F_Mem_10_Ex_1_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_F_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_sd_3_rinit_F_trial_0_fold_0_BAE_TestC.npy"
    aug = orig 
    #pprNames=["F"]
    names = ["swapdown"]
    #names = ["attr", "swap", "attrDouble", "swapDouble"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = [ "swap", "swapDouble", "swapTriple", "attr", "attrDouble", "attrTriple"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = ["attr", "attrDouble", "attrTriple", "attrQuad"]
    #namesJoel = ["PL-EM (CAL)", 'CL-EM (10)', 'PL-EM', 'RLR', 'trial']
    namesJoel = ["PL-EM (CAL)", 'LR', 'LP', "PL-EM (CAL)+N2V", "LR+N2V"]
    #namesJoel = ['trial']
    filePaths = []
    for rin in rinit:
        for name in names:
            fPath = prefix+aug.replace("aug_none", "aug_"+name)
            filePaths.append(fPath)

    names2 = ["DCI"]
    plotOrig(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials)
    BAEsPairwise(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials)

#PL-EM, PL-EM-M, LR, LP
#DCI-S, DCI
def mainJohnvsJoelN2V(dataSet, JohnType="N2V", JoelType="N2V"):
    trials = 10
    prefix1  = "/scratch/snyder/m/moore269/GraphDeepLearning3/GraphDeepLearning/pylearn2/experiments/tests/models/"
    prefix = "/scratch/snyder/m/moore269/GraphDeepLearning3"+JohnType+"/GraphDeepLearning/pylearn2/experiments/tests/models/"
    rinit=["F"]
    orig = JohnType+"w_no0_LSTM_aug_none_"+dataSet+"_noVl_F_Mem_10_Ex_1_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_F_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_sd_3_rinit_F_trial_0_fold_0_BAE_TestC.npy"
    orig1 = "w_no0_LSTM_aug_none_"+dataSet+"_noVl_F_Mem_10_Ex_1_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_F_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_sd_3_rinit_F_trial_0_fold_0_BAE_TestC.npy"

    aug = orig 
    #pprNames=["F"]
    names = ["swapdown"]
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
            fPath = prefix1+orig1.replace("aug_none", "aug_"+name)
            filePaths.append(fPath)
            fPath = prefix+orig.replace("aug_none", "aug_"+name)
            filePaths.append(fPath)

    names2 = ["DCI-S", "DCI-S-N2V"]
    plotOrig(dataSet, names2, filePaths, JoelType=JoelType, namesJoel=namesJoel, trials=trials)
    BAEsPairwise(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials)




#PL-EM, PL-EM-M
#DCI-10, DCI-S-10
def mainPatents(dataSet):
    trials = 5
    prefix = "tests/models/"
    rinit=["F"]
    orig = "w_no0_LSTM_aug_none_"+dataSet+"_noVl_F_Mem_10_Ex_1_MTra_T_mEp_200_mNPro_100_trls_10_sFlds_T_avgNLab_F_dynLab_T_ActLabs_F_onlyLab_F_PPR_F_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_sd_3_rinit_F_trial_0_fold_0_BAE_TestC.npy"
    aug = orig 
    names = ["swapdown"]
    #names = ["none", "attr", "attrdown",  "attrup",  "swap"]
    #names = ["attr", "swap", "attrDouble", "swapDouble"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = [ "swap", "swapDouble", "swapTriple", "attr", "attrDouble", "attrTriple"]
    #names = [ "swap", "swapDouble", "swapTriple", "swapQuad"]
    #names = ["attr", "attrDouble", "attrTriple", "attrQuad"]
    #namesJoel = ["PL-EM (CAL)", 'CL-EM (10)', 'PL-EM', 'RLR', 'trial']
    namesJoel = ["PL-EM (CAL)", 'LR', 'LP', "PL-EM (CAL)+N2V", "LR+N2V"]
    filePaths = []
    for rin in rinit:
        for name in names:
            fPath = prefix+aug.replace("aug_none", "aug_"+name).replace("rinit_F", "rinit_"+rin)
            filePaths.append(fPath)

    names2 = ["DCI-10"]
    plotOrig(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials)
    BAEsPairwise(dataSet, names2, filePaths, namesJoel=namesJoel, trials=trials)


if __name__ == "__main__":
    mainPatents("patents_computers_50attr")
    """main1("amazon_DVD_7500")
    main1("facebook")
    main1("amazon_DVD_20000")
    main1("amazon_Music_64500")
    main1("amazon_Music_7500")
    main1("IMDB_5")
    """
    """main1("amazon_DVD_7500", JoelType="N2V")
    main1("facebook", JoelType="N2V")
    main1("amazon_DVD_20000", JoelType="N2V")
    main1("amazon_Music_64500", JoelType="N2V")
    
    main1("amazon_Music_7500", JoelType="N2V")
    main1("IMDB_5", JoelType="N2V")
    """
    #mainJohnvsJoelN2V("amazon_DVD_7500")
    #mainJohnvsJoelN2V("amazon_DVD_20000")
    #mainJohnvsJoelN2V("facebook")
    #mainJohnvsJoelN2V("IMDB_5")
    #mainJohnvsJoelN2V("amazon_Music_7500")
    #mainJohnvsJoelN2V("amazon_Music_64500")
    #main1("amazon_DVD_7500", JohnType="N2V", JoelType="N2V")
    #main1("facebook", JohnType="N2V",JoelType="N2V")
    #main1("amazon_DVD_20000", JohnType="N2V",JoelType="N2V")
    #main1("amazon_Music_64500", JohnType="N2V",JoelType="N2V")
    #main1("amazon_Music_7500", JohnType="N2V",JoelType="N2V")
    #main1("IMDB_5", JohnType="N2V",JoelType="N2V")
    
    #main2("amazon_DVD_7500", JoelType="N2V")
    #main2("facebook", JoelType="N2V")
    #main2("IMDB_5", JoelType="N2V")
     
