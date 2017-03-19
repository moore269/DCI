import os
import time
#time.sleep(17000)

selectFolds = [1, 3, 5, 7, 9, 11, 13, 15]
#selectFolds = [9, 11, 13, 15]
#selectFolds = [1, 5, 9, 11, 13]
#selectFolds = [5, 11, 7, 9]
#selectFolds = [3,7, 15]
#selectFolds = [15]
#selectFolds = [6]
#selectFolds = [15]
#selectFolds = [15]
#selectFolds = [1, 5, 9, 11, 13]

nodeType=os.environ.get('RCAC_SCRATCH')
queue=""
if "snyder" in nodeType:
    queue='ribeirob'
if "hammer" in nodeType:
    queue='csit'
queue = 'csit'
#queue = 'standby'
newQueue = queue
#names, 1 - amazon_DVD_20000, 2 - facebook, 3 - amazon_Music_10000
"""
dataSets=["amazon_Music_10000"]
memory =[5]
summaries = [3]
prTypes = ["neutral"]
netTypes=["LSTMPAR"]
singlePRs=[0]
biasselves = ["posneg"]
degrees=[1, 0]
limits=[0]

"""

#dataSets = ["facebook_filtered", "imdb_binary_filtered"]
#dataSets = ["imdb_binary_filtered"]
#dataSets= ["IMDB_5"]
#dataSets = ["facebook", "amazon_DVD_20000", "amazon_DVD_7500"]
#dataSets = ["facebook", "amazon_DVD_20000"]
#dataSets = ["amazon_DVD_7500", "IMDB_5"]
#dataSets=["facebook", "IMDB_5", "amazon_DVD_7500", "amazon_DVD_20000"]
#dataSets=["facebook"]
#dataSets = ["amazon_DVD_20000"]
#dataSets=["facebook", "amazon_DVD_7500", "amazon_DVD_20000"]
#dataSets=["amazon_DVD_7500", "amazon_DVD_20000"]
#dataSets=["IMDB_5", "amazon_DVD_7500"]
#dataSets=["amazon_DVD_20000"]
#dataSets=["amazon_DVD_7500"]
#dataSets=["amazon_DVD_7500"]
#dataSets = ["amazon_DVD_20000", "facebook"]
#dataSets=["amazon_Music_64500"]
#dataSets=["IMDB_5", "amazon_Music_7500", "amazon_Music_64500"]
#dataSets=["amazon_DVD_20000", "amazon_Music_64500"]
#dataSets=["amazon_Music_7500"]
#dataSets=["amazon_DVD_7500"]
#dataSets = ["amazon_Music_7500", "amazon_Music_64500"]
#dataSets = ["amazon_Music_64500"]
#dataSets=["amazon_DVD_7500", "amazon_DVD_20000", "amazon_Music_64500", "amazon_Music_7500"]
#dataSets = ["facebook", "IMDB_5", "amazon_DVD_7500", "amazon_DVD_64500", "amazon_Music_7500", "amazon_Music_64500"]
dataSets = ["patents_computers_50attr"]
memory =[10]
#memory = [20]
minidims = [5]
prTypes = ["neutral"]
#netTypes=["LSTMwMini", "LSTM2"]
#netTypes=["LSTMwMini"]
#netTypes=["LSTM"]
netTypes=["LSTM2"]
#netTypes=["LRAVG", "LRAVGaverage", "LRAVGaverageOrig"]
singlepr=0
bias = "none"
degrees=[1]
limits=[0]
usePrevWeights=[1]
noDeg0s = [1]
#dataAug=["none"]
#dataAug = ["none"]
dataAug=["none"]
#dataAug=["swapdown", "none"]

#dataAug=["swapdown"]

#dataAug=["none", "swap", "swapdown"]
#dataAug=["swap", "swapdown"]
#dataAug=["attr", "attrdown"]
#dataAug=["none", "swap", "swapdown", "attr", "attrdown"]
#dataAug = ["attr", "swap", "attrup", "attrdown", "swapup", "swapdown"]
#dataAug = ["swapTriple", "swapQuad"]
#dataAug = ["swapup", "swapdown"]
randInit = [0]
#usePRs = ["for", "back"]
#usePRs = ["for"]
usePRs = [0]
gpus = ["cpu"]
#perturb = [0,1]
perturb = [0]
lHs = [0]
lHRs = [0]
usePros = [1]
changeTraVals = [3]
#PostFix="PPR"
#PostFix="N2V"
if dataSets[0]=="patents_computers_50attr" and len(dataSets)==1:
    PostFix=""
    queue = 'csit'
    newQueue = 'csit'
    trials = xrange(5, 10)
else:
    PostFix=""
    trials = xrange(0, 10)
if "LSTM2" in netTypes:
    PostFix="RNCC"

#PostFix="10_2"
buildStr='./parallel_commands '
for nettype in netTypes:
    for mem in memory:
        for minidim in minidims:
            for prtype in prTypes:
                for deg in degrees:
                    for limit in limits:
                        for weight in usePrevWeights:
                            for deg0 in noDeg0s:
                                for aug in dataAug:
                                    for test in dataSets:
                                        for rinit in randInit:
                                            for usepr in usePRs:
                                                for gpu in gpus: 
                                                    for trial in trials:
                                                        for fold in selectFolds:
                                                            for pur in perturb:
                                                                for usePro in usePros:
                                                                    for lH in lHs:
                                                                        for lhr in lHRs:
                                                                            for cHR in changeTraVals:
                                                                                if nettype!="LSTM2" and ((test=="IMDB_5" and fold>5) or ("amazon_Music" in test and fold>5)):
                                                                                    newQueue = "csit"
                                                                                else:
                                                                                    newQueue = queue
                                                                                if newQueue=="standby":
                                                                                    walltime = " -l walltime=04:00:00 "
                                                                                else:
                                                                                    walltime = " -l walltime=30:00:00 "
                                                                                submitStr="qsub -q "+newQueue+walltime+"-v nettype='" + str(nettype)+"',mem='" +str(mem)+"',minidim='"+str(minidim)+"',prtype='" + prtype + "',singlepr='"+str(singlepr) + "',test='"+str(test) + "',bias='" + bias + "',deg='" + str(deg) + "',limit='" +str(limit)+"',weight='"+str(weight)+ "',nodeg='" +str(deg0) + "',aug='" + str(aug) + "',rinit='" +str(rinit) + "',usepr='" + str(usepr)  + "',gpu='" + str(gpu) +  "',trial='"+str(trial)+"',fold='"+str(fold)+ "',pur='"+str(pur) + "',pro='" + str(usePro)+ "',lh='"+ str(lH)+"',lhr='"+str(lhr)+"',chr='"+str(cHR)+ "' rnn_exp_mini"+PostFix+".sub"
                                                                                print(submitStr)
                                                                                os.system(submitStr)
