import os
nodeType=os.environ.get('RCAC_SCRATCH')
queue=""
if "snyder" in nodeType:
    queue='ribeirob'
if "hammer" in nodeType:
    queue='csit'
#queue = 'csit'

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
#dataSets=["facebook"]
#dataSets=["amazon_DVD_20000"]
dataSets=["amazon_Music_10000", "amazon_DVD_7500", "amazon_Music_7500" ]
memory =[10]
summaries = [3]
prTypes = ["neutral"]
netTypes=["LSTM"]
singlePRs=[0]
biasselves = ["none"]
degrees=[1]
limits=[0]
usePrevWeights=[1, 0]
noDeg0s = [0, 1]

buildStr='./parallel_commands '
for nettype in netTypes:
    for mem in memory:
        for prtype in prTypes:
            for singlepr in singlePRs:
                for bias in biasselves:
                    for deg in degrees:
                        for limit in limits:
                            for summ in summaries:
                                for weight in usePrevWeights:
                                    for deg0 in noDeg0s:
                                        for test in dataSets:
                                            submitStr="qsub -q "+queue+" -v nettype='" + str(nettype)+"',mem='" +str(mem)+"',prtype='" + prtype + "',singlepr='"+str(singlepr) + "',test='"+str(test) + "',bias='" + bias + "',deg='" + str(deg) + "',limit='" +str(limit)+ "',summ='"+str(summ)+"',weight='"+str(weight)+ "',nodeg='" +str(deg0) + "' rnn_exp2.sub"
                                            print(submitStr)
                                            os.system(submitStr)
