import os
import sys
nodeType=os.environ.get('RCAC_SCRATCH')
queue=""
if "snyder" in nodeType:
    queue='ribeirob'
if "hammer" in nodeType:
    queue='csit'
queue='csit'

# original test settings
"""
memory=[5, 10, 20]
prTypes = ["pos", "neg", "neutral"]
#1 for LSTM, 0 for GRU
netTypes=["LSTM", "LSTM2", "StackLSTM", "BIDLSTM"]
singlePRs = [1]
"""

# fine grained test
memory =[1,5,10, 20]
prTypes = ["neutral"]
netTypes=["LSTM"]
biasselves = ["self", "notself", "pos", "neg"]
degrees=[0]
#singlePRs=[1]
singlePRs=[0]
dataSets = ["facebook"]


for data in dataSets:
    for net in netTypes:
        for mem in memory:
            for prtype in prTypes:
                for singlepr in singlePRs:
                    for bias in biasselves:
                        for deg in degrees:
                            submitStr="qsub -q "+queue+" -v data='" +str(data)+"',nettype='"+str(net)+"',mem='" +str(mem)+"',prtype='" +str(prtype)+"',singlepr='"+str(singlepr) + "',biasself='" + bias + "',degree='" + str(deg) + "' rnn_prr.sub"
                            print(submitStr)
                            #os.system(submitStr)
