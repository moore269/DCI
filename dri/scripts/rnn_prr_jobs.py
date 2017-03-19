import os
import sys
queue='csit'

#names, 1 - amazon_DVD_20000, 2 - facebook, 3 - amazon_Music_10000
dataSets=["amazon_DVD_20000"]
memory =[10]
prTypes = ["neutral"]
netTypes=["LSTM"]
singlePRs=[0]
biasselves = ["self", "notself", "pos", "neg"]
degrees=[1, 0]
limits=[1,0]


for nettype in netTypes:
    for mem in memory:
        for prtype in prTypes:
            for singlepr in singlePRs:
                for test in dataSets:
                    for bias in biasselves:
                        for deg in degrees:
                            for limit in limits:
                                submitStr="qsub -q " + queue +" -v nettype='"+ nettype + "',mem='" + str(mem) + "',prtype='" + str(prtype) + "',singlepr='" + str(singlepr)+ "',test='" + str(test)  + "',bias='" + bias + "',deg='" + str(deg) + "',limit='" + str(limit) + "' rnn_prr.sub"
                                print(submitStr)