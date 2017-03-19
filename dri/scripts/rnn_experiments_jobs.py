import os

#names, 1 - amazon_DVD_20000, 2 - facebook, 3 - amazon_Music_10000
names=[2]
memory =[5,10, 20]
prTypes = ["pos", "neg", "neutral"]
netTypes=["LSTM"]
singlePRs=[0]


buildStr='./parallel_commands '
for nettype in netTypes:
    for mem in memory:
        for prtype in prTypes:
            for singlepr in singlePRs:
                for test in names:
                    buildStr=buildStr+'"time python rnnExperiments.py ' + nettype + ' ' + str(mem) + ' ' + prtype + ' ' + str(singlepr) + ' ' + str(test)  + '" '
print(buildStr)
os.system(buildStr)
