import os

dataset = 'amazon_Music_10000'
trialFolds = [(0, 10), (0,11), (0,13), (0, 15), 
    (1, 11), (1, 13), (2, 11), (3, 4), (3, 9), (3, 11), (3, 13),
    (4,4), (4, 9), (4, 13), (5, 1), (5, 9), (5, 13),
    (6, 1), (6, 9), (6, 11), (7, 7), (7, 9), (7, 11),
    (8, 9), (8, 11), (9, 9), (9, 11)]
queue = 'ribeirob'
prtype = 'neutral'

for setting in trialFolds:
    trial = setting[0]
    fold = setting[1]
    submitStr="qsub -q " + queue + " -v dataset='" + dataset + "',trial='" + str(trial) + "',fold='" + str(fold)  +"',prtype='" + prtype+"' PRR.sub"
    print(submitStr)
    os.system(submitStr)

