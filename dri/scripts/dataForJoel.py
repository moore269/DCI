import sys
import os
#add previous folder to path
cwd = os.getcwd(); rIndex = cwd.rfind("/");cwd = cwd[:rIndex];
sys.path.insert(0, cwd)

import code.readData.readData as readData


def printData(data, fileName):
    f = open(fileName+".txt", 'w')
    for node in data:
        f.write(str(node)+"\n")

trials = 10
#percentValidation=0.15
percentValidation=0.1
numFolds=9
dataSets=["facebook_oneday_filtered"]

for fName in dataSets:
    for i in range(0, trials):

        #read trial from file
        rest, validationNodes= readData.readTrial("../experiments/data/", fName, i, percentValidation)
        printData(validationNodes, "../experiments/data/"+fName+"_trial_"+str(i)+"_val")

        #split into folds
        folds= readData.splitNodeFolds(rest, numFolds)
        for j, fold in enumerate(folds):
            printData(fold, "../experiments/data/"+fName+"_trial_"+str(i)+"_fold_"+str(j))



    