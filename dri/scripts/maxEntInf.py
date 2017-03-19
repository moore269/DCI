import sys
import os
#add previous folder to path
cwd = os.getcwd(); rIndex = cwd.rfind("/");cwd = cwd[:rIndex];
sys.path.insert(0, cwd)

import numpy as np
import code.readData.readData as readData
import code.graph_helper.graph_helper as graph_helper
from experiments.config import config
import math
locals().update(config)

#predict per example
def BAE(nodes, actualLabs): 
    err1=0.0
    err0=0.0
    count1=0
    count0=0
    for key in nodes:

        pred = nodes[key]
        actual = actualLabs[key][0]
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

    if divideBy==0:
        return 1.0

    BAE=(err1+err0)/divideBy
    return BAE

def maxEntInf(namePath):
    dataNames = ["facebook", "IMDB_5", "amazon_DVD_7500", "amazon_DVD_20000", "amazon_Music_7500", "amazon_Music_64500", "patents_computers_50attr"]
    trainN = np.load(namePath+"TraC.npy")
    validN = np.load(namePath+"ValC.npy")
    testN = np.load(namePath+"TestC.npy")

    #grab name from filePath
    fName=""
    for name in dataNames:
        if name in namePath:
            fName=name

    #compute label proportions
    G = readData.readDataset(dataFolder, fName) 
    traValNodes = [x[0] for x in trainN] + [x[0] for x in validN] 
    labProps = graph_helper.getLabelCounts(G, traValNodes)
    total = 0
    negProp = 0.0
    for key in labProps:
        total += labProps[key]
    for key in labProps:
        labProps[key] = float(labProps[key])/total
    #assume that negative labels are higher proportioned
    for key in labProps:
        if negProp < labProps[key]:
            negProp = labProps[key]

    #print(negProp)

    testNodes = {}
    testNodesOld = {}
    for tu in testN:
        x = float(tu[1])
        if x == 1:
            x = 0.999999999999999
        elif x == 0:
            x = 0.000000000000001
        testNodes[tu[0]]=math.log(x/ (1-x))
        testNodesOld[tu[0]] = x


    centralIndex = int(negProp*len(testNodes))
    centralValue = sorted(testNodes.values())[centralIndex]

    testNodesNewProbs = {}
    actualLabs = {}
    for tu in testNodes:
        x = testNodes[tu]
        newX = x - centralValue
        testNodesNewProbs[tu] = 1.0/(1+math.exp(-newX))
        #print('nodeID: '+str(tu)+', prob: '+str(testNodesNewProbs[tu]) + ', oldprob: '+str(testNodesOld[tu]))
        actualLabs[tu] = G.node[tu]['label']
    return BAE(testNodesNewProbs, actualLabs)

def main():
    namePath = sys.argv[1]
    print(maxEntInf(namePath))

if __name__ == "__main__":
    main()