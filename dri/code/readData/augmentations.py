import sys
import random
import copy

#attr, swap - adds examples to match proportions
#attrdown, swapdown - adds examples to match proportions, then downsamples to get to size of original data
#attrup, swapup - adds 1x more examples regardless of class proportions for a total of 2x
#
def dataAugmentation(G, data_All, labCounts, maxNeighbors, dataAug="attr", pageRankOrder=False):
    dFlag = False
    if "Double" in dataAug:
        dFlag = True
        multBy = 2
    elif "Triple" in dataAug:
        dFlag = True
        multBy = 3
    elif "Quad" in dataAug:
        dFlag=True
        multBy = 4
    if dataAug=="attr" or dataAug=="swap":
        return data_All + attrSwapByProportion(G, data_All, labCounts, maxNeighbors, dataAug, pageRankOrder)
    elif dFlag and ("attr" in dataAug or "swap" in dataAug):
        smallLab = 1 if labCounts[0] > labCounts[1] else 0
        bigLab = abs(smallLab - 1)
        newLabCounts={}

        #TWICE AS MUCH EXAMPLE
        #e.g. len(small)=10, len(big)=100
        #  => match labCount[small]=10, labCount[big] = 200
        #  => 190 new small
        newLabCounts[smallLab] = labCounts[smallLab]
        newLabCounts[bigLab] = multBy*labCounts[bigLab]
        tSmall = attrSwapByProportion(G, data_All, newLabCounts, maxNeighbors, dataAug, pageRankOrder)

        #e.g. len(small)=10, len(big)=100 (from before)
        #  => match labCount[small]=200, labCount[big] = 100
        #  => 100 new big
        #  => total 200 small, 200 big
        #  => data is twice as much 
        newLabCounts[smallLab] = multBy*labCounts[bigLab]
        newLabCounts[bigLab] = labCounts[bigLab]
        tBig = attrSwapByProportion(G, data_All, newLabCounts, maxNeighbors, dataAug, pageRankOrder)
        return data_All + tSmall + tBig
    elif dataAug=="attrdown" or dataAug=="swapdown":
        newData = attrSwapByProportion(G, data_All, labCounts, maxNeighbors, dataAug, pageRankOrder)
        dataBig = newData + data_All
        randomIndexes = random.sample(xrange(0, len(dataBig)), len(data_All))
        return [dataBig[index] for index in randomIndexes]
    elif dataAug=="attrup" or dataAug=="swapup":
        return data_All + attrSwap(G, data_All, labCounts, maxNeighbors, dataAug, pageRankOrder)

#generate 1x more examples regardless of class proportions
def attrSwap(G, data_All, labCounts, maxNeighbors, dataAug="attr", pageRankOrder=False):
    endData=[]
    lenAttr = len(data_All[0][0][0])
    lenOrigAttr = len(G.node[data_All[0][-1]]['attr0'])
    labIncrease=1
    swapIndexes = random.sample(xrange(0, len(data_All)), len(data_All))

    #iterate through dataset and create 1x new examples
    #must pass 1 or 2 examples at a time with j=labIncrease or j=labIncrease-1
    i=0
    while i < len(data_All):
        if "attr" in dataAug :
            newData, dummy= AugmentAttributes([data_All[i]], lenAttr, lenOrigAttr, maxNeighbors, labIncrease, labIncrease, pageRankOrder)  
        elif "swap" in dataAug:
            n1 = data_All[swapIndexes[i]]
            i+=1
            #if odd, we return early
            if i==len(data_All):
                break
            n2 = data_All[swapIndexes[i]]
            newData, dummy= swapNeighbors([n1, n2], lenAttr, lenOrigAttr, maxNeighbors, labIncrease-1, labIncrease, pageRankOrder)
        else:
            print("please provide a valid data augmentation scheme")
            print("dataAug = "+str(dataAug))
            sys.exit(0)
        #updates
        endData+=newData
        i+=1     

    return endData

def attrSwapByProportion(G, data_All, labCounts, maxNeighbors, dataAug="attr", pageRankOrder=False):
    #want to increase smaller proportion label
    if labCounts[0] > labCounts[1]:
        smallLab=1
    else:
        smallLab=0
    dataSmallLab = [data for data in data_All if data[2][0]==smallLab]
    labIncrease = abs(labCounts[1] -labCounts[0])

    endData=[]
    lenAttr = len(dataSmallLab[0][0][0])
    lenOrigAttr = len(G.node[dataSmallLab[0][-1]]['attr0'])
    j= 0
    while j < labIncrease:
        if "attr" in dataAug :
            newData, j= AugmentAttributes(dataSmallLab, lenAttr, lenOrigAttr, maxNeighbors, j, labIncrease, pageRankOrder)  
        elif "swap" in dataAug:
            newData, j= swapNeighbors(dataSmallLab, lenAttr, lenOrigAttr, maxNeighbors, j, labIncrease, pageRankOrder)
        else:
            print("please provide a valid data augmentation scheme")
            print("dataAug = "+str(dataAug))
            sys.exit(0)
        endData+=newData
    return endData


#accepts a candidate dataset (list), current index j, and when to stop, labIncrease
#if we want 1 step, we simply pass list of 1 example, j=1, and labIncrease=1
def AugmentAttributes(dataSmallLab, lenAttr, lenOrigAttr, maxNeighbors, j, labIncrease, pageRankOrder):
    dataRet = []
    for data in dataSmallLab:
        j+=1
        neighbIndexes = random.sample(xrange(0,len(data[0])-1),  (len(data[0])-1)/2)
        newExampleX = copy.deepcopy(data[0])
        for neighbInd in neighbIndexes:
            attrIndex = random.randint(0, lenAttr-1)
            newNeighbor = augmentAttr(copy.deepcopy(data[0][neighbInd]), attrIndex, lenOrigAttr, pageRankOrder)
            #reassign
            newExampleX[neighbInd] = newNeighbor
        #randomly permute last guy which is the original node
        attrIndex = random.randint(0, lenAttr-1)
        newOrig = augmentAttr(copy.deepcopy(data[0][len(data[0])-1]), attrIndex, lenOrigAttr, pageRankOrder)
        #reassign
        newExampleX[len(data[0])-1] = newOrig
        row = [newExampleX] + [dd for dd in data[1:]]
        dataRet.append(row)
        if j> labIncrease:
            break

    return (dataRet, j)

#accepts a candidate dataset (list), current index j, and when to stop, labIncrease
#if we want 1 step, we simply pass list of 2 examples, j=1, and labIncrease=1
#Note that this requires 2 examples
def swapNeighbors(dataSmallLab, lenAttr, lenOrigAttr, maxNeighbors, j, labIncrease, pageRankOrder):
    dataRet=[]
    lenData = len(dataSmallLab)
    while j < labIncrease:
        j+=2
        #select two neighbors at random
        n1 = random.randint(0, lenData-1)
        n2=n1
        while n1==n2:
            n2 = random.randint(0, lenData-1)

        x1 = copy.deepcopy(dataSmallLab[n1][0])
        x2 = copy.deepcopy(dataSmallLab[n2][0])

        #if not PR, simply swap this way
        if pageRankOrder:
            neighbIndexes = random.sample(xrange(0,len(x1)-1),  (len(x1)-1)/2)
            #swapping
            for index in neighbIndexes:
                temp = x1[index]
                x1[index] = x2[index]
                x2[index] = temp

            #now swap actual nodes
            index = maxNeighbors-1
            temp = x1[index]
            x1[index] = x2[index]
            x2[index] = temp
        else:
            numSwaps = min(len(x1)/2, len(x2)/2)

            x1Indexes = random.sample(xrange(0,len(x1)-1),  numSwaps)
            x2Indexes = random.sample(xrange(0,len(x2)-1),  numSwaps)
            for i in range(0, len(x1Indexes)):
                i1 = x1Indexes[i]
                i2 = x2Indexes[i]
                temp = x1[i1]
                x1[i1] = x2[i2]
                x2[i2] = temp
        #reassign
        row1 = [x1] + [dd for dd in dataSmallLab[n1][1:]]
        row2 = [x2] + [dd for dd in dataSmallLab[n2][1:]]

        dataRet.append(row1)
        dataRet.append(row2)
    return (dataRet, j)

def augmentAttr(newNeighbor, attrIndex, lenOrigAttr, pageRankOrder):
    #binary case, flip attribute
    if attrIndex< lenOrigAttr:
        newNeighbor[attrIndex] = abs(newNeighbor[attrIndex]-1)
    #else floating point, multiply by either 90% or 110%
    else:
        coinFlip = random.randint(0, 1)
        if coinFlip==0:
            newNeighbor[attrIndex]=0.9*newNeighbor[attrIndex]
        else:
            newNeighbor[attrIndex]=1.1*newNeighbor[attrIndex]
    return newNeighbor