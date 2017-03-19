import sys
from collections import defaultdict
import numpy as np
import theano
from random import shuffle
from collections import OrderedDict
import os
from code.graph_helper.OrderedGraph import *
from augmentations import *

#this function is for encoding data for use in the RNN
#max neighbors limits the number of neighbors a sample can have
#also keep track of node ID via dataID
def encode_data_VarLen(G, nodes, attrKey, maxNeighbors, useActualLabs=False, onlyLabs=False, useInputX2=False, nodeIDs=False, usePrevWeights=False, labCounts=None, dataAug="none", pageRankOrder="F", usePro=False, lastH=False):
    X2AttrKey=attrKey
    if onlyLabs and useActualLabs:
        X2AttrKey='JustLabInfo_TrueLab'
    elif onlyLabs:
        X2AttrKey='JustLabInfo'
    elif useActualLabs:
        X2AttrKey='attr3'
    else:
        X2AttrKey=attrKey

    pprName='PPRweightsOrder'
    #can only use onlyLabs when we are using inputx2
    #maybe we can
    #if onlyLabs and not useInputX2:
    #    print("onlyLabs=True only when useInputX2=True")
    #    sys.exit(0)

    #build dictionary of examples
    data_All=[]
    for node in nodes:
        exampleX = [] 
        exampleY = []
        examplePro = []
        #Add node to be predicted at beginning
        #exampleX.append(G.node[node][attrKey])
        exampleY=G.node[node]['label']
        if usePro:
            if exampleY[0]==0:
                examplePro = [labCounts[1]]
            elif exampleY[0]==1:
                examplePro = [labCounts[0]]
        else:
            examplePro = [-1]

        lenNeighb = len(G.neighbors(node))
        origAttr = None

        #if we don't want order to matter
        if not isinstance(G, OrderedGraph):
            neighbors = G.neighbors(node)
            #randomly permute neighbors and then feed it in as examples
            shuffle(neighbors)
            for i, neighbor in enumerate(neighbors):
                if i>maxNeighbors:
                    break
                neighbAttr = G.node[neighbor][X2AttrKey]
                #print(neighbAttr)
                if lastH:
                    neighbAttr = neighbAttr+G.node[neighbor]['lastH']
                exampleX.append(neighbAttr)
                #exampleY.append(G.node[neighbor]['label'])

            #Add node to be predicted at end
            #if only labs, then we can't append the attribute vector
            #we add on a default attr vector consisting of 0.5s if we have no neighbors

            if onlyLabs:
                if len(G.neighbors(node))==0:
                    exampleX.append([0.5]*len(G.node[node][X2AttrKey]))
                origAttr = G.node[node][attrKey]
            else:
                if 'attrWithPreds' in G.node[node] and usePrevWeights:
                    origAttr = G.node[node]['attrWithPreds']
                else:
                    origAttr = G.node[node][attrKey]

        #PPR logic
        else:
            neighbors=G.node[node][pprName]
            for i, neighbor in enumerate(neighbors):
                if i>maxNeighbors or i==len(neighbors)-1:
                    break
                exampleX.append(G.node[neighbor[0]][X2AttrKey]+[neighbor[1]])
            #Add node to be predicted at end
            #The graph will always have neighbors for PPR.
            #therefore, adding on a default attr vector does not apply
            if len(G.node[node][pprName])>0:
                if 'attrWithPreds' in G.node[node] and usePrevWeights:
                    origAttr = G.node[node]['attrWithPreds']+[G.node[node][pprName][i][1]] 
                else:
                    origAttr = G.node[node][attrKey]+[G.node[node][pprName][i][1]]
            #if back reverse 
            if pageRankOrder=="back":
                exampleX = exampleX[::-1]
        #if we want to append hidden representation
        #also if LSTM2, don't append hidden representation
        #this is to compare RNCC
        if lastH and not useInputX2:
            origAttr = origAttr+G.node[node]['lastH']

        #append at end if relationalLSTM
        if not useInputX2:
            exampleX.append(origAttr)

        #only include if examples made it
        if len(exampleX)>0:
            #keep original attribute vector
            #X2, X, Y, ID
            data_All.append([exampleX, origAttr, exampleY, examplePro, node])

    if dataAug !="none":
        data_All = dataAugmentation(G, data_All, labCounts, maxNeighbors, dataAug)
  
    #shuffle & separate so that we can index into them later
    shuffle(data_All)
    dataX=[]
    dataX2=[]
    dataY=[]
    dataPro = []
    dataID=[]
    for data in data_All:
        #print("======")
        #print(data[0])
        #for row in data[0]:
        #    print(len(row))
        dataX.append(np.array(data[0], dtype=theano.config.floatX))
        dataX2.append(data[1])
        dataY.append(data[2])
        dataPro.append(data[3])
        dataID.append(data[4])

    dataX=np.array(dataX)
    dataX2=np.array(dataX2, dtype=theano.config.floatX)
    dataY=np.array(dataY, dtype=theano.config.floatX)
    dataPro=np.array(dataPro, dtype=theano.config.floatX)

    XYTuples = [('x', dataX)]
    if useInputX2:
        XYTuples.append(('x2', dataX2))
    XYTuples.append(('y', dataY))
    if usePro:
        XYTuples.append(('pro', dataPro))
    if nodeIDs:
        XYTuples.append(('nodeID', np.array(dataID, dtype='uint32')))

    XY = OrderedDict(XYTuples)

    return (XY, dataID)


def pad_to_dense(M, maxlen2nd=None, maxlen3rd=None, mask=None):
    """Appends the minimal required amount of zeroes at the end of each 
     array in the jagged array `M`, such that `M` looses its jagedness."""
    """also return mask"""
    if maxlen2nd is None:
        maxlen2nd = max(len(r) for r in M)
    if maxlen3rd is None:
        maxlen3rd = M[0].shape[1:]

    #batch size, maxlen, attr length
    #or num2order, maxlen, attr length
    shape = (len(M), maxlen2nd)+maxlen3rd

    Z = np.zeros(shape, dtype=theano.config.floatX)
    Zmask = np.zeros(shape[:-1], dtype=theano.config.floatX)
    for enu, row in enumerate(M):
        Z[enu, :len(row)] += row
        if mask is None:
            oldmask=np.ones(row.shape[:-1])
        else:
            oldmask = mask[enu]

        Zmask[enu, :len(row)] += np.ones(row.shape[:-1])*oldmask
    return Z, Zmask

def pad_to_dense2(M, maxlen, attrlen, mask=None):
    """Appends the minimal required amount of zeroes at the end of each 
     array in the jagged array `M`, such that `M` looses its jagedness."""
    """also return mask"""

    #batch size, maxlen, attr length
    #or num2order, maxlen, attr length
    shape = (maxlen, attrlen)

    Z = np.zeros(shape)
    Zmask = np.zeros(maxlen)
    for enu, row in enumerate(M):
        Z[enu, :len(row)] += row

        Zmask[enu] = 1
    return Z, Zmask

#swap neighbors2 randomly, 50% are swapped
from random import randint
def perturbXmini(exampleXmini):
    numNeighbors = len(exampleXmini)
    swapIndices = [randint(0, 1) for i in range(0, numNeighbors) ]
    prevI = None
    for i, index in enumerate(swapIndices):
        if index==1 and prevI!=None:
            temp = exampleXmini[prevI]
            exampleXmini[prevI] = exampleXmini[i]
            exampleXmini[i] = temp
        if index ==1:
            prevI = i
    return exampleXmini

def redistribute_VarLenMini(data_All, batch_size):
    #shuffle & separate so that we can index into them later
    shuffle(data_All)

    dataX=[]
    dataXmini=[]
    dataXattr=[]
    dataY=[]
    dataID=[]
    for data in data_All:
        #print("======")
        #print(data[0])
        #for row in data[0]:
        #    print(len(row))
        dataX.append(data[0])
        dataXattr.append(data[1])
        dataXmini.append(data[2])
        dataY.append(data[4])
        dataID.append(data[5])

    #grab maxN2s per batch
    maxN2s = []
    for i in range(batch_size, len(dataX)+batch_size, batch_size):
        end = min(i, len(dataX))

        maxn2 = 0
        #grab max lens
        for exampleXmini in dataXmini[i-batch_size:end]:
            for neighbor in exampleXmini:
                if maxn2< len(neighbor):
                    maxn2=len(neighbor)
        maxN2s.append((maxn2, end))

    #preprocessing xmini data
    newXmini = []
    newXminiMask = []
    j=0; maxn2 = maxN2s[j][0];
    for i, exampleXmini in enumerate(dataXmini):
        if i>=maxN2s[j][1]:
            j+=1
            maxn2 = maxN2s[j][0]

        neighbors = []
        neighborsMask = []
        for neighbor in exampleXmini:
            #pad neighbors2 first, to obtain maxlen2 x attr size matrices, maxlen2 size masks
            neighbors2=[]
            for neighbor2 in neighbor:
                neighbors2.append(np.array(neighbor2, dtype=theano.config.floatX))
            n2Dense = pad_to_dense2(neighbors2, maxlen=maxn2, attrlen=len(neighbor2))
            neighbors.append( n2Dense[0])
            neighborsMask.append(n2Dense[1])
        newXmini.append(np.array(neighbors, dtype=theano.config.floatX))
        newXminiMask.append(np.array(neighborsMask, dtype=theano.config.floatX))

    #coordinate batches
    XYTuples = defaultdict(lambda:[])
    for i in range(batch_size, len(dataX)+batch_size, batch_size):
        end = min(i, len(dataX))
        batchX, batchXmask = pad_to_dense(dataX[i-batch_size:end])
        XYTuples['x'].append(np.transpose(batchX, (1, 0, 2)))
        XYTuples['xmask'].append(np.transpose(batchXmask, (1, 0)))

        batchXmini, batchXminimask = pad_to_dense(newXmini[i-batch_size:end], mask=newXminiMask[i-batch_size:end])

        XYTuples['xmini'].append(np.transpose(batchXmini, (1, 2, 0, 3)))
        XYTuples['xmini_mask'].append(np.transpose(batchXminimask, (1, 2, 0)))
        XYTuples['y'].append(np.array(dataY[i-batch_size:end], dtype=theano.config.floatX))
        XYTuples['xattr'].append(np.array(dataXattr[i-batch_size:end], dtype=theano.config.floatX))
        XYTuples['nodeID'].append(dataID[i-batch_size:end])
    return XYTuples


def encode_data_VarLenMiniHogun(G, nodes, attrKey, maxNeighbors, maxNeighbors2, useActualLabs=False, onlyLabs=False, nodeIDs=False, usePrevWeights=False, pageRankOrder="F", batch_size=100, perturb=False):
    X2AttrKey=attrKey
    if onlyLabs and useActualLabs:
        X2AttrKey='JustLabInfo_TrueLab'
    elif onlyLabs:
        X2AttrKey='JustLabInfo'
    elif useActualLabs:
        X2AttrKey='attr3'
    else:
        X2AttrKey=attrKey

    CountMissing = 0
    pprName='PPRweightsOrder'
    data_All=[]
    for node in nodes:
        exampleX = []
        exampleXattr = []
        exampleXmini = []
        exampleY = []
        #Add node to be predicted at beginning
        #exampleX.append(G.node[node][attrKey])
        exampleY=G.node[node]['label']
        for i, neighborSet in enumerate(G.node[node]['neighbors']):
            if i>maxNeighbors:
                break
            if 'attrWithPreds' in G.node[node] and usePrevWeights:
                exampleXattr = [0]
            else:
                exampleXattr = [0]

            #record how many neighbors appear more than once
            neighborDict = defaultdict(lambda: 0)
            for neighborsAtTime in neighborSet:
                neighborDict[neighborsAtTime]+=1
            #randomize neighbors
            dictKeys = neighborDict.keys()
            shuffle(dictKeys)
            j=-1
            #concatenate neighbors with # times neighbor appears
            for neighborsAtTime in dictKeys:
                if neighborsAtTime in G.node:
                    #if neighbors in this timestamp
                    if j==-1:
                        exampleX.append(exampleXattr)
                        exampleXmini.append([])

                    j+=1
                    if j>maxNeighbors2:
                        break

                    neighbor2Key = X2AttrKey
                    # if my neighbor of my neighbor is me
                    # and we are performing collective inference
                    # I want my attribute vector to still contain the prediction
                    if node==neighborsAtTime and "attrWithPreds" in G.node[node] and usePrevWeights:
                        neighbor2Key = 'attrWithPreds'
                    exampleXmini[-1].append(G.node[neighborsAtTime][neighbor2Key]+[neighborDict[neighborsAtTime]])
        if len(exampleX)==0:
            CountMissing+=1
            continue
        #if performing collective inference
        if 'attrWithPreds' in G.node[node] and usePrevWeights:
            exampleXattr = G.node[node]['attrWithPreds']
        else:
            exampleXattr = G.node[node][attrKey]

        if (perturb):
            exampleXmini = perturbXmini(exampleXmini)
        #x, xattr, xmini, orig attr, y, node
        data_All.append([np.array(exampleX, dtype=theano.config.floatX), exampleXattr, exampleXmini, G.node[node][attrKey], exampleY, node])
    #print(CountMissing)
    #print(len(data_All))
    return redistribute_VarLenMini(data_All, batch_size)

        
        

#this function is for encoding data for use in the RNNwMini
#max neighbors limits the number of neighbors a sample can have
#also keep track of node ID via dataID
def encode_data_VarLenMini(G, nodes, attrKey, maxNeighbors, maxNeighbors2, useActualLabs=False, onlyLabs=False, nodeIDs=False, usePrevWeights=False, pageRankOrder="F", batch_size=100, perturb=False):
    X2AttrKey=attrKey
    if onlyLabs and useActualLabs:
        X2AttrKey='JustLabInfo_TrueLab'
    elif onlyLabs:
        X2AttrKey='JustLabInfo'
    elif useActualLabs:
        X2AttrKey='attr3'
    else:
        X2AttrKey=attrKey

    pprName='PPRweightsOrder'
    #can only use onlyLabs when we are using inputx2
    #maybe we can
    #if onlyLabs and not useInputX2:
    #    print("onlyLabs=True only when useInputX2=True")
    #    sys.exit(0)

    #build dictionary of examples
    data_All=[]
    for node in nodes:
        exampleX = []
        exampleXattr = []
        exampleXmini = []
        exampleY = []
        #Add node to be predicted at beginning
        #exampleX.append(G.node[node][attrKey])
        exampleY=G.node[node]['label']
        lenNeighb = len(G.neighbors(node))

        #if we don't want order to matter
        if not isinstance(G, OrderedGraph):
            neighbors = G.neighbors(node)
            #randomly permute neighbors and then feed it in as examples
            shuffle(neighbors)
            for i, neighbor in enumerate(neighbors):
                if i>maxNeighbors:
                    break
                exampleX.append(G.node[neighbor][X2AttrKey])
                exampleXmini.append([])

                #encode rnn mini
                neighbors_mini = G.neighbors(neighbor)
                shuffle(neighbors_mini)
                for j, neighbor2 in enumerate(neighbors_mini):
                    if j>maxNeighbors2:
                        break
                    neighbor2Key = X2AttrKey
                    # if my neighbor of my neighbor is me
                    # and we are performing collective inference
                    # I want my attribute vector to still contain the prediction
                    if node==neighbor2 and "attrWithPreds" in G.node[node] and usePrevWeights:
                        neighbor2Key = 'attrWithPreds'
                    exampleXmini[-1].append(G.node[neighbor2][neighbor2Key])

                #exampleY.append(G.node[neighbor]['label'])
            #we add on a default attr vector consisting of 0.5s if we have no neighbors
            if len(G.neighbors(node))==0:
                exampleX.append([0.5]*len(G.node[node][X2AttrKey]))

            #if performing collective inference
            if 'attrWithPreds' in G.node[node] and usePrevWeights:
                exampleXattr = G.node[node]['attrWithPreds']
            else:
                exampleXattr = G.node[node][attrKey]

        if (perturb):
            exampleXmini = perturbXmini(exampleXmini)
        #x, xattr, xmini, orig attr, y, node
        data_All.append([np.array(exampleX, dtype=theano.config.floatX), exampleXattr, exampleXmini, G.node[node][attrKey], exampleY, node])

    return redistribute_VarLenMini(data_All)



    #returns xmini with dimension (batch_size, mlen1, mlen2, attrsize)
    #mask is                      (batch_size, mlen1, mlen2)
    #returns x with dimension (batch_size, mlen1, attrsize)
    #mask is                  (batch_size, mlen1)


    #xmini transposed
    #mlen1, mlen2, batch_size, attrsize

    #x transposed
    #mlen1, batch_size, attrsize

    return XYTuples