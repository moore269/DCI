import random
from operator import add
import numpy as np
from pylab import *
from OrderedGraph import *
import code.readData.readData as readData
from collections import defaultdict

#helper function to predict label iff changeTrainValid==3
def predictLabel(prediction, changeTrainValid):
    if changeTrainValid==3 or changeTrainValid==5:
        if prediction>=0.5:
            return 1.0
        else:
            return 0.0
    else:
        return prediction

#transfer attributes from one graph to another
def transferAttr(GFirst, G, attr):
    for node in GFirst.nodes():
        G.node[node][attr] = GFirst.node[node][attr]
        G.node[node]['attr'].append(GFirst.node[node][attr])

#prune out 0s
def prune0s(GFirst, nodes):
    newNodes=[]
    for node in nodes:
        try:
            GFirst.node[node]
            newNodes.append(node)
        except KeyError:
            continue
    return newNodes

#use dynamic label or previous predictions and add to attributes of each node
#train sets dynamic label should be same as label
#validation set should have a changing dynamic label
#testing set should have a changing dynamic label
#encode actual predictions in attr3
#encode approximate predictions in attr2
#useful when we want to see what happens when we have true labels

def usePreviousPredictions(G, attr2, attr1, attr3='attr3', dynamLabel=False, avgNeighbLabel=False, attrLabs='JustLabInfo', pageRankOrder=False, PPRs=None, maxNeighbors=0, bias_self="", testLimit=False, proportions=False, lastHs=None, lastH=False, dim=10, changeTrainValid=0):
    if lastH:
        if lastHs==None:
            for node in G.nodes():
                G.node[node]['lastH'] = [0]*dim
        else:
            for node in G.nodes():
                G.node[node]['lastH'] = lastHs[node]

    # switch around the order
    # only switch when not having testLimit (or optimal ordering)
    if (pageRankOrder=="for" or pageRankOrder=="back") and not testLimit:
        Gold = G
        G = OrderedGraph()
        G.add_nodes_from(Gold.nodes(data=True))
        readData.addEdgesPR(G, PPRs, maxNeighbors, bias_self, testLimit)

    for node in G.nodes():     
        averageSelected(G, node, attr2, attr1, avgNeighbLabel=avgNeighbLabel, avgPosNegAttr=False, num01s=False, proportions=proportions, dynamLabel=dynamLabel, attrLabs=attrLabs, changeTrainValid=changeTrainValid)
    return G

#average list of labels using neighbors
def averageNeighbors(G, attrNewKey, attrOldKey, avgNeighbLabel = True, avgPosNegAttr=True, num01s=True, attrLabs='JustLabInfo', changeTrainValid=0):
    for node in G.nodes():
        averageSelected(G, node, attrNewKey, attrOldKey, avgNeighbLabel, avgPosNegAttr, num01s, attrLabs=attrLabs, changeTrainValid=changeTrainValid)

#attr3 is our hidden attr vector that also has the true labels
def averageSelected(G, node, attrNewKey, attrOldKey, avgNeighbLabel, avgPosNegAttr, num01s, proportions=False, dynamLabel=False, attrLabs='JustLabInfo', changeTrainValid=0):
    #calculate our new attributes
    if avgNeighbLabel:
        averageNeighborLabel(G, node, changeTrainValid)
    if avgPosNegAttr or num01s or proportions:
        averagePosNegAttr(G, node, attrOldKey)

    #if we didn't get predictions, set default to 1
    if G.node[node]['dynamic_label'][0]==None or math.isnan(G.node[node]['dynamic_label'][0]):
        G.node[node]['dynamic_label']=[1.0]

    dynPred = G.node[node]['dynamic_label'][0]
    #encode 
    trueAttrLabs=attrLabs+"_TrueLab"
    #now select them
    G.node[node][attrNewKey]=G.node[node][attrOldKey]
    G.node[node]['attr3']=G.node[node][attrOldKey]
    G.node[node]['attrWithPreds']=G.node[node][attrOldKey]
    G.node[node][attrLabs]=[]
    G.node[node][trueAttrLabs]=[]
    if dynamLabel:
        G.node[node][attrNewKey]=G.node[node][attrNewKey]+ [predictLabel(dynPred, changeTrainValid)]
        G.node[node]['attr3'] = G.node[node]['attr3']+G.node[node]['label']
        G.node[node][trueAttrLabs]=G.node[node][trueAttrLabs]+G.node[node]['label']
        G.node[node][attrLabs]=G.node[node][attrLabs]+[predictLabel(dynPred, changeTrainValid)]
        if 'pred_label' in G.node[node]:
            G.node[node]['attrWithPreds'] = G.node[node]['attrWithPreds']+[predictLabel(G.node[node]['pred_label'][0], changeTrainValid)]
    if avgNeighbLabel:
        G.node[node][attrNewKey]=G.node[node][attrNewKey]+G.node[node]['avgNeighborsLabel']
        G.node[node]['attr3'] = G.node[node]['attr3']+G.node[node]['avgNeighborsLabel'] 
        G.node[node]['attrWithPreds'] = G.node[node]['attrWithPreds']+G.node[node]['avgNeighborsLabel'] 
        G.node[node][attrLabs]=G.node[node][attrLabs]+G.node[node]['avgNeighborsLabel']
        G.node[node][trueAttrLabs]=G.node[node][trueAttrLabs]+G.node[node]['avgNeighborsLabel']
    if avgPosNegAttr:
        G.node[node][attrNewKey]=G.node[node][attrNewKey]+G.node[node]['avgNeighborsNegAttr']
        G.node[node][attrNewKey]=G.node[node][attrNewKey]+G.node[node]['avgNeighborsPosAttr']
        G.node[node]['attr3'] = G.node[node]['attr3']+G.node[node]['avgNeighborsNegAttr'] 
        G.node[node]['attr3'] = G.node[node]['attr3']+G.node[node]['avgNeighborsPosAttr']
        G.node[node]['attrWithPreds'] = G.node[node]['attrWithPreds']+G.node[node]['avgNeighborsNegAttr'] 
        G.node[node]['attrWithPreds'] = G.node[node]['attrWithPreds']+G.node[node]['avgNeighborsPosAttr']

        #attrLabs and true attr labs
        G.node[node][attrLabs]=G.node[node][attrLabs]+G.node[node]['avgNeighborsNegAttr']
        G.node[node][attrLabs]=G.node[node][attrLabs]+G.node[node]['avgNeighborsPosAttr']
        G.node[node][trueAttrLabs]=G.node[node][trueAttrLabs]+G.node[node]['avgNeighborsNegAttr']
        G.node[node][trueAttrLabs]=G.node[node][trueAttrLabs]+G.node[node]['avgNeighborsPosAttr']

    if num01s:
        G.node[node][attrNewKey]=G.node[node][attrNewKey]+G.node[node]['num0s']
        G.node[node][attrNewKey]=G.node[node][attrNewKey]+G.node[node]['num1s']
        G.node[node]['attr3'] = G.node[node]['attr3']+G.node[node]['num0s']
        G.node[node]['attr3'] = G.node[node]['attr3']+G.node[node]['num1s'] 
        G.node[node]['attrWithPreds'] = G.node[node]['attrWithPreds']+G.node[node]['num0s']
        G.node[node]['attrWithPreds'] = G.node[node]['attrWithPreds']+G.node[node]['num1s'] 

        #attrLabs and true attr labs
        G.node[node][attrLabs]=G.node[node][attrLabs]+G.node[node]['num0s']
        G.node[node][attrLabs]=G.node[node][attrLabs]+G.node[node]['num1s']
        G.node[node][trueAttrLabs]=G.node[node][trueAttrLabs]+G.node[node]['num0s']
        G.node[node][trueAttrLabs]=G.node[node][trueAttrLabs]+G.node[node]['num1s']
    if proportions:
        G.node[node][attrNewKey]=G.node[node][attrNewKey]+G.node[node]['negPro']
        G.node[node][attrNewKey]=G.node[node][attrNewKey]+G.node[node]['posPro']
        G.node[node]['attr3'] = G.node[node]['attr3']+G.node[node]['negPro']
        G.node[node]['attr3'] = G.node[node]['attr3']+G.node[node]['posPro'] 
        G.node[node]['attrWithPreds'] = G.node[node]['attrWithPreds']+G.node[node]['negPro']
        G.node[node]['attrWithPreds'] = G.node[node]['attrWithPreds']+G.node[node]['posPro'] 

        #attrLabs and true attr labs
        G.node[node][attrLabs]=G.node[node][attrLabs]+G.node[node]['negPro']
        G.node[node][attrLabs]=G.node[node][attrLabs]+G.node[node]['posPro']
        G.node[node][trueAttrLabs]=G.node[node][trueAttrLabs]+G.node[node]['negPro']
        G.node[node][trueAttrLabs]=G.node[node][trueAttrLabs]+G.node[node]['posPro']


#average list of labels using neighbors
def averageNeighborLabel(G, node, changeTrainValid):
    dynPred = G.node[node]['dynamic_label'][0]
    avgs = [0]*len(G.node[node]['dynamic_label'])
    for neighbor in G.neighbors(node):     
        avgs = map(add, avgs, [predictLabel(dynPred, changeTrainValid)])

    deg = G.degree(node)
    for i in range(0, len(avgs)):
        if(deg!=0):
            avgs[i]=float(avgs[i])/deg
        else:
            avgs[i]=0.5
    G.node[node]['avgNeighborsLabel']=avgs


#take each of our neighbors and average their attributes based on their dynamic label
def averagePosNegAttr(G, node, attrOldKey):
    avgsNeg = [0]*len(G.node[node][attrOldKey])
    avgsPos = [0]*len(G.node[node][attrOldKey])
    negDegree=0
    posDegree=0
    for neighbor in G.neighbors(node):
        if G.node[neighbor]['dynamic_label']==[0]:
            avgsNeg = map(add, avgsNeg, G.node[neighbor][attrOldKey])
            negDegree+=1
        elif G.node[neighbor]['dynamic_label']==[1]:
            avgsPos = map(add, avgsPos, G.node[neighbor][attrOldKey])
            posDegree+=1
    #average by dividing by degree
    for i in range(0, len(avgsNeg)):
        if(negDegree!=0):
            avgsNeg[i]=float(avgsNeg[i])/negDegree
        else:
            avgsNeg[i]=0.5
    #average by dividing by degree
    for i in range(0, len(avgsPos)):
        if(posDegree!=0):
            avgsPos[i]=float(avgsPos[i])/posDegree
        else:
            avgsPos[i]=0.5
    G.node[node]['avgNeighborsNegAttr']=avgsNeg
    G.node[node]['avgNeighborsPosAttr']=avgsPos
    G.node[node]['num0s']=[negDegree]
    G.node[node]['num1s']=[posDegree]
    degree = posDegree+negDegree
    G.node[node]['negPro']=[float(negDegree)/degree if degree!=0 else 0.0]
    G.node[node]['posPro']=[float(posDegree)/degree if degree!=0 else 0.0]

#resets all dynamic labels back to labels       
def resetDynamicLabels(G):
    for node in G.nodes():
        G.node[node]['dynamic_label']=G.node[node]['label']

def plotBAE(accuracies, name, t):
    fig=figure()
    ax = subplot(111)
    for key in accuracies:
        s = np.mean(accuracies[key], axis=0)
        stds=np.std(accuracies[key], axis=0)
        ax.errorbar(t, s,  yerr=stds, label=key)

    xlabel(name)
    ylabel('BAE')
    box=ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, fancybox=True, shadow=True)
    #title(name)
    savefig("output/"+name+".png")

#print accuracies for NP
def printNPBAE(accuracies, outName):
    for key in accuracies:
        np.save("output/"+outName+"_"+key, accuracies[key])

#given path, trial, fold, & name
#return dicts corresponding to train, valid, & test
#
def getPreds(path, trial, fold, name):

    sets={"Train": None, "Valid": None, "Test": None, "Test_C":None}
    for setType in sets.keys():
        predHash={}
        preds=np.load(path+name+setType+".npy")
        #shape is trials, folds, id pred pairs, id or pred
        a,b,c,d=preds.shape
        #print(preds.shape)

        #iterate through predictions
        for i in range(0, c):
            id=int(preds[trial][fold][i][0])
            #0s means no more data
            if id==0:
                break
            predHash[id]=preds[trial][fold][i][1]

        sets[setType]=predHash
    return sets

#given G and a set of nodes, return postive and negative label proportions
def getLabelCounts(G, nodes):
    labProportions = defaultdict(lambda:0)
    for node in nodes:
        lab = G.node[node]['label'][0]
        labProportions[lab]+=1
    return labProportions




import random
#For trainNodes, randomly set pred_label
#For testNodes, randomly set both dynamic_label and pred_label
def setLabels(G, trainNodes, validationNodes, testNodes, changeTrainValid):
    for node in trainNodes:
        predLab = random.random()
        G.node[node]['pred_label'] = [predictLabel(predLab, changeTrainValid)]
        if changeTrainValid < 0:
            G.node[node]['dynamic_label'] = [predictLabel(predLab, changeTrainValid)]

    for node in validationNodes:
        predLab = random.random()
        G.node[node]['pred_label'] = [predictLabel(predLab, changeTrainValid)]
        if changeTrainValid <0 :
            G.node[node]['dynamic_label'] = [predictLabel(predLab, changeTrainValid)] 

    for node in testNodes:
        predLab = random.random()
        G.node[node]['dynamic_label'] = [predictLabel(predLab, changeTrainValid)]
        G.node[node]['pred_label'] = [predictLabel(predLab, changeTrainValid)]        


