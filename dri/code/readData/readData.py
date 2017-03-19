import networkx as nx
import sys
from collections import defaultdict
from operator import itemgetter
import random
import numpy as np
from operator import add
import theano
from random import shuffle
import copy
import math
from collections import OrderedDict
import os
from code.graph_helper.OrderedGraph import *
from augmentations import *

#split "nodes" into two lists. An individual node has a "percent" chance of appearing in the first list
def splitNodes(nodes,trainPercent, valPercent):
    group1=[]
    group2=[]
    group3=[]
    shuffle(nodes)
    valNext=trainPercent+valPercent
    for node in nodes:
        rand=random.random()
        if rand < trainPercent:
            group1.append(node)
        elif rand> trainPercent and rand< valNext:
            group2.append(node)
        elif rand>valNext:
            group3.append(node)
            
    return (group1,group2, group3)


#use to split test nodes
def splitNodeFolds(nodes, numFolds):
    numFolds = int(numFolds)
    cuts=int(math.ceil(float(len(nodes))/numFolds))
    folds = []
    for i in xrange(0, len(nodes), cuts):
        folds.append(nodes[i:min(i+cuts, len(nodes))])
        
    return folds

#how far away are we from a clique?
#count number of links between my neighbors
#divide by possible number of links
def computeLocalClustering(G, node):

    lenNeighbors=len(G.neighbors(node))
    if lenNeighbors < 2:
        return 1.0

    #put neighbors into hash table
    neighborsHash={}
    for neighbor in G.neighbors(node):
        neighborsHash[neighbor]=1

    count=0
    for neighbor in G.neighbors(node):
        for neighbor2s in G.neighbors(neighbor):
            if neighbor2s in neighborsHash:
                count+=1

    return float(count)/(lenNeighbors*(lenNeighbors-1))


# helper method for sampling attributes in readDataset
# default behavior for just reading all attributes is to have sampleAttrs=0
def sampleAttributes(G, sampleAttrs, attrs):
    first=True
    attributeIndices=None
    for line in attrs:
        attributes=line.replace("\n", "").split("::")
        x=int(attributes[0])
        G.add_node(x)
        G.node[x]['attr']=[]

        if(first):
            attributeIndices = range(1, len(attributes))
            shuffle(attributeIndices)
            first=False

        nodeAttributes=[]
        for i in range(0, len(attributes)-1):
            nodeAttributes.append(float(attributes[attributeIndices[i]]))

        if sampleAttrs==0:
            G.node[x]['attr']=nodeAttributes
        else:
            for i in range(0, min(sampleAttrs, len(nodeAttributes))):
                G.node[x]['attr'].append(nodeAttributes[i])
        G.node[x]['attr0'] = copy.deepcopy(G.node[x]['attr'])

# helper method for selecting attributes in readDataset
def selectAttributes(G, selectAttrIndices, attrs):

    for line in attrs:
        attributes=line.replace("\n", "").split("::")
        x=int(attributes[0])
        G.add_node(x)
        G.node[x]['attr']=[]

        nodeAttributes=[]
        for i in range(0, len(selectAttrIndices)):
            nodeAttributes.append(int(attributes[selectAttrIndices[i]]))

        G.node[x]['attr']=nodeAttributes

#helper method to add edges
def addEdges(G, links):
    for line in links:
        nodePair= line.replace("\n", "").split("::")
        x=int(nodePair[0])
        y=int(nodePair[1])
        G.add_edge(x, y, weight=1.0)

# helper method to add edges that match a label
def addLabEdges(G, node, pprNeighbors, labelName, lab, pprName, appendSelf=True):
    neighbors=[]
    #always append self
    if appendSelf:
        neighbors.append(pprNeighbors[0])
    countNeighb=0
    for neighbor in pprNeighbors[1:]:

        try:
            G.node[neighbor[0]]
        except KeyError:
            continue

        #if unknown, skip
        if G.node[neighbor[0]][labelName][0]==None:
            continue
        labThreshold = 1.0 if G.node[neighbor[0]][labelName][0]>=0.5 else 0.0
        if (lab == labThreshold):
            countNeighb+=1
            neighbors.append(neighbor)
        if countNeighb>=10:
            break   
    G.node[node][pprName] = neighbors[::-1] 

# helper method of a helper method to add edges according to yourself or not yourself
def addSelfEdges(G, node, pprNeighbors, labelName, bias_self, pprName):
    neighbors=[]
    #always append self
    neighbors.append(pprNeighbors[0])
    countNeighb=0
    nodeThreshold = 1.0 if G.node[node][labelName][0] >=0.5 else 0.0
    for neighbor in pprNeighbors[1:]:
        try:
            G.node[neighbor[0]]
        except KeyError:
            continue

        #if unknown, skip
        if G.node[neighbor[0]][labelName][0]==None or  G.node[node][labelName][0]==None:
            continue
        labThreshold = 1.0 if G.node[neighbor[0]][labelName][0]>=0.5 else 0.0
        if (bias_self=="self" and nodeThreshold == labThreshold) \
            or (bias_self=="notself" and nodeThreshold!= labThreshold):
            countNeighb+=1
            neighbors.append(neighbor)
        if countNeighb>=10:
            break

    G.node[node][pprName] = neighbors[::-1]

#helper method for adding edges with no preference and in the general LSTM neutral model
def addNoneEdges(G, node, pprNeighbors, maxNeighbors, pprName):
    neighbors = []
    skipped=0
    for neighbor in pprNeighbors:
        try:
            G.node[neighbor[0]]
            neighbors.append(neighbor)
        except KeyError:
            skipped+=1
            #print(skipped)

        if len(neighbors) >=maxNeighbors:
            break
    G.node[node][pprName] = neighbors[::-1]


#helper method to add edges for PR
#right now just adds lists to each node
def addEdgesPR(G, PPRs, maxNeighbors, bias_self="", testLimit=False):
    if bias_self!="none" and bias_self!="self" and bias_self!="notself" and bias_self!="pos" and bias_self!="neg" and bias_self!="posneg":
        print("bias_self must be equal to either "", 'self', or 'notself'")
        sys.exit(0)

    if testLimit==True:
        labelName = 'label'
    else:
        labelName = 'dynamic_label'

    pprName='PPRweightsOrder'

    neighborsLenCount=0
    for node in G.nodes():
        G.node[node][pprName]=[]
        G.node[node]['blank']=[]

        if bias_self=="none":
            addNoneEdges(G, node, PPRs[node], maxNeighbors, pprName)

        #if we want to bias purely positve or purely negative
        elif bias_self=="pos":
            addLabEdges(G, node, PPRs[node], labelName, 1, pprName)
        elif bias_self=="neg":
            addLabEdges(G, node, PPRs[node], labelName, 0, pprName)
        #add both
        elif bias_self=="posneg":
            addLabEdges(G, node, PPRs[node], labelName, 1, pprName+"_pos", appendSelf=False)
            addLabEdges(G, node, PPRs[node], labelName, 0, pprName+"_neg", appendSelf=False) 
        #if we want to bias neighbors with same or not same label to ourselves
        elif bias_self=="self" or bias_self=="notself":
            addSelfEdges(G, node, PPRs[node], labelName, bias_self, pprName)
        else:
            print("we got a problem, captain. this statement should be impossible to get to")
            sys.exit(0)

# returns a graph dataset with attr1 - just attributes
#                                 attr2 - attributes + label
#                                label - the label
# sampleAttrs is number of sampled attributes
# if sampleAttrs=0, don't do any sampling
# selectAttrIndices means we select specific attributes
# if selectedAttrIndices=[], we don't select any specific attributes
# selectAttrIndices is index from 1
# you cannot have both sampleAttrs and selectedAttributes on.
# To read PR and trial/fold data, please provide pageRankOrder=False, foldNum=0, trialNum
def readDataset(dataFolder, fName, sampleAttrs=0, selectAttrIndices=[], knownIndices=None, unknownIndices=None, averageNeighborAttr=False, degree=False, neighbDegree=False, localClustering=False, pageRankOrder="F", PPRs=None, maxNeighbors=10, bias_self="", trainNodes=None, testNodes=None, testLimit=False, no0Deg=False):

    #if train and test nodes are provided, set them for O(1) access
    if trainNodes!=None and testNodes!=None:
        trainNodesSet = set(trainNodes)
        testNodesSet = set(testNodes)

    links=open(dataFolder+fName+".edges", 'r')
    attrs=open(dataFolder+fName+".attr", 'r')
    labels=open(dataFolder+fName+".lab", 'r')

    #if we are not inducing page rank, do regular undirected graph
    if pageRankOrder=="for" or pageRankOrder=="back":
        G=OrderedGraph()
    else:
        G=nx.Graph()
    
    #create nodes
    if len(selectAttrIndices)==0:
        sampleAttributes(G, sampleAttrs, attrs)
    else:
        selectAttributes(G, selectAttrIndices, attrs)

    # allocate labels
    for line in labels:
        labs=line.replace("\n", "").split("::")
        node = int(labs[0])
        lab=int(labs[1])
        if node in G.node:
            G.node[node]['label']=[lab]

            if pageRankOrder=="for" or pageRankOrder=="back":
                #only append known labels to dynamic label
                if node in trainNodesSet:
                    G.node[node]['dynamic_label'] = [float(lab)]
                elif node in testNodesSet:
                    G.node[node]['dynamic_label']=[None]
            else:
                G.node[node]['dynamic_label']=[float(lab)]

    #add edges between them, this depends on the page rank order
    if pageRankOrder=="for" or pageRankOrder=="back":
        addEdgesPR(G, PPRs, maxNeighbors, bias_self=bias_self, testLimit=testLimit)
    else:
        addEdges(G, links)

    for node in G.nodes():
        G.node[node]['degree'] = len(G.neighbors(node))


    if no0Deg:
        for node in G.nodes():
            #prune out deg=0 when reading normally
            if not (pageRankOrder=="for" or pageRankOrder=="back") and G.node[node]['degree']==0:
                G.remove_node(node)
            # if reading in for pagerank order and checking if node is in our train/test sets 
            if (pageRankOrder=="for" or pageRankOrder=="back") and not (node in trainNodesSet or node in testNodesSet):
                G.remove_node(node)

    for node in G.nodes():
        G.node[node]['clustering']=computeLocalClustering(G, node)



    for node in G.nodes():
        avgDegree=0.0
        for neighbor in G.neighbors(node):
            avgDegree += G.node[neighbor]['degree']

        if(G.node[node]['degree']==0):
            G.node[node]['avgNeighbDegree']=0.0
        else:
            G.node[node]['avgNeighbDegree']=avgDegree/G.node[node]['degree']


    #take average attributes of neighbors
    #iterate through all nodes and record averages
    for node in G.nodes():
        nodeAttrVec=G.node[node]['attr']
        #go through each neighbor of each node, add their attributes
        avgAttr=[0.0]*len(nodeAttrVec)
        if len(G.neighbors(node))>0:
            for neighb in G.neighbors(node):
                avgAttr=map(add, avgAttr, G.node[neighb]['attr'])
            #take average and record it
            avgAttr = [x/len(G.neighbors(node)) for x in avgAttr]
        else:
            avgAttr=[0.5]*len(nodeAttrVec)
        G.node[node]['avgAttr']=avgAttr

    if(degree):
        for node in G.nodes():
            G.node[node]['attr'].append(G.node[node]['degree'])

    if(neighbDegree):
        for node in G.nodes():
            G.node[node]['attr'].append(G.node[node]['avgNeighbDegree'])

    if(localClustering):
        for node in G.nodes():
            G.node[node]['attr'].append(G.node[node]['clustering'])

    #add to attr vector if option is true
    if averageNeighborAttr:
        #iterate through all nodes and assign averages
        for node in G.nodes():
            G.node[node]['attr']=G.node[node]['attr']+G.node[node]['avgAttr']

    #create a full vector
    for node in G.nodes():
        G.node[node]['full']=list(G.node[node]['attr'])
        G.node[node]['full'] = G.node[node]['full'] + G.node[node]['label']

    return G

#read just attributes
def readDatasetAttr(linksName, attrName, knownIndices=None, unknownIndices=None):
    G=nx.Graph()
    links=open(linksName, 'r')
    attrs=open(attrName, 'r')

    for line in links:
        nodePair= line.replace("\n", "").split("::")
        x=int(nodePair[0])
        y=int(nodePair[1])
        G.add_edge(x, y)

    for line in attrs:
        attributes=line.replace("\n", "").split("::")
        x=int(attributes[0])
        G.add_node(x)
        G.node[x]['attr']=[]
        for i in range(1, len(attributes)):
            G.node[x]['attr'].append(int(attributes[i]))

    if(knownIndices!=None and unknownIndices!=None):
        for node in G.nodes():
            getKnown =itemgetter(*knownIndices)
            getUnknown =itemgetter(*unknownIndices)
            if str(type(gK)) == "<type 'tuple'>":
                G.node[x]['attr']=list(gK)
            else:
                G.node[x]['attr']=[gK]
            if str(type(gU)) == "<type 'tuple'>":
                G.node[x]['label']=list(gU)
                G.node[x]['dynamic_label']=list(gU)
            else:
                G.node[x]['label']=[gU]
                G.node[x]['dynamic_label']=[gU]

    return G

import pickle

def readHogun(dataFolder, fName):
    f_t_graph  = dataFolder+fName+".p"
    x = pickle.load( open( f_t_graph, "rb" ) )
    nodes = x[0] ; neighborTimeSeries = x[1] ; labels = x[2] ; features = x[3] ;
    G = nx.Graph()
    for i in range(0, len(nodes)):
        x = nodes[i]
        G.add_node(x)
        G.node[x]['label'] = [labels[i]]
        G.node[x]['dynamic_label'] = [float(labels[i])]
        if type(features[i]) is np.ndarray:
            G.node[x]['attr'] = features[i].tolist()
        else:
            G.node[x]['attr'] = features[i].toarray().tolist()[0]
        G.node[x]['neighbors'] = neighborTimeSeries[i]

    return G



def main():
    G = readDataset("data/amazon_DVD_20000.edges","data/amazon_DVD_20000.attr","data/amazon_DVD_20000.lab")
    #train, valid, test = splitNodes(G.nodes(),0.7, 0.2)


#saves each trials
def generateTrials(fName, trials, percentValidation):
    print(fName)
    #dataFolder = "data/"
    #G = readDataset(dataFolder, fName) 
    dataFolder = "../../experiments/data/"
    G = readHogun(dataFolder, fName)
    for i in range(0, trials):
        #get random partitioning
        (validationNodes, testNodes, rest) = splitNodes(G.nodes(),percentValidation, 0.0)
        #shuffle(validationNodes)
        #shuffle(testNodes)
        shuffle(rest)
        npArr = np.array([rest, validationNodes], dtype='object')
        np.save(dataFolder+fName+"_trial_"+str(i)+"_val_"+str(percentValidation), npArr)

#need to pass in trial # and percentValidation
def readTrial(dataFolder, fName, i, percentValidation, changeTrainValid=0):
    trial = np.load(dataFolder+fName+"_trial_"+str(i)+"_val_"+str(percentValidation)+".npy")
    rest, valNodes=trial.tolist()
    if changeTrainValid<4:
        return (rest, valNodes)
    else:
        percent3 = int(len(valNodes)*0.2)
        valNodes12 = valNodes[:percent3]
        valNodes3 = valNodes[percent3:]
        return (rest, valNodes12, valNodes3)

def outputProportions(fName):
    #G = readDataset("data/", fName)
    dataFolder = "../../experiments/data/"
    G = readHogun(dataFolder, fName)
    pos = 0
    neg = 0
    lenNodes = 0
    lenEdges = len(G.edges())
    for node in G.nodes():
        degree = len(G.neighbors(node))
        if degree>-1:
            lenNodes+=1
            if G.node[node]['label'][0] == 1:
                pos+=1
            else:
                neg+=1
    print(fName)
    print("numNodes:" + str(lenNodes))
    print("lenEdges:"+str(lenEdges))
    print("pos: "+str(float(pos)/lenNodes))
    print("neg: "+str(float(neg)/lenNodes))

if __name__ == "__main__":
    fName="facebook"
    ##pageRankOrder=False
    #foldNum=0
    #trialNum=0
    #fNamePR="facebook_10pr_neg_trial_0_fold_0.p"
    #G = readDataset(fName=fName, pageRankOrder=pageRankOrder, foldNum=foldNum, trialNum=trialNum, fNamePR=fNamePR)
    #print(len(G.edges()))
    outputProportions("facebook")
    #outputProportions("amazon_DVD_20000")
    #outputProportions("amazon_DVD_7500")
    #readHogun("../../experiments/data/", "preprocessed_facebook.p")
    #generateTrials("preprocessed_facebook_filtered", 10, 0.1)
    #generateTrials("preprocessed_imdb", 10, 0.1)
    #generateTrials("preprocessed_reality_mining", 10, 0.15)
    #generateTrials("preprocessed_dblp", 10, 0.15)   
    
    #outputProportions("preprocessed_facebook")
    #outputProportions("amazon_Music_7500")
    #outputProportions("IMDB_5")
    #generateTrials("IMDB_5", 10, 0.15)
    #generateTrials("facebook", 10, 0.15)
    #generateTrials("amazon_DVD_20000", 10, 0.15)
    #generateTrials("amazon_Music_10000", 10, 0.15)
    #generateTrials("amazon_DVD_7500", 10, 0.15)
    #generateTrials("amazon_Music_7500", 10, 0.15)
    #numFolds=3
    #for i in range(0, 10):
    #    rest, valNodes = readTrial("amazon_DVD_20000", i, 0.15)
    #    folds=splitNodeFolds(rest, numFolds)
    #python rnnExperiments.py LSTM 10 neutral 0 amazon_DVD_20000 none 1 0 3 1 0 swap:q
