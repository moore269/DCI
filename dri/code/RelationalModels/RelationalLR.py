from RelationalRNN import *
from RelationalLSTM import *
from RelationalLSTMPAR import *
import cPickle as pickle
import readData
from sklearn.linear_model import LogisticRegression
from config import config
import graph_helper
# Load config parameters
locals().update(config)


def bStr(boolStr):
    if boolStr:
        return "T"
    else:
        return "F"

class RelationalLR(object):

    def __init__(self, rnnType, rnnCollective=False, modelInputPath="", GFirst=None, testNodes=None, 
        accuracyOutputName='', netType='', dynamLabel=False, avgNeighbLabel=False, PPRs=None, 
        bias_self="none", testLimit=False, trial=0, fold=0, onlyMakeData=False, **kwargs):
        self.modelInputPath=modelInputPath
        self.GFirst=GFirst
        self.attr = 'attr'

        #handle collective scenario, and modify G so that it has previous preds
        if rnnCollective:

            kwargs['attrKey'] = 'attr2'
            
            predHash = self.getPreds("Test_C", accuracyOutputName, netType, trial, fold)

            #set all nodes dynamic label back to original label
            for node in kwargs['G'].nodes():
                kwargs['G'].node[node]['dynamic_label'] = kwargs['G'].node[node]['label']
            for node in self.GFirst.nodes():
                self.GFirst.node[node]['dynamic_label'] = self.GFirst.node[node]['label'] 

            #now reset to predictions for test nodes
            for node in predHash:
                kwargs['G'].node[node]['dynamic_label'] = [predHash[node]]
                self.GFirst.node[node]['dynamic_label'] = [predHash[node]]

            kwargs['G'] = graph_helper.usePreviousPredictions(kwargs['G'], 'attr2', self.attr, dynamLabel=dynamLabel, avgNeighbLabel=avgNeighbLabel, pageRankOrder=pageRankOrder, PPRs=PPRs, maxNeighbors=maxNeighbors, bias_self=bias_self, testLimit=testLimit)
            
            self.modelInputPath=self.modelInputPath+"_RNNC_0"

        self.mainloop = load(open(self.modelInputPath+".pkl", 'rb'))
        self.model = self.mainloop.model

        #if logistic regression with aggregation features
        if "LRAG" in rnnType:
            graph_helper.usePreviousPredictions(self.GFirst, 'attr2', 'attr', proportions=True)
            self.attr='attr2'

        # construct network based on LSTMPAR or LSTM
        if "LSTMPAR" in rnnType:
            names = ['mlp_xpos_myinput_apply_output', 'mlp_xneg_myinput_apply_output']
            self.relationalOb = RelationalLSTMPAR(**kwargs)

        elif "LSTM" in rnnType:
            names=['h_to_o_apply_input_']
            self.relationalOb = RelationalLSTM(**kwargs)
        #else we don't want any LSTM input
        else:
            names=None
            self.relationalOb=None

        #finally, we can generate our training, validation, and testing data
        #trainDataLR is for LR
        self.trainDataLR = self.createData(kwargs['trainNodes']+kwargs['validationNodes'], kwargs['maxNeighbors'], names)
        
        #train, valid, and test
        self.trainData = self.createData(kwargs['trainNodes'], kwargs['maxNeighbors'], names)
        self.validData = self.createData(kwargs['validationNodes'], kwargs['maxNeighbors'], names)
        self.testData = self.createData(testNodes, kwargs['maxNeighbors'], names)
        if not onlyMakeData:
            self.lr = LogisticRegression()

    def getPreds(self, name, accuracyOutputName, netType, trial, fold):
        predPath = 'output/'+accuracyOutputName+"_predictions_"+netType + "_"+name+".npy"
        originalPreds = np.load(predPath)
        a,b,c,d = originalPreds.shape
        predHash={}
        #iterate through predictions
        for i in range(0, c):
            id=int(originalPreds[trial][fold][i][0])
            #0s means no more data
            if id==0:
                break
            predHash[id]=originalPreds[trial][fold][i][1]
        return predHash

    def train(self):  
        self.lr.fit(self.trainDataLR[0], self.trainDataLR[1])

    def predictBAE(self, testSet = "test", changeLabel=True):
        if testSet=="test":
            testData = self.testData
        elif testSet=="valid":
            testData = self.validData
        elif testSet=="train":
            testData = self.trainData

        err1=0.0
        err0=0.0
        count1=0
        count0=0
        predictions={}
        probs_posneg = self.lr.predict_proba(testData[0])
        for i in range(0, probs_posneg.shape[0]):
            pred = probs_posneg[i][1]
            actual = testData[1][i]
            nodeID = testData[2][i]
            predictions[nodeID]=pred
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
            return (1.0, predictions)

        BAE=(err1+err0)/divideBy

        return (BAE, predictions)


    def constructRNN(self, hiddenNames):
        batch_index_To, batch_index_From = self.model.inputs
        sharedData = self.relationalOb.sharedData
        output=[]
        sharedVarsTemp = []
        for var in self.model.variables:
            for hiddenName in hiddenNames:
                if var.name==hiddenName:
                    output.append(var)
            if var.name!=None and "sharedData" in var.name and var.name.replace("sharedData_", "") in sharedData:
                name = var.name.replace("sharedData_", "")
                var.set_value(sharedData[name].get_value(borrow=True), borrow=True)


        fRR = theano.function(inputs=[batch_index_From, batch_index_To], outputs=output)
        return fRR

    #names has to be empty or none if we don't want any LSTM features
    def createData(self, nodes, maxNeighbors, names):

        #if names none then we don't add LSTM features, else we do
        if names==None or len(names)==0:        
            examplesX=[]
            examplesY=[]
            examplesIDs=[]
            for node in nodes:
                examplesIDs.append(node)
                attrVector = self.GFirst.node[node][self.attr]
                examplesX.append(np.array(attrVector))
                examplesY.append(self.GFirst.node[node]['label'])
        else:
            self.relationalOb.replaceTestData(nodes, maxNeighbors =maxNeighbors)
            stream = self.relationalOb.stream_test_int 
            data = self.relationalOb.test_all

            fRR = self.constructRNN(names)

            examplesX=[]
            examplesY=[]
            examplesIDs=[]
            for i, batch in enumerate(stream.get_epoch_iterator(as_dict=True)):
                if names==None or len(names)==0:
                    outputs=[]
                else:
                    outputs = fRR(batch['int_stream_From'], batch['int_stream_To'])
                nodeID = data['nodeID_myinput'][i]
                examplesIDs.append(nodeID)
                attrVector = self.GFirst.node[nodeID][self.attr]
                outs = [out.flatten() for out in outputs]
                outs.append(attrVector)
                example = np.concatenate(outs, axis=0)

                examplesX.append(example)
                examplesY.append(self.GFirst.node[nodeID]['label'])

        examplesX = np.array(examplesX)
        examplesY = np.ravel(np.array(examplesY))
        return (examplesX, examplesY, examplesIDs)



def unitTest1():
 
    accuracyOutputName=""
    fName="facebook"
    netType="LSTM"
    memory=5
    bias_self="none"
    testLimit=False
    summary_dim=3
    PRType="neutral"
    generateUnordered=1
    useActualLabs=False
    onlyLabs=False
    avgNeighb=False
    degree = True
    neighbDegree=False
    avgNeighbLabel=False
    dynamLabel=True
    #avgPosNegAttr=False
    #num01s=False
    localClustering=False
    
    noValid=False
    singlePR=False

    attr1='attr'

    i=0
    j=3
    accuracyOutputName=accuracyOutputName+netType+"_"+fName+"_noVal_"+str(bStr(noValid))+"_Mem_"+str(memory)+"_Ex_"+str(generateUnordered)+"_MultTra_"+str(bStr(multipleCollectiveTraining))+"_maxEpoch_"+str(max_epochs)+"_maxNProp_"+str(maxNProp)+"_trials_"+str(trials)+"_selectFolds_"+str(bStr(onlySelectFolds))+"_avgNeigLab_"+str(bStr(avgNeighbLabel))+"_dynamLab_"+str(bStr(dynamLabel))+"_useActLabs_"+str(bStr(useActualLabs))+"_onlyLabs_"+str(bStr(onlyLabs))+"_PPR_"+str(bStr(pageRankOrder)) + "_onlyPR_"+str(bStr(singlePR))+"_prtype_"+PRType + "_biasself_" + bias_self + "_d_"+str(bStr(degree) + "_lim_"+str(bStr(testLimit)))+"_sd_"+str(summary_dim)
    print(accuracyOutputName)
    save_path=save_path_prefix+accuracyOutputName
    actual_save_path = save_path+"_trial_"+str(i)+"_fold_"+str(j)
    #read trial from file
    rest, validationNodes= readData.readTrial(fName, i, percentValidation)

    #split into folds
    folds= readData.splitNodeFolds(rest, numFolds)
    trainNodes=[] 
    for k in range(0, j+1):
        trainNodes+=folds[k]
    testNodes=[]
    for k in range(j+1, numFolds):
        testNodes+=folds[k]

    G = readData.readDataset(dataFolder, fName, sampleAttrs=sampleAttrs, averageNeighborAttr=avgNeighb, degree=degree, neighbDegree=neighbDegree, 
        localClustering=localClustering) 
    GFirst=G
    PPRs = pickle.load(open(dataFolder+fName+"_10pr_"+PRType+"_trial_"+str(i)+"_fold_"+str(j)+".p", 'rb'))
    G = readData.readDataset(dataFolder, fName, sampleAttrs=sampleAttrs, averageNeighborAttr=avgNeighb, degree=degree, neighbDegree=neighbDegree, 
        localClustering=localClustering, pageRankOrder=pageRankOrder, PPRs=PPRs, maxNeighbors=maxNeighbors, bias_self=bias_self, 
        trainNodes=trainNodes+validationNodes, testNodes=testNodes, testLimit=testLimit)
    if degree:
        graph_helper.transferAttr(GFirst, G, 'degree')
    """r = RelationalLR("LSTM_LRAG", rnnCollective=False, modelInputPath=actual_save_path,
        G=G, trainNodes=trainNodes, validationNodes=validationNodes, dim=memory, 
        maxNeighbors=maxNeighbors, attrKey=attr1, debug=debug, 
        generateUnordered=generateUnordered, epsilon=epsilon)"""

    r = RelationalLR("LRAG_"+netType, rnnCollective=True, modelInputPath=actual_save_path, GFirst=GFirst, testNodes=testNodes, 
        accuracyOutputName=accuracyOutputName, netType=netType, dynamLabel=dynamLabel, avgNeighbLabel=avgNeighbLabel,
        PPRs=PPRs, bias_self=bias_self, testLimit=testLimit, trial=i, fold=j,
        G=G, trainNodes=trainNodes, validationNodes=validationNodes, dim=memory, 
        batch_size=1, num_epochs=num_epochs, save_path=actual_save_path, 
        max_epochs=max_epochs, maxNeighbors=maxNeighbors, attrKey=attr1, debug=debug, 
        generateUnordered=generateUnordered, epsilon=epsilon)
    r.train()
    baeTra = r.predictBAE("train")
    baeValid = r.predictBAE("valid")
    baeTest = r.predictBAE("test")
    print(baeTra[0])
    print(baeValid[0])
    print(baeTest[0])


    print("no error")    

if __name__ == "__main__":
    unitTest1()