import os
config = {}

#how many processes to run in parallel
config['numProcesses'] = 1
config['trials'] = 10
config['percentValidation'] = 0.15
config['percentRest'] = 1-config['percentValidation']
config['numFolds'] = 9
config['percentBy'] = config['percentRest']/config['numFolds'] 
config['batch_size'] = 100
config['num_epochs'] = 10
config['max_epochs'] = 1
config['epsilon'] = 0.00000000001
config['batchesInferences']=False

config['maxNProp'] = 2
config['maxNeighbors'] = 10000
config['maxNeighbors2'] = 10
config['sampleAttrs'] = 0
config['debug'] = False
config['multipleCollectiveTraining'] = True

#PPR parameters
#config['pageRankOrder'] = False

#number of layers for stacked LSTM (if selected)
config['num_layers'] = 2
config['attr1d'] = 'attr'
config['attr2'] = 'attr2'
config['onlySelectFolds'] = True
config['selectFolds'] = [1, 3, 5, 7, 9, 11, 13, 15]
#config['selectFolds'] = [13, 15]


if config['onlySelectFolds']:
    config['numSelectFolds']=len(config['selectFolds'])
else:
    config['numSelectFolds']=config['numFolds']-1
    config['selectFolds']=xrange(0, config['numSelectFolds'])

if config['numFolds']<=1:
    print("please select appropriate proportions for percentValidation and percentBy")
    sys.exit(0)    


nodeType=os.environ.get('RCAC_SCRATCH')
if nodeType==None:
    config['save_path_prefix']="models/"
    config['dataFolder']="data/"
    config['dataForJoel']="JoelInput/"
else:
    config['save_path_prefix']="models/"
    config['dataFolder']=nodeType+"/GraphDeepLearning/data/"
    config['dataForJoel']=nodeType+"/JoelInput/"