import os
nodeType=os.environ.get('RCAC_SCRATCH')
queue=""
if "snyder" in nodeType:
    queue='ribeirob'
if "hammer" in nodeType:
    queue='csit'
queue = 'csit'
#test 0 parameters - debugging test


#old version of jobs
"""
#Snyder 1
queue='csit'
dataSets=['amazon_Music_10000']
prTypes=['pos', 'neg']

#Snyder 2
queue='ribeirob'
dataSets=['amazon_Music_10000', 'amazon_DVD_20000']
prTypes=['neutral']

#Hammer 1
queue='csit'
dataSets=['amazon_DVD_20000']
prTypes=['pos', 'neg']
"""

#new version of jobs
#in common
dataSets=['IMDB_5']
folds=[1,3,5,7, 9, 11, 13, 15]
#folds=[2, 6, 12]
#folds=range(0, 2)+range(3,6)+range(7,12)+range(13,17)
trials=xrange(0, 10)


#Snyder 1
queue='csit'
prTypes=['neutral']
"""
#Snyder 2
queue='ribeirob'
prTypes=['neg']

#Hammer 1
queue='csit'
prTypes=['neutral']
"""

for dataset in dataSets:
    for fold in folds:
        for trial in trials:
            for prtype in prTypes:
                submitStr="qsub -q " + queue + " -v dataset='" + dataset + "',trial='" + str(trial) + "',fold='" + str(fold)  +"',prtype='" + prtype+"' PRR.sub"
                print(submitStr)
                os.system(submitStr)
