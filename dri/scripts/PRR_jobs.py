import os
nodeType=os.environ.get('RCAC_SCRATCH')
queue=""
if "snyder" in nodeType:
    queue='ribeirob'
if "hammer" in nodeType:
    queue='csit'
queue = 'csit'
#test 0 parameters - debugging test
dataSets=['facebook']
trials = xrange(0, 10)

for dataset in dataSets:
    for trial in trials:
        submitStr="qsub -q " + queue + " -v dataset='" + dataset + "',trial='" + str(trial) + "' PRR.sub"
        print(submitStr)
        #os.system(submitStr)