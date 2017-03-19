import os

dataSets = [("imdb",4)]
#dataSets = [("dblp",6)]
#dataSets = [("reality_mining", 2)]
#tuples of dataset, then numclasses
#dataSets = [("facebook", 2)]

#0 for regular rnn, 1 for cnn, 2 for collective rnn
NNTypes = [2]
rInits = [1]
neighbAvgs = [1]
pNeighbs = [0,1]
iterEpochs = [1]
poolings = [0,1]
queue = "csit"
trials = range(0, 5)
for data in dataSets:
    for nn in NNTypes:
        for rinit in rInits:
            for neighbavg in neighbAvgs:
                for pneighb in pNeighbs:
                    for iterepoch in iterEpochs:
                        for pool in poolings:    
                            for i in trials:
                                submitStr = "qsub -q "+queue+" -v data='" + str(data[0]) + "',nclasses='"+str(data[1])+ "',nn='"+str(nn)+"',pool='"+str(pool)+"',rinit='"+str(rinit)+ "',neighbavg='"+str(neighbavg)+"',pneighb='"+str(pneighb)+"',iterepoch='"+str(iterepoch) + "',trial='"+str(i)+"' rnn_hogun_time.sub"
                                print(submitStr)
                                #os.system(submitStr)
