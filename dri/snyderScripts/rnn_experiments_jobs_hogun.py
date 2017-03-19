import os

#dataSets = ["imdb", "reality_mining", "dblp"]
#dataSets = ["dblp"]
#dataSets = ["reality_mining"]
dataSets = ["facebook"]
netTypes = ["LSTMwMini"]
degrees = [0]
queue = "csit"
mems = [5]
minidims = [20]
trials = range(0, 10)
folds = [1,3,5,7,9,11,13, 15]
#folds = [13]
for data in dataSets:
    for netType in netTypes:
        for degree in degrees:
            for mem in mems:
                for minidim in minidims:
                    for i in trials:
                        for j in folds:
                            submitStr = "qsub -q "+queue+" -v nettype='" + str(netType) +"',mem='"+str(mem)+"',minidim='"+str(minidim)+ "',deg='"+str(degree)+"',fname='preprocessed_"+str(data)+"',trial='"+str(i)+"',fold='"+str(j)+"' rnn_exp_hogun.sub"
                            print(submitStr)
                            os.system(submitStr)
