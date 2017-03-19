import os
import sys
nodeType=os.environ.get('RCAC_SCRATCH')
queue=""
if "snyder" in nodeType:
    queue='ribeirob'
if "hammer" in nodeType:
    queue='csit'
#queue='csit'

memory=[10]
examples = [10]
#1 for LSTM, 0 for GRU
netTypes=["LSTM", "LSTM2", "StackLSTM", "BIDLSTM"]

#first test to start
begIndex=1

#last test + 1
endIndex=4

for i in range(begIndex, endIndex):
    for mem in memory:
        for exam in examples:
            for net in netTypes:
                submitStr="qsub -q "+queue+" -v mem='" +str(mem)+"',test='"+str(i)+"',examples='" +str(exam)+"',nettype='" +str(net)+"' rnn_exp.sub"
                print(submitStr)
                os.system(submitStr)
