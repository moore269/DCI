import os
nodeType=os.environ.get('RCAC_SCRATCH')
queue=""
machine=""
postFix=""
if "snyder" in nodeType:
    queue='ribeirob'
    machine='Snyder'
if "hammer" in nodeType:
    queue='csit'
    machine='Hammer'


#queue = 'csit'
print(nodeType)
#dataSets=["facebook", "amazon_DVD_20000"]
#dataSets=["amazon_DVD_7500",  "amazon_Music_10000", "amazon_Music_7500" ]
#inputtypes =["no0_LSTM", "no0_LSTM_C", "w_no0_LSTM", "w_no0_LSTM_C"]
dataSets=["facebook", "amazon_DVD_20000", "amazon_DVD_7500",  "amazon_Music_10000", "amazon_Music_7500" ]
inputtypes = [ "w_no0_LSTM", "w_no0_LSTM_C"]
for data in dataSets:
    for inputType in inputtypes:
        submitStr = "qsub -q " + queue + " -v data='" + str(data) + \
            "',inputtype='" + str(inputType) + "' joel"+machine+".sub"
        print(submitStr)
        os.system(submitStr)
