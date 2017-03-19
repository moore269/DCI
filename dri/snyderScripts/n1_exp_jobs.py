import os
nodeType=os.environ.get('RCAC_SCRATCH')
queue=""
if "snyder" in nodeType:
    queue='ribeirob'
if "hammer" in nodeType:
    queue='csit'

for i in range(5, 6):
    submitStr="qsub -q "+ queue +" -v test='"+str(i)+"' n1_exp.sub"
    print(submitStr)
    os.system(submitStr)

