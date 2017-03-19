import os
nodeType=os.environ.get('RCAC_SCRATCH')
queue=""
if "snyder" in nodeType:
    queue='ribeirob'
if "hammer" in nodeType:
    queue='csit'
queue = 'csit'
"""
#test 0 parameters - debugging test
rnn_types=['lstm']
hidden_sizes=[10]
num_layers=[1]
dropouts=[0]
train_sizes=[0.8]
transitions=[20000]
"""

#test 1 parameters - general test
rnn_types=['lstm', 'gru']
hidden_sizes=[120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
num_layers=[1,2]
dropouts=[0]
train_sizes=[0.8]
transitions=[10000, 20000, 30000, 40000]
"""
#test 2 parameters - look at train set size
#determined after test 1 finishes
rnn_types=['lstm', 'gru']
hidden_sizes=[240]
num_layers=[1]
dropouts=[0]
train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
transitions=[40000]

#test 3 parameters - dropout test
rnn_types=['lstm', 'gru']
hidden_sizes=[400, 500]
num_layers=[2]
#dropouts=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dropouts=[0.1, 0.2]
train_sizes=[0.8]
transitions=[10000, 20000, 30000, 40000]

#test 4 parameters - dropout test - look at train set size
rnn_types=['lstm', 'gru']
hidden_sizes=[400, 500]
num_layers=[2]
dropouts=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
train_sizes=[0.8]
transitions=[10000, 20000, 30000, 40000]
"""

for rnn_type in rnn_types:
	for hidden_size in hidden_sizes:
		for num_layer in num_layers:
			for dropout in dropouts:
				for train_size in train_sizes:
					for transition in transitions:
						print(rnn_type+"\t"+str(hidden_size)+"\t"+str(num_layer)+"\t"+str(dropout)+"\t"+str(train_size)+"\t"+str(transition)+"\t")
