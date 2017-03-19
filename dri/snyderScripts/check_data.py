import os
folder=os.environ.get('RCAC_SCRATCH')+"/lstm2models/yoochoose-clicks/"
print(folder)
#test 0 parameters - debugging test
rnn_types=['lstm']
hidden_sizes=[10]
num_layers=[1]
dropouts=[0]
train_sizes=[0.8]
transitions=[20000]


#test 1 parameters - general test
rnn_types=['rnn']
hidden_sizes=[60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
num_layers=[1]
dropouts=[0.0]
train_sizes=[0.8]
transitions=[10000, 40000]
"""
#test 2 parameters - look at train set size
#determined after test 1 finishes
rnn_types=['lstm', 'gru']
hidden_sizes=[300]
num_layers=[1]
dropouts=[0.0]
train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8. 0.9]
transitions=[10000, 20000, 30000, 40000]

#test 3 parameters - dropout test
rnn_types=['lstm', 'gru']
hidden_sizes=[400, 500]
num_layers=[2]
dropouts=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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
						subStr = "rnn_type_"+rnn_type+"_trial_11_hiddenSize_"+str(hidden_size)+"_numLayers_"+str(num_layer)+"_dropout_"+str(dropout)+"_train_size_"+str(train_size)+"_transitions_"+str(transition)+"_gru_best.pkl"
						print(subStr)
						check=os.path.isfile(folder+subStr)
						if not check:
							print(folder+subStr)
