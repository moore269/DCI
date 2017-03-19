import os
nodeType=os.environ.get('RCAC_SCRATCH')
queue=""
if "snyder" in nodeType:
    queue='ribeirob'
if "hammer" in nodeType:
    queue='csit'
queue = 'csit'
#mods = m0 for normal way of handling long sequences
mods=['m0']

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
rnn_types=['rnn']
hidden_sizes=[60, 100, 140, 180, 220]
num_layers=[1]
dropouts=[0]
train_sizes=[0.8]
transitions=[10000]
datasets = ['four_sq', 'yoochoose-clicks']
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

#test 5 parameters, test 1 methodology but use rnn
rnn_types=['rnn']
hidden_sizes=[120]
#hidden_sizes=[60, 80, 100]
num_layers=[1]
dropouts=[0]
train_sizes=[0.8]
transitions=[10000, 20000, 30000, 40000]
mods=['m0']

#/scratch/hammer/m/moore269/lstm2models/yoochoose-clicks/rnn_type_gru_trial_11_hiddenSize_160_numLayers_2_dropout_0.0_train_size_0.8_transitions_10000_gru_best.pkl
#/scratch/hammer/m/moore269/lstm2models/yoochoose-clicks/rnn_type_gru_trial_11_hiddenSize_180_numLayers_2_dropout_0.0_train_size_0.8_transitions_40000_gru_best.pkl
#extra
rnn_types=['gru']
hidden_sizes=[180]
num_layers=[2]
dropouts=[0]
train_sizes=[0.8]
transitions=[40000]
"""

for rnn_type in rnn_types:
	for hidden_size in hidden_sizes:
		for num_layer in num_layers:
			for dropout in dropouts:
				for train_size in train_sizes:
					for transition in transitions:
						for mod in mods:
                                                        for dataset in datasets:
							        submitStr="qsub -q " + queue + " -v rnn_type='" + rnn_type + "',hidden_size='" + str(hidden_size) + "',num_layer='" + str(num_layer) + "',dropout='" + str(dropout) + "',train_size='" + str(train_size) + "',transition='"+str(transition) + "',mod='" +str(mod) + "',data='" +str(dataset) +  "' lstm2_exp.sub"
							        print(submitStr)
							        #os.system(submitStr)
