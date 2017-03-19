# Prerequisites
Blocks. At the time, I used 0.1.1

Theano

matplotlib

networkx

numpy 

cpickle

scipy

# Data
Download the facebook dataset

https://www.cs.purdue.edu/homes/moore269/data/facebook.zip

This includes all the preprocessed trials, so you should be able to run this from scratch.

For your own data, generate into a similar format for .attr, .edges, .lab

Run generateTrials in code/readData/readData.py



# Config.py
experiments/config.py controls many parameters

numProcesses: leave this as 1 

trials: should control trials when not using the rnnExperimentsPerTrial.py experiment. 

percentValidation: what percentage of nodes go into validation

percentRest: Everything besides nodes in validation set

numFolds: number of folds to divide percentRest nodes by.

percentBy: percentrest/numfolds

batch_size: controls batch size

num_epochs: number of epochs for early stopping. E.g. num_epochs=10 so stop early if no improvement on validation set in last 10 epochs.

max_epochs: maximum number of epochs to perform

epsilon: epsilon value for looking at early stopping performance.

batchesInferences: do we predict in batches or per example


maxNProp: maximum number of collective iterations

maxNeighbors: maximum neighbors to consider

maxNeighbors2: maximum 2nd order neighbors to consider (currently not in paper)

sampleAttrs: ignore this

debug: in debug mode or not

multipleCollectiveTraining: train collectively

#PPR parameters
pageRankOrder: consider the page rank order

#number of layers for stacked LSTM (if selected)

num_layers: currently disabled

attr1d: just some naming conventions for attribute vectors

attr2: same as above

onlySelectFolds: only perform evaluations on specified folds

selectFolds: specify selectfolds

save_path_prefix: where to save models to

dataFolder: specify the data folder

dataForJoel: PLEM data folder



# Example and parameters

Note that to get results from the full version you must run both the swapping and the cross entropy balancing method separately, then analyze their validation set 2's output to select the best method. This is done by modifying plot.py in scripts/ to point to the model directory.

example

rnnExperimentsPerTrial.py LSTM 10 5 neutral 0 facebook none 1 0 1 1 swapdown 0 0 0 0 0 0 3 cpu 0 1

netType: default:LSTM. Options are "LSTM", "LSTM2", "RNNwMini", or "LSTMwMini". 

memory: hidden unit size in RNN

mini_dim: hidden unit size inside 2nd order RNN

prtype: Which PPR do we want to use for comparison. Neutral means do regular PPR without any bias. Negative means only consider negatively labeled neighbors. Positive means only consider positive neighbors. default:neutral.

singlepr: default:0

fName: dataset name

bias_self: bias towards yourself or not.  default:none

degree: do we include degree as a feature. default:1

testLimit: test limit by peeking true labels. default:0

usePrevWeights: Use previous weights from last collective iteration or train from scratch. default:1

no0Deg: Don't include 0 degree nodes. default:1

dataAug: do we perform data augmentation? default:swapdown. Options are none, swapdown for swapping then downsampling, and swap for swapping without downsampling.

randInit: Initialize to random predictions on first iteration or use DRI. default:0

pageRankOrder: Do we specify a page rank order. default:0

perturb: Do we perturb labels for experiments. default:0

usePro: Do we perform Cross entropy balancing. default:0

lastH: propagating last hidden states collectively in addition to predictions. default:0

lastHStartRan: Let the first hidden states propagated start from random. default:0

changeTrainValid: What should we propagate variants. In particular, should we propagate predictions or probabilities of testing sets. This is described further below. default:3

gpu: Use gpu or cpu

i: trial number

j: fold number

Currently, I have it set up to process 17 folds.  The split is 15% total validation, which leaves 85% for training and testing sets.
85/17 = 5, so each fold contains 5% of data. For example, Fold 0 means train on 5%, rest testing
    
    
changeTrainValid means multiple things

-1 means to propagate predictions of train,valid,test

0 means to propagate labels of train,valid on first iteration, propagate predictions thereafter

1 means to propagate labels of train and propagate prediction probabilities on validation,test

2 means to propagate labels of train and valid and prediction probabilities of test

3 means to propagate labels of train and valid and predictions of test, actual propagation is the actual prediction (ie >=0.5 then 1, else 0)

4 means to propagate labels of train and valid and predictions of test, except cut validation into 3% and 12%, use 3% as pseudotesting

5 means to propagate labels of train and valid and predictions of test, actual propagation is the actual prediction (ie >=0.5 then 1, else 0)
  except cut validation into 3% and 12%, use 3% as validation2

# Parallel experiments

For much of this work, I have relied on purdue clusters to parallel experiments. That is, for a set of trials/folds, I run each individually as separate processes, then aggregate results in the end using the plot.py script.

If you have a similar setup, you can run code similar to 
snyderScripts/rnn_experiments_jobs_mini.py

which ends up calling snyderScripts/rnn_exp_mini.sub

# Simplification

I am well-aware this code has many parameters and unnecessary things (for other experiments), which will make it hard to read.  As is, you should be able to run it given the sample dataset and a given modified one (See data section).  However, I wish to simplify it in the future for easier readability and use but have been preocupied with work lately.  I hope to get better readable versions out soon.
