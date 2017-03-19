#!/bin/bash

USERNAME=moore269
JOBNAME=rnn

#to kill all the jobs
qstat -u$USERNAME | grep "$JOBNAME" | grep csit | cut -d"." -f1 | xargs qalter -l walltime=11:59:00 

