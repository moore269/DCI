#!/bin/bash

USERNAME=moore269
JOBNAME=rnn_prr

#to kill all the jobs
qstat -u$USERNAME | grep ribeirob | cut -d"." -f1 | xargs qrls

