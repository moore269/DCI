#!/bin/bash

USERNAME=moore269
JOBNAME=standby
#sleep 5h

#qstat -u $USERNAME | cut -d"." -f1 | xargs qdel

#to kill all the jobs
qstat -u$USERNAME | grep "$JOBNAME" | cut -d"." -f1 | xargs qdel

#to kill all the running jobs
#qstat -u$USERNAME | grep "R" | cut -d"." -f1 | xargs qdel

#to kill all the queued jobs
#qstat -u$USERNAME | grep "Q" | cut -d"." -f1 | xargs qdel
