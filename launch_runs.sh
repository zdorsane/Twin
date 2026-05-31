#!/bin/bash
# One-shot launcher: detaches the sequential run from this shell.
cd /home/crbt/Twin
source venv_tf/bin/activate
nohup bash run_sequential.sh > run_sequential.log 2>&1 &
BGPID=$!
echo "$BGPID" > /tmp/twin_run_pid
echo "Launched PID $BGPID"
disown $BGPID
