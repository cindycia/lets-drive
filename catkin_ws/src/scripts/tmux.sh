#!/bin/sh
SessionName=$1
# Use -d to allow the rest of the function to run
# bash clean.sh
python clear_process.py
sleep 2
SessionName='zero'

# tmux new-session -d -s $SessionName 'python3 ../sac_discrete/src/memory/replay_service.py --size 100000'
# tmux new-window -d -n labelling 'python3 ../sac_discrete/src/memory/labeller_service.py'
# tmux new-window -d -n logging 'python3 ../sac_discrete/src/agent/sac_discrete/log_service.py'

tmux new-session -d -s $SessionName 'python3 ../sac_discrete/src/env/env_service.py'
tmux new-window -d -n actor 
tmux new-window -d -n recorder 
# tmux send-keys -t actor 'bash experiment_summit.sh imitation' Enter
tmux send-keys -t actor 'bash experiment_summit.sh lets-drive' Enter
tmux send-keys -t recorder 'bash record_imagebag.sh 2111' Enter

tmux attach-session -d -t $SessionName

