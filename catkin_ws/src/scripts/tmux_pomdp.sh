#!/bin/sh
SessionName=$1
# Use -d to allow the rest of the function to run
python clear_process.py
SessionName='zero'

tmux new-session -d -s $SessionName 'python3 ../sac_discrete/src/memory/replay_service.py --size 100000'
tmux new-window -d -n logging 'python3 ../sac_discrete/src/agent/sac_discrete/log_service.py'

lr=0.00015

tmux new-window -d -n actor_0 'python3 ../sac_discrete/src/agent/sac_discrete/actor.py --aid 0 --port 2000 --drive_mode joint_pomdp --env_mode server --term 20000 --lr '$lr
tmux new-window -d -n actor_1 'python3 ../sac_discrete/src/agent/sac_discrete/actor.py --aid 1 --port 3000 --drive_mode joint_pomdp --env_mode server --term 20000 --lr '$lr
tmux new-window -d -n actor_2 'python3 ../sac_discrete/src/agent/sac_discrete/actor.py --aid 2 --port 4000 --drive_mode joint_pomdp --env_mode server --term 20000 --lr '$lr 
tmux new-window -d -n actor_3 'python3 ../sac_discrete/src/agent/sac_discrete/actor.py --aid 3 --port 5000 --drive_mode joint_pomdp --env_mode server --term 20000 --lr '$lr

tmux attach-session -d -t $SessionName

