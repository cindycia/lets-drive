#!/bin/sh
SessionName=$1
# Use -d to allow the rest of the function to run
bash clean.sh
# python clear_process.py
sleep 2
SessionName='zero'

tmux new-session -d -s $SessionName 'python3 ../sac_discrete/src/memory/replay_service.py --size 100000'
tmux new-window -d -n labelling 'python3 ../sac_discrete/src/memory/labeller_service.py'
tmux new-window -d -n logging 'python3 ../sac_discrete/src/agent/sac_discrete/log_service.py'


# lr=0.00015
lr=0.0001
r=0
r=11442
r=44787

tmux new-window -d -n learn_0
tmux send-keys -t learn_0 'python3 ../sac_discrete/src/agent/imitation/learner.py --gpu 0 --port 2000 --drive_mode imitation --env_mode server --lr '$lr' --seed '$r Enter

tmux new-window -d -n actor_1 'python3 ../sac_discrete/src/agent/imitation/actor.py --aid 1 --port 3000 --drive_mode lets-drive --env_mode server --lr '$lr
sleep 30
tmux new-window -d -n actor_2 'python3 ../sac_discrete/src/agent/imitation/actor.py --aid 2 --port 4000 --drive_mode lets-drive-labeller --env_mode server --lr '$lr 
tmux new-window -d -n actor_3 'python3 ../sac_discrete/src/agent/imitation/actor.py --aid 3 --port 5000 --drive_mode lets-drive-labeller --env_mode server --lr '$lr

tmux attach-session -d -t $SessionName

