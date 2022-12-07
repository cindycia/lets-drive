#!/bin/sh
SessionName=$1
# Use -d to allow the rest of the function to run
bash clean.sh
# python clear_process.py
sleep 2
SessionName='zero'

tmux new-session -d -s $SessionName 'python3 ../sac_discrete/src/memory/replay_service.py --size 100000 --type real'
tmux new-window -d -n virtual_replay 'python3 ../sac_discrete/src/memory/replay_service.py --size 60000 --type virtual'
tmux new-window -d -n labelling 'python3 ../sac_discrete/src/memory/labeller_service.py'
tmux new-window -d -n logging 'python3 ../sac_discrete/src/agent/sac_discrete/log_service.py'


lr=0.0001
r=12345
r=11442
# r=44787
# r=0

tmux new-window -d -n learn_0
tmux send-keys -t learn_0 'python3 ../sac_discrete/src/agent/sac_discrete/learner.py --gpu 0 --port 2000 --drive_mode imitation --env_mode server --lr '$lr' --seed '$r Enter

tmux new-window -d -n actor_1
tmux new-window -d -n actor_2
tmux new-window -d -n actor_3
tmux send-keys -t actor_1 'bash repeated_sac_actor.sh lets-drive 1 3000 '$lr Enter
tmux send-keys -t actor_2 'bash repeated_sac_actor.sh lets-drive-zero 2 4000 '$lr Enter
tmux send-keys -t actor_3 'bash repeated_sac_actor.sh lets-drive-zero 3 5000 '$lr Enter
# tmux send-keys -t actor_1 'python3 ../sac_discrete/src/agent/sac_discrete/actor.py --aid 1 --port 3000 --drive_mode lets-drive --env_mode server --lr '$lr Enter
# tmux send-keys -t actor_2 'python3 ../sac_discrete/src/agent/sac_discrete/actor.py --aid 2 --port 4000 --drive_mode lets-drive-zero --env_mode server --lr '$lr Enter
# tmux send-keys -t actor_3 'python3 ../sac_discrete/src/agent/sac_discrete/actor.py --aid 3 --port 5000 --drive_mode lets-drive-zero --env_mode server --lr '$lr Enter
# sleep 30
#tmux send-keys -t actor_3 'bash repeated_sac_actor.sh lets-drive-labeller 3 5000 '$lr Enter

# tmux new-window -d -n actor_3 'python3 ../sac_discrete/src/agent/sac_discrete/actor.py --aid 3 --port 5000 --drive_mode lets-drive-labeller --env_mode server --lr '$lr
tmux attach-session -d -t $SessionName

