#!/bin/bash
_term() {
  echo "Caught SIGTERM signal!"
  kill -TERM "$child" 2>/dev/null
}

trap _term SIGTERM
trap _term SIGINT

gpu=$1
actor_lr=$2
critic_lr=$3
alpha_lr=$4
rand=$5

export CUDA_VISIBLE_DEVICES=$gpu

for i in {1..100}
do
    timeout 3600 python3 ../reinforcement/scripts/sac_train.py 0 $actor_lr $critic_lr $alpha_lr $rand &
    child=$!
    wait "$child"
    echo "Do you want to exit? q:yes (2s)" # to get a newline after quitting
    read -n 1 -t 2 input
    if [[ $input = "q" ]] || [[ $input = "Q" ]]
    then
        echo # to get a newline after quitting
        break
    fi
done
