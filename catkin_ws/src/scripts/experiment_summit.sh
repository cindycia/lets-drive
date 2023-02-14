#!/bin/bash
#!/usr/bin/env python3

SECONDS=0

gpu=0
s=0
e=0
port=2000
launch_sim=0
record_bags=0
mode=lets-drive
maploc=random

if [ "$#" -gt 0 ]; then
    mode=$1
fi

if [ "$#" -gt 1 ]; then
    gpu=$2    
fi

if [ "$#" -gt 2 ]; then
    launch_sim=$3
fi

if [ "$#" -gt 3 ]; then
    record_bags=$4
fi

if [ "$#" -gt 4 ]; then
    s=$5
fi

if [ "$#" -gt 5 ]; then
    e=$6
fi

if [ "$#" -gt 6 ]; then
    port=$7
fi



# maploc=test
# maploc=beijing
# maploc=magic
maploc=meskel_square
# maploc=beijing
# maploc=shi_men_er_lu
# maploc=highway

# mode=rollout
# mode=joint_pomdp
# mode=imitation
# mode=gamma
# mode=lets-drive
# mode=lets-drive-zero

# rands=4592987
# rands=604234
# rands=4156662
# rands=9475
# rands=1310663
rands=-1
# rands=7780225

eps_len=120.0

# debug=1
debug=0

echo "mode $mode"
# echo "User: $USER"
# echo "PATH: $PATH"
# echo "PYTHON: $(which python3)"

echo $(nvidia-smi)
num_rounds=1
rm exp_log_*
echo "log: exp_log_"$s'_'$e
# export CUDA_VISIBLE_DEVICES=$gpu
# echo "CUDA_VISIBLE_DEVICES=" $CUDA_VISIBLE_DEVICES

echo "source /opt/ros/noetic/setup.bash"

for i in $(seq $s $e)
do
    echo "[experiment_summit] clearing process"
    python ./clear_process.py $port
    sleep 1
    echo "[experiment_summit] starting run_data_collection.py script"
    start_batch=$((i*num_rounds))
    echo "[experiment_summit] start_batch: $start_batch"
    end_batch=$(((i+1)*num_rounds))
    echo "[experiment_summit] end_batch: $end_batch"
    echo "[experiment_summit] gpu_id: $gpu"
    python3 run_data_collection.py --record $record_bags \
    --sround $start_batch --eround $end_batch \
    --make 0 --verb 1 --gpu_id $gpu --debug $debug \
    --port $port --maploc $maploc --rands $rands --launch_sim $launch_sim --eps_len $eps_len --drive_mode $mode \
    --num-car 75 --num-bike 25 --num-pedestrian 10 2>&1 | tee -a exp_log_$BASHPID
    echo "[experiment_summit] clearing process"
    python ./clear_process.py $port
    sleep 1
    # echo "Do you want to exit? q:yes (2s)"# to get a newline after quitting
    # read -n 1 -t 3 input
    # if [[ $input = "q" ]] || [[ $input = "Q" ]]
    # then
    #     echo # to get a newline after quitting
    #     break
    # fi
done
echo "Exp finished in "$SECONDS" seconds"
