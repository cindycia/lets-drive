#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash" --

# cd ~/catkin_ws

# exec catkin config --merge-devel
# exec catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3



# cd ~/catkin_ws/src/scripts/
# echo Container args: "$@"
export CUDA_VISIBLE_DEVICES=0
server_start(){
    source ~/catkin_ws/devel/setup.bash
    source activate
    conda activate summit36
    pip install psutil
    cd ~/catkin_ws/src/scripts
    # python ./run_data_collection_summit.py
}
server_start
# source ./.bashrc && conda activate summit36
# # exec python ~/catkin_ws/src/scripts/run_data_collection_summit.py
# exec "$@"
exec bash