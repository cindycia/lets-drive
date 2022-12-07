#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash" --

#cd /summit/catkin_ws

#exec catkin config --merge-devel
#exec catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3

source /home/summit/catkin_ws/devel/setup.bash

cd /home/summit/catkin_ws/src/scripts/
echo Container args: "$@"
export CUDA_VISIBLE_DEVICES=0
exec python3 run_data_collection_planner.py --round $1 --run $2
# exec "$@"