#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
source ~/catkin_ws/devel/setup.bash

cd ~/catkin_ws/src/scripts/
echo Container args: "$@"
export CUDA_VISIBLE_DEVICES=$1
exec bash experiment_summit.sh $4 0 0 $3 0 0 $2
