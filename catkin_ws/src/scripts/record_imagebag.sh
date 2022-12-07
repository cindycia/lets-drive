port=$1
ROS_MASTER_URI=http://localhost:$port rosbag record -O spectator_images.bag /spectator_images

