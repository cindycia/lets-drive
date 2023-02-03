# LeTS-Drive docker Setup in Ubuntu 20.04
## 1. Build the images locally
If you want to build the images yourself, please follow the [readme](https://github.com/cindycia/lets-drive/blob/ubuntu-20-04/docker/README.md#setup-a-docker-environment-for-lets_drive) under docker directory. Then change [line 30 in launch_docker.py](https://github.com/cindycia/lets-drive/blob/a8fee1bca519b7440d30f35cdb16af1cde9bc37d/catkin_ws/src/scripts/launch_docker.py#L30) to
```python
default="noetic_cuda11_1_cudnn8_libtorch_opencv4_ws",
```
## 2. Pull the image from docker hub
If you want to pull the images from docker hub instead of building it yourself, you need to pull down two images:
wenjingtang/noetic_cuda11_1_cudnn8_libtorch_opencv4_ws_noentry and wenjingtang/noetic_cuda11_1_cudnn8_libtorch_opencv4_ws.
The first one is for compile the workspace.


## 3. Create ROS Package from Source Code
```bash
mv catkin_ws/src .
cd && mkdir -p catkin_ws/src
cd catkin_ws
catkin config --merge-devel
catkin build
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
mv src/* catkin_ws/src/
```
Then you need to compile the workspace in the container without entry file.
```bash
cd ~/catkin_ws
catkin config --merge-devel
catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3
```
