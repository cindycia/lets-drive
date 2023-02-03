# Setup a docker environment for lets_drive

## 1. noetic_cuda11_1_cudnn8

```bash
docker build -t noetic_cuda11_1_cudnn8 -f Dockerfile.ros .

#run the container
docker run -it noetic_cuda11_1_cudnn8
```



## 2. noetic_cuda11_1_cudnn8_libtorch_opencv

```bash
docker build -t noetic_cuda11_1_cudnn8_libtorch_opencv -f Dockerfile.ros_torch_opencv .
#run the container
docker run -it noetic_cuda11_1_cudnn8_libtorch_opencv .
```


## 3. The container withour entry file.

Need to mount the catkin_ws to the container

```bash
docker build -t noetic_cuda11_1_cudnn8_libtorch_opencv4_ws_noentry -f Dockerfile.init_empty .
docker run -it -v /path/to/your/lets_drive/docker/catkin_ws:/root/catkin_ws noetic_cuda11_1_cudnn8_libtorch_opencv4_ws_noentry /bin/bash
```

## 4. The final container

Need to mount the catkin_ws to the container

```bash
docker build -t noetic_cuda11_1_cudnn8_libtorch_opencv4_ws -f Dockerfile.init .
docker run -it -v /path/to/your/lets_drive/docker/catkin_ws:/root/catkin_ws noetic_cuda11_1_cudnn8_libtorch_opencv4_ws /bin/bash
```

