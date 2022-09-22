sudo apt-get -y update && apt-get -y upgrade
sudo apt-get -y install g++ git libgflags-dev libgoogle-glog-dev \
    libomp5 libomp-dev libiomp-dev libopenmpi-dev protobuf-compiler \
    python3 python3-pip python3-setuptools python3-yaml wget

# Install CMake 3.14

CMAKE=cmake-3.14.1.tar.gz
CMAKE_FOLDER=cmake-3.14.1
cd && wget https://github.com/Kitware/CMake/releases/download/v3.14.1/${CMAKE}
tar xvzf ${CMAKE} && rm ${CMAKE} && cd ${CMAKE_FOLDER} && ./bootstrap --parallel=$(nproc)
make -j$(nproc) && sudo make install

# Intel MKL installation

cd && wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && rm GPG-PUB*
sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
sudo apt-get update && sudo apt-get -y install intel-mkl-64bit-2019.1-053
sudo rm /opt/intel/mkl/lib/intel64/*.so

# Download and build libtorch with MKL support

export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5"
echo TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST
export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
pytorch_dir=pytorch
libtorch_dir=libtorch
cd && git clone --recurse-submodules -j8 https://github.com/pytorch/pytorch.git $pytorch_dir

# Compile libtorch and install
# cd && cd $pytorch_dir && git checkout 8fb756d3b25be6b8f56bbd0d7b52b1f1ed1e7119
cd && cd $pytorch_dir && mkdir -p build && cd build && BUILD_TEST=OFF USE_NCCL=OFF python3 ../tools/build_libtorch.py
cd && mkdir -p ~/$libtorch_dir/include && mkdir -p ~/$libtorch_dir/share
cp -r $pytorch_dir/build/build/lib ~/$libtorch_dir
cp -r $pytorch_dir/torch/share/cmake ~/$libtorch_dir/share/cmake
for dir in ATen c10 caffe2 torch; do cp -r $pytorch_dir/torch/include/$dir ~/$libtorch_dir/include; done

# Compile pytorch and install
pip3 install pyyaml mkl mkl-include setuptools cffi typing
cd && cd pytorch && sudo python3 setup.py install --cmake
