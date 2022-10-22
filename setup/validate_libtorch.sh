g++ torch-example.cc \
-std=c++17 \
-D_GLIBCXX_USE_CXX11_ABI=1 \
-I${LIBTORCH_SYS}/include \
-I${LIBTORCH_SYS}/include/torch/csrc/api/include \
-L${LIBTORCH_SYS}/lib \
-Wl,-R${LIBTORCH_SYS}/lib \
-ltorch -ltorch_cpu -lc10 -lgomp -lpthread \
-o torch-example
