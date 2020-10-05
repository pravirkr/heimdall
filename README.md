# Heimdall
Transient Detection Pipeline 

## Dependencies
* [CUDA 10.0+](https://developer.nvidia.com/cuda-toolkit-archive)
* [Boost 1.60+](https://www.boost.org/)
* [CMake 3.15+](https://cmake.org/download/)

## Installation
1.  Update CMakeLists.txt with your CUDA, Boost path and GPU architecture.
2.  mkdir build && cd build && cmake ..
3.  make && make install
    
This will also build a shared object library named libdedisp.so
