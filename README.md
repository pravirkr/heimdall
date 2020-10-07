# Heimdall
Transient Detection Pipeline

[![Build](https://github.com/pravirkr/heimdall/workflows/Build/badge.svg)](https://github.com/pravirkr/heimdall/actions)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/71ddadb7cb954d928dec08b862d4bfac)](https://www.codacy.com/gh/pravirkr/heimdall/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=pravirkr/heimdall&amp;utm_campaign=Badge_Grade)

## Dependencies
* [CUDA 10.0+](https://developer.nvidia.com/cuda-toolkit-archive)
* [Boost 1.60+](https://www.boost.org/)
* [CMake 3.15+](https://cmake.org/download/)

## Installation
1.  Update CMakeLists.txt with your CUDA, Boost path and GPU architecture.
2.  mkdir build && cd build && cmake ..
3.  make && make install
    
This will also build a shared object library named libdedisp.so
