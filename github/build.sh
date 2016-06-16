#!/bin/bash 

# Environment
module load PrgEnv-gnu/15.11_cuda_7.0_gdr
export GTEST=../../googletest/

# Clean
rm *.o
rm miniUnitTests.out

# Compilation
# nvcc --std=c++11 -rdc=true -arch=sm_37 -isystem ${GTEST}/googletest/include -dc tolerance.cpp

g++ -std=c++11 -c test_tools.cpp
g++ -std=c++11 -isystem ${GTEST}/googletest/include -c tolerance.cpp
g++ -std=c++11 -c portable_math.cpp -o cpu_portable_math.o
nvcc -rdc=true -arch=sm_37 -dc portable_math.cu -o gpu_portable_math.o

nvcc -rdc=true -g -arch=sm_37 --std=c++11 -isystem ${GTEST}/googletest/include minitest.cu ${GTEST}/build/googlemock/gtest/libgtest.a test_tools.o cpu_portable_math.o gpu_portable_math.o tolerance.o -o miniUnitTests.out
