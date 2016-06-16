#!/bin/bash 

# Environment
echo "Loading environment"
module load PrgEnv-gnu/15.11_cuda_7.0_gdr
export GTEST=../../googletest/

# Clean
echo "Cleaning"
rm -f *.o
rm -f *.out

# Compilation
echo "Compilation"
# nvcc --std=c++11 -rdc=true -arch=sm_37 -isystem ${GTEST}/googletest/include -dc tolerance.cpp

# Test helper libraries
echo "Helper libraries"
g++ -std=c++11 -c test_tools.cpp
g++ -std=c++11 -isystem ${GTEST}/googletest/include -c tolerance.cpp

echo "Portable transcendental functions"
g++ -std=c++11 -ffp-contract=off -c portable_math.cpp -o cpu_portable_math.o
nvcc -rdc=true --fmad=false -arch=sm_37 -dc portable_math.cu -o gpu_portable_math.o

# Unit tests
echo "Unit tests"
nvcc -rdc=true -g -arch=sm_37 --std=c++11 -isystem ${GTEST}/googletest/include approxtests.cu ${GTEST}/build/googlemock/gtest/libgtest.a test_tools.o cpu_portable_math.o gpu_portable_math.o tolerance.o -o approxTests.out
nvcc -rdc=true -g -arch=sm_37 --std=c++11 -isystem ${GTEST}/googletest/include bitreprotests.cu ${GTEST}/build/googlemock/gtest/libgtest.a test_tools.o cpu_portable_math.o gpu_portable_math.o tolerance.o -o reproTests.out
