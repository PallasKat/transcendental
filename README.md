# README

This repository contains a set of portable transcendental functions to be used for bit-reproducibility. Currently this is aimed to be integrated in the Stella library used by the Dycore and Cosmo model in the context of the CrClim project. This project is a joint collaboration between MeteoSchweiz and the ETHZ.

## Introduction

The goal is to provide a set of transcendental function that are used regardless the compiler or the mathematical library. Indeed as the approximation of these functions can differ between libraries, this produces results that are not bit-reproductible. Hence the provided code should be compiled for any accelerator. For now we propose to only support GPU accelerators (CUDA).

The proposed implementation is written in C++ and contains the logarithm, exponential and exponentiation mathematical functions. For now the implementation are only for `double` values. But the support for `single` should be added and the set of function extended with trigonometric, inverse trigonometric and hyperbolic functions. These functions are required by the Cosmo model. The current code is actually experimental and is not yet intended for production.

## Dependencies

To compile the functions, a Cuda and a C++ compiler compatible with C++11 are needed (ie. nvcc 7.0.27 or g++ 4.9.3).

Unit tests are also provided to assess the reproducibility of the computation as long with the accuracy regarding the standard mathematical C++ library. The tests depend on Google Test (https://github.com/google/googletest) and should be installed on the machine running the tests.

## Compilation and execution 

First the test helpers should be compiled:
`g++ -std=c++11 -c test_tools.cpp`
and
`g++ -std=c++11 -isystem ${GTEST}/googletest/include -c tolerance.cpp`

Then the transcendental functions:
`g++ -std=c++11 -ffp-contract=off -c portable_math.cpp -o cpu_portable_math.o`
and
`nvcc -rdc=true --fmad=false -arch=sm_37 -dc portable_math.cu -o gpu_portable_math.o`

Finally the tests for the approximation:
`nvcc -rdc=true -g -arch=sm_37 --std=c++11 -isystem ${GTEST}/googletest/include approxtests.cu ${GTEST}/build/googlemock/gtest/libgtest.a test_tools.o cpu_portable_math.o gpu_portable_math.o tolerance.o -o approxTests.out`
and the tests for reproduciblity:
`nvcc -rdc=true -g -arch=sm_37 --std=c++11 -isystem ${GTEST}/googletest/include bitreprotests.cu ${GTEST}/build/googlemock/gtest/libgtest.a test_tools.o cpu_portable_math.o gpu_portable_math.o tolerance.o -o reproTests.out`

Finally to run the tests on a CSCS machine, use `srun -n 1 -p debug --gres=gpu:1 -t 00:10:00 ./approxTests.out` and this produces an output of the form:
```
[==========] Running 2 tests from 2 test cases.
[----------] Global test environment set-up.
[----------] 1 test from LogCPUTest
[ RUN      ] LogCPUTest.PositiveValues
[       OK ] LogCPUTest.PositiveValues (0 ms)
[----------] 1 test from LogCPUTest (0 ms total)

[----------] 1 test from LogGPUTest
[ RUN      ] LogGPUTest.PositiveValues
[       OK ] LogGPUTest.PositiveValues (455 ms)
[----------] 1 test from LogGPUTest (455 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 2 test cases ran. (455 ms total)
[  PASSED  ] 2 tests.
```
and also `srun -n 1 -p debug --gres=gpu:1 -t 00:10:00 ./reproTests.out`:
```
[==========] Running 2 tests from 2 test cases.
[----------] Global test environment set-up.
...
[  PASSED  ] 2 tests.
```

## Contribution guidelines

The code and the test are written by PallasKat (MeteoSchweiz) and the mathematical implementation are inspired by the one proposed by Nvidia Corporation (http://www.nvidia.com).
