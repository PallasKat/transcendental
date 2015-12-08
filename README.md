# README #

Transcendental functions for _bitreproducibility_ in the context of `Stella`, `Cosmo` and the CrClim project.

### What is this repository for? ###

* This repository contains the experimental code for the transcendental functions that are added to Stella for bitreproduciblity. The proposed implementation concerns the logarithm, the exponential and the exponentiation mathematical functions. For now the implementation are only for `double` values.

* The code is actually experimental and it intended to run on CPU and GPU.

### How do I get set up? ###

* The test depends on Google Test (https://github.com/google/googletest) and should be installed on the machine running the tests.

* A Cuda and a C++ compiler (ie. nvcc 7.0.27 or g++ 4.9.3) compatible with C++11 is needed

* The test tools and the math functions must be compiled first:
    * `g++ -std=c++11 -c test_tools.cpp`
    * `g++ -c cpu_mch_math.cpp`

* Then the tests are compiled like: 
    * `nvcc -g --std=c++11 -isystem ${GTEST}/googletest/include cuda_unittest.cu ${GTEST}/build/googlemock/gtest/libgtest.a test_tools.o cpu_mch_math.o -o mathUnitTests.out`

Finally to run the tests on a CSCS machine, use `srun -n 1 -p debug --gres=gpu:1 -t 00:10:00 ./mathUnitTests.out` and this produces an output of the form:
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
### Contribution guidelines ###

* The code and the test are written by Christophe Charpilloz (MeteoSwiss)
* The mathematical implementation are inspired by the one proposed by Nvidia Corporation (http://www.nvidia.com)

### Who do I talk to? ###

* Repo owner or admin