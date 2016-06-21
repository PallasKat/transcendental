# README
This repository contains a set of portable transcendental functions to be used for bit-reproducibility. Currently this is aimed to be integrated in the Stella library used by the Dycore and Cosmo model in the context of the [crClim project](http://www.c2sm.ethz.ch/research/crCLIM.html). This project is a joint collaboration between MeteoSchweiz and the ETHZ.

## Introduction
The goal of this repository is to provide a set of transcendental function that are used regardless the compiler or the mathematical library. Indeed as the approximation of these functions can differ between libraries, this produces results that are not bit-reproducible. Hence the provided code should be compiled for any accelerator. For now we propose to only support GPU accelerators ([CUDA](https://developer.nvidia.com/about-cuda)).

The proposed implementation is written in C++ and contains the logarithm, exponential and exponentiation mathematical functions. For now the implementation are only for double precision (`double`) values. But the support for single precision (`float`) values should be added. Also the set of function should be extended with trigonometric, inverse trigonometric and hyperbolic functions. These functions are required by the [COMSO model](http://www.cosmo-model.org/). The current code is actually experimental and is not yet intended for production.

## Dependencies
To compile the functions, a Cuda and a C++ compiler compatible with C++11 are needed (ie. nvcc 7.0.27 or g++ 4.9.3).

Unit tests are provided to assess the reproducibility of the computation as long with the accuracy regarding the standard mathematical C++ library. The tests depend on [Google Test](https://github.com/google/googletest) framework and should be installed on the machine running the tests. There are two kind of tests: the ones that assess the accuracy of the implementation (i.e. versus a reference implementation like `cmath` or `cudamath`) and the ones that assess the reproducibility between our accelerated and non accelerated implementations.

Benchmarks are also provided to measure the penalty of proposing portable transcendental functions. The benchmarks depend on the [Hayai](https://github.com/nickbruun/hayai) library and should be installed on the machine running the benchmarks. The benchmarks measure the time of the accelerated and non accelerated version of the portable functions as long with the equivalent operations implemented in `cmath` and `cudamath`.

## Compilation
This section describe how to compile and execute the tests and benchmarks. Two way are proposed: Make and a build script.

### Make
There are seven targets but only four are interesting for compilation and one for cleaning. The target `approx` builds the functions and unit tests to evaluate the accuracy of the portable functions: 
```
> make approx
g++ -std=c++11 -c test_tools.cpp
g++ -std=c++11 -isystem ../googletest//googletest/include  -c tolerance.cpp
...
>
```
and produce the executable `approxTests.out`. 

The target `repro` builds the functions and unit tests to evaluate the reproducibility of the portable functions by comparing the results of the accelerated on non accelerated version: 
```
> make repro
g++ -std=c++11 -c test_tools.cpp
g++ -std=c++11 -isystem ../googletest//googletest/include  -c tolerance.cpp
g++ -std=c++11 -ffp-contract=off -c portable_math.cpp -o cpu_portable_math.o
...
>
```
and produce the executable `reproTests.out`.

The target `bench` builds the benchmarks.  
```
> make bench
g++ -std=c++11 -ffp-contract=off -c portable_math.cpp -o cpu_portable_math.o
In file included from portable_math.cpp:1:0:
portable_math.h:16:19: note: #pragma message: probably using gcc
...
>
```
and produce the executable `perfTests.out`.

Finally the target `all` build both tests and the benchmarks executables and `clean` remove all produced files by the compilation steps.

We also add that some variable are specific to our environment:
```
GTEST=../googletest/
HAYAI=../hayai/
```
and those two variables are path to the dependencies and may be reassigned.

### The build script (or manual steps)
The build script (`build.sh`) contains all the commands that should be executed in the terminal to build the tests and benchmarks. The command are self-explanatory. However we only add that the beginning of the script is CSCS specific and should be adapted to the current environment if used:
```
module load PrgEnv-gnu/15.11_cuda_7.0_gdr
export GTEST=../../googletest/
export HAYAI=../../hayai/
```
Then the script is simply used as:
```
> ./build.sh 
Loading environment
Cleaning
Compilation
Helper libraries
Portable transcendental functions
In file included from portable_math.cpp:1:0:
portable_math.h:16:19: note: #pragma message: probably using gcc
   #pragma message "probably using gcc"
                   ^
portable_math.h:12:17: note: #pragma message: using nvcc
   #pragma message "using nvcc"
                 ^
Unit tests (approximation)
Unit tests (reproductibility)
Performance tests
> 
```
and produces the three executables: `approxTests.out`, `reproTests.out` and `perfTests.out`.

## Execution and outputs examples
It's straightforward to use the produced executables. However we describe here how to execute the tests on the CSCS machine Keash or Escha.

First the correct environment should be loaded:
```
> module load PrgEnv-gnu/15.11_cuda_7.0_gdr
```
Then `srun` is used to execute the executables. For example with `approxTests.out` one can type (when logged on a CSCS machine):
```
srun -n 1 -p debug --gres=gpu:1 -t 00:10:00 ./approxTests.out
``` 
and this produces an output of the form:
```
[==========] Running 42 tests from 7 test cases.
[----------] Global test environment set-up.
[----------] 2 tests from LogCPUTest
[ RUN      ] LogCPUTest.PositiveValues
[       OK ] LogCPUTest.PositiveValues (3 ms)
[ RUN      ] LogCPUTest.NegativeValues
[       OK ] LogCPUTest.NegativeValues (19 ms)
[----------] 2 tests from LogCPUTest (22 ms total)
...
[  FAILED  ] PowGPUTest.zeroToOneBaseNegInfExponentValues

14 FAILED TESTS
srun: error: eschacn-0010: task 0: Exited with exit code 1
srun: Terminating job step 3703527.0
```
where there is still work to do for the accuracy of the functions in some (14/42, ~33%) test cases.

### Reproducibility tests
This is an example of the reproducibility tests' execution on the CSCS machine Kesch:
```
> srun -n 1 -p debug --gres=gpu:1 -t 00:10:00 ./reproTests.out 
[==========] Running 42 tests from 8 test cases.
[----------] Global test environment set-up.
[----------] 1 test from LogTest
[ RUN      ] LogTest.PositiveValues
[       OK ] LogTest.PositiveValues (435 ms)
[----------] 1 test from LogTest (435 ms total)

[----------] 1 test from LogCPUTest
[ RUN      ] LogCPUTest.NegativeValues
[       OK ] LogCPUTest.NegativeValues (21 ms)
[----------] 1 test from LogCPUTest (22 ms total)

[----------] 2 tests from LogErrCPUTest
[ RUN      ] LogErrCPUTest.Zero
[       OK ] LogErrCPUTest.Zero (1 ms)
[ RUN      ] LogErrCPUTest.Infinity
[       OK ] LogErrCPUTest.Infinity (1 ms)
[----------] 2 tests from LogErrCPUTest (2 ms total)
...
[----------] Global test environment tear-down
[==========] 42 tests from 8 test cases ran. (698 ms total)
[  PASSED  ] 42 tests.
```
which shows that all reproducibility tests pass.

### Benchmarks
This is an example of the benchmarks' execution on the CSCS machine Kesch:
```
> srun -n 1 -p debug --gres=gpu:1 -t 00:10:00 ./perfTests.out 
[==========] Running 12 benchmarks.
[ RUN      ] CpuNaturalLogarithm.Random_1000 (10 runs, 100 iterations per run)
[     DONE ] CpuNaturalLogarithm.Random_1000 (1277.780620 ms)
[   RUNS   ]        Average time: 127778.062 us
                         Fastest: 126947.762 us (-830.300 us / -0.650 %)
                         Slowest: 128973.762 us (+1195.700 us / +0.936 %)
                                  
             Average performance: 7.82607 runs/s
                Best performance: 7.87726 runs/s (+0.05119 runs/s / +0.65405 %)
               Worst performance: 7.75352 runs/s (-0.07255 runs/s / -0.92709 %)
[ITERATIONS]        Average time: 1277.781 us
                         Fastest: 1269.478 us (-8.303 us / -0.650 %)
                         Slowest: 1289.738 us (+11.957 us / +0.936 %)
                                  
             Average performance: 782.60695 iterations/s
                Best performance: 787.72558 iterations/s (+5.11863 iterations/s / +0.65405 %)
               Worst performance: 775.35150 iterations/s (-7.25545 iterations/s / -0.92709 %)
[ RUN      ] LibNaturalLogarithm.Random_1000 (10 runs, 100 iterations per run)
[     DONE ] LibNaturalLogarithm.Random_1000 (659.169620 ms)
...
[==========] Ran 12 benchmarks.
```

## Tools
Tools are provided in the `tools/` directory of the project. Please read the corresponding `README.md` located in the directory.

## Issues
Currently some accuracy tests versus the `cmath` (the reference) are failing. Those are listed below:
```
[  FAILED  ] ExpCPUTest.PositiveValues
[  FAILED  ] ExpCPUTest.NegativeValues
[  FAILED  ] ExpCPUTest.Zero
[  FAILED  ] ExpCPUTest.Infinity
[  FAILED  ] PowCPUTest.zeroBaseZeroInfExponentValues
[  FAILED  ] PowCPUTest.posInfBaseZeroInfExponentValues
[  FAILED  ] PowCPUTest.negInfBaseZeroInfExponentValues
[  FAILED  ] PowCPUTest.zeroToOneBasePosInfExponentValues
[  FAILED  ] PowCPUTest.zeroToOneBaseNegInfExponentValues
[  FAILED  ] PowGPUTest.zeroBaseZeroInfExponentValues
[  FAILED  ] PowGPUTest.posInfBaseZeroInfExponentValues
[  FAILED  ] PowGPUTest.negInfBaseZeroInfExponentValues
[  FAILED  ] PowGPUTest.zeroToOneBasePosInfExponentValues
[  FAILED  ] PowGPUTest.zeroToOneBaseNegInfExponentValues
```

## Contribution guidelines

The code, tests and benchmarks are written by PallasKat (Christophe Charpilloz, [MeteoSchweiz](http://www.meteoswiss.admin.ch/home.html)) and the mathematical implementations are inspired by the one proposed by [Nvidia Corporation](http://www.nvidia.com).
