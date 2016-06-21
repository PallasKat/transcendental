# README
Two tools are provided with the set of portable functions. The first one is only an illustration to dump SASS for analysis. The second one is an helper that allows to generate functors from a list of functions.

## SASS Dumper
The script is only a list of command to dump the sass from a compiled C++ file. First the mathematical functions `portable_math.cpp` are compiled (the file name is just an example):
```
g++ -c portable_math.cpp -O3 -ffp-contract=off
```
Then it's compiled and linked with it's CUDA counter part (`mainn.cu` is only an example):
```
nvcc -arch=sm_37 --fmad=false portable_math.o main.cu -o exec.out
```
Finally the SASS is dumped:
```
cuobjdump -sass exec.out > exec.sass
```
And the SASS looks like:
```
atbin elf code:
================
arch = sm_37
code version = [1,7]
producer = <unknown>
host = linux
compile_size = 64bit

        code for sm_37

Fatbin elf code:
================
arch = sm_37
code version = [1,7]
producer = cuda
host = linux
compile_size = 64bit

        code for sm_37
                Function : _Z15test_device_powPdS_S_i
        .headerflags    @"EF_CUDA_SM37 EF_CUDA_PTX_SM(EF_CUDA_SM37)"
                                                                                    /* 0x08b0a010b8b0a000 */
        /*0008*/                   MOV R1, c[0x0][0x44];                            /* 0x64c03c00089c0006 */
        /*0010*/                   S2R R0, SR_CTAID.X;                              /* 0x86400000129c0002 */
        /*0018*/                   ISETP.GE.AND P0, PT, R0, c[0x0][0x158], PT;      /* 0x5b681c002b1c001e */
        /*0020*/               @P0 EXIT;                                            /* 0x180000000000003c */
...
Fatbin ptx code:
================
arch = sm_37
code version = [4,2]
producer = cuda
host = linux
compile_size = 64bit
compressed
```

## Functors generator
The functor generator is a python script that generate C++ functors based on the data provided in a `CSV` file. The goal is to help to ease the task of the unit tests writer as functors are not very scalable and impose to repeat code that can be easily generated. The Python code has been developed for Python 2.x as this is the default available on the CSCS environment.

### The CSV description
The `CSV` file should have the following format for its rows:
```
name of the function, name of the portable CPU version, name of the portable GPU version, name of the reference version, number of function arguments
```
for example the following `CSV` fileis valid:
```
exp,cpu_exp,gpu_exp,exp,1
log,cpu_log,gpu_log,log,1
pow,cpu_pow,gpu_pow,pow,2
```
### The execution of the script
The previous `CSV` can be used to produce the following functors. The script is simply directly executed with `python` (for now the `CSV` file name is hard-coded in the script) and produce the following output:
```
> python tf_gen.py 
class CpuExp {
  public:
    double operator() (double x) const {
      return cpu_exp(x);
    }
};
class GpuExp {
  public:
    double operator() (double x) const {
      return gpu_exp(x);
    }
};
...
class LibPow {
  public:
    double operator() (double x, double y) const {
      return pow(x, y);
    }
};
```

## Contribution guidelines
The code is written by PallasKat (Christophe Charpilloz, [MeteoSchweiz](http://www.meteoswiss.admin.ch/home.html)).


