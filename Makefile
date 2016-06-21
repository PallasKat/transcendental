CC=g++
NV=nvcc

CCFLAGS=-std=c++11
NVFLAGS=-rdc=true -arch=sm_37 -dc 

GTEST=../../googletest/
HAYAI=../../hayai/

GINC=-isystem $(GTEST)/googletest/include 
GLIB=${GTEST}/build/googlemock/gtest/libgtest.a
HINC=-I $(HAYAI)/src

protablefct:
	$(CC) $(CCFLAGS) -ffp-contract=off -c portable_math.cpp -o cpu_portable_math.o
	$(NV) $(NVFLAGS) --fmad=false portable_math.cu -o gpu_portable_math.o
	$(NV) $(NVFLAGS) cuda_functors.cu

helpers:
	$(CC) $(CCFLAGS) -c test_tools.cpp
	$(CC) $(CCFLAGS) $(GINC) -c tolerance.cpp

approx: helpers protablefct
	$(NV) -rdc=true -g -arch=sm_37 --std=c++11 $(GINC) approxtests.cu $(GLIB) test_tools.o cpu_portable_math.o gpu_portable_math.o tolerance.o -o approxTests.out

repro: helpers protablefct
	$(NV) -rdc=true -g -arch=sm_37 --std=c++11 $(GINC) bitreprotests.cu $(GLIB) test_tools.o cpu_portable_math.o gpu_portable_math.o tolerance.o -o reproTests.out

bench: protablefct
	$(NV) -rdc=true -g -arch=sm_37 --std=c++11 $(HINC) benchtests.cu test_tools.o cuda_functors.o cpu_portable_math.o gpu_portable_math.o -o perfTests.out

all: approx repro bench
	.PHONY : all

clean:
	rm -rf *.o
	rm -rf *.out
