#!/bin/bash

g++ --version
nvcc --version

rm cpu_mch_math.o b.out b.sass a.out a.sass

g++ -c cpu_mch_math.cpp -O3 -ffp-contract=off
nvcc -arch=sm_37 --fmad=false cpu_mch_math.o main.cu -o a.out
cuobjdump -sass a.out > a.sass

rm cpu_mch_math.o

g++ -c cpu_mch_math.cpp -O3 -ffp-contract=off
nvcc -arch=sm_37 cpu_mch_math.o main.cu -o b.out
cuobjdump -sass b.out > b.sass

