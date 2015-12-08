// printf, scanf, puts, NULL
#include <stdio.h>
// srand, rand
#include <stdlib.h>
// time
#include <time.h>
// cuda
#include <cuda.h>
#include <cuda_runtime.h>
// test helpers
#include "test_tools.h"
// cuda helpers
#include "cuda_tools.h"
// mch cpu math
#include "cpu_mch_math.h"
// mch gpu math
#include "gpu_mch_math.h"

template<class T>
__global__ void execGPU(double* x, double* y, int n)
{
  int i = blockIdx.x;
  if (i < n) {
    y[i] = T::fGpu(x[i]);
  }
}

template<class T>
class IComputation
{
  public:
    virtual ~IComputation() {}
    __device__ static virtual double fGpu(double x) { return -1.0; };
    virtual double fCpu(double x) = 0;
    
    double* applyGpu(double* x, int n) {
      double* pDevX = NULL;
      sendToDevice(pDevX, x, n);

      double* pDevY = NULL;
      gpuErrchk(cudaMalloc((void**) &pDevY, n*sizeof(double)));
      execGPU<T><<<n,1>>>(pDevX, pDevY, n);

      double* y = new double[n];
      gpuErrchk(cudaMemcpy(y, pDevY, n*sizeof(double), cudaMemcpyDeviceToHost));

      return y;
    }

    double* applyCpu(double* x, int n) {
      double* y = new double[n];
      for (unsigned i = 0; i < n; i++) {
        y[i] = fCpu(x[i]);
      }
      return y;
    }
};

/*
class Computation
{
  public:
    virtual ~Computation();
};
*/

class LogComputation// : public IComputation
{
  public:
    __device__ static double fGpu(double x) {
      return friendly_log(x);
    }

    double fCpu(double x) {
      return cpu_friendly_log(x);
    }
};
