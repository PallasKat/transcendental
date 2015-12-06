#include "cuda_math.h"

__global__ void cuda_log(double* x, double* y, double* z, int n) {
  int i = blockIdx.x;
  if (i < n) {
    z[i] = log(x[i], y[i]);
  }
}

__global__ void cuda_pow(double* x, double* y, double* z, int n) {
  int i = blockIdx.x;
  if (i < n) {
    z[i] = pow(x[i], y[i]);
  }
}

__global__ void cuda_exp(double* x, double* z int n) {
  int i = blockIdx.x;
  if (i < n) {
    z[i] = exp(x[i]);
  }
}

