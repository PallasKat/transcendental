// printf
#include "stdio.h"
// cuda
#include <cuda.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(
  cudaError_t code,
  const char *file,
  int line,
  bool abort=true
) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
      exit(code);
    }
  }
}

void sendToDevice(double*& ptr, const double* data, const int n) {
  gpuErrchk(cudaMalloc((void**) &ptr, n*sizeof(double)));
  gpuErrchk(cudaMemcpy(ptr, data, n*sizeof(double), cudaMemcpyHostToDevice));
}
