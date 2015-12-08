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

// size of the imputs or preimages
const int N = 1000*1000;
// tolerance for the error
const double TOL = 10E-10;

__device__ double incr(double x) {
  return x + 1.0;
}

// =============================================================================
// FUNCTION EVALUATION HELPERS
// =============================================================================

// evaluate a function of the form y = f(x)
__global__ void evalCuFct1(double* x, double* y, int n, double (*f)(double)) {
  int i = blockIdx.x;
  if (i < n) {
    printf("> %f\n", x[i]);
    y[i] = (*f)(x[i]);
    //y[i] = incr(x[i]);
    printf(">> %f %f\n", x[i], y[i]);
  }
}

// copy the data to device memery and evaluate a function of the form y = f(x)
double* sendAndEval1(double* x, int n, double (*f)(double)) {
  // copy preimages to device
  double* pDevX = NULL;
  sendToDevice(pDevX, x, n);
  
  // images
  double* pDevY = NULL;
  gpuErrchk(cudaMalloc((void**) &pDevY, n*sizeof(double)));  
  
  // function application
  evalCuFct1<<<n,1>>>(pDevX, pDevY, n, f);
  gpuErrchk(cudaPeekAtLastError());
  
  // copy images from device to host  
  double* y = new double[n];
  gpuErrchk(cudaMemcpy(y, pDevY, n*sizeof(double), cudaMemcpyDeviceToHost));
  
  // returning the computed images
  return y;
}

// =============================================================================
// ENTRY POINT
// =============================================================================
int main(int argc, char **argv) {
  double* x = new double[N];
  valueFill(x, 1.0, N);
//  double* y = sendAndEval1(x, N, incr);


  // copy preimages to device
  double* pDevX = NULL;
  sendToDevice(pDevX, x, N);
  
  // images
  double* pDevY = NULL;
  gpuErrchk(cudaMalloc((void**) &pDevY, N*sizeof(double)));  
  
  // function application
  evalCuFct1<<<N,1>>>(pDevX, pDevY, N, incr);
  gpuErrchk(cudaPeekAtLastError());
  
  // copy images from device to host  
  double* y = new double[N];
  gpuErrchk(cudaMemcpy(y, pDevY, N*sizeof(double), cudaMemcpyDeviceToHost));
  


  return 0;
}
