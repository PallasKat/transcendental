#include "cuda_log.h"
#include "hell_log.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <fstream> 
#include <stdio.h>
#include <limits>
#include <iomanip>

/*
union SI {
  unsigned int ii;
  unsigned long int li;
  double dd;
};
*/

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(
  cudaError_t code, 
  const char *file, 
  int line, 
  bool abort=true
) {
 if (code != cudaSuccess) {
  fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
 }
}

// ===========================================================
// FUNCTION TO TEST CPU AND GPU
// ===========================================================

void test_host(double* a, double* b, int n) {
  for (int i = 0; i < n; i++) {
    b[i] = cpu_log(a[i]);
  }
}

__global__ void test_dev(double* a, double* b, int n) {    
  int i = blockIdx.x;
  if (i < n) {
    b[i] = gpumath::cuda_log(a[i]);
  }
}

// ===========================================================
// FUNCTION TO FILL THE INPUT VECTOR
// ===========================================================

// Fill the array ary with random values in [a,b).
void randomFill(double* ary, int a, int b, int n) {    
  for (int i = 0; i < n; i++) {
    ary[i] = (b - a) * ((double) rand() / (double) RAND_MAX ) + a;
  }  
}

void sendToDevice(double*& pDev, double* X, int n) {
  cudaMalloc((void**) &pDev, n*sizeof(double));
  cudaMemcpy(pDev, X, n*sizeof(double), cudaMemcpyHostToDevice);
}

int main(void) {
  // NUMBER OF TESTS
  const int N = 1000000; 
  
  // INPUT
  double X[N];
  randomFill(X, 0, 100, N);
  
  // OUTPUT
  double hostY[N];
  double devY[N];
  
  // ====================================================================
  // GPU
  // ====================================================================
  
  // Send data to device
  double* pDevX = NULL;
  double* pDevY = NULL;
  sendToDevice(pDevX, X, N);
  
  cudaMalloc((void**) &pDevY, N*sizeof(double));    
  test_dev<<<N,1>>>(pDevX, pDevY, N);
  cudaMemcpy(devY, pDevY, N*sizeof(double), cudaMemcpyDeviceToHost);
  
  // ====================================================================
  // CPU
  // ====================================================================
  test_host(X, hostY, N);
  
  // ====================================================================
  // PRINT RESULT TO FILE ERROR 
  // ====================================================================
  
  std::ofstream f;
  f.open("log_gpu_cpu.txt");
    
  bool isBitrepro = true;
  for (int j = 0; j < N; j++) {
    bool isEq = (devY[j] == hostY[j]);
    f << 
        std::setprecision(std::numeric_limits<long double>::digits10 + 1) 
      << X[j] 
      << " " <<
        std::setprecision(std::numeric_limits<long double>::digits10 + 1)
      << hostY[j] 
      << " " <<
        std::setprecision(std::numeric_limits<long double>::digits10 + 1)  
      << devY[j]
      << " "
      << isEq
      << std::endl;
    if (! isEq) {
      isBitrepro = false;
    }
  }
  
  std::cout << "Is bit-reproducible? " << isBitrepro << std::endl;
  f.close();
  
  cudaFree(pDevX);
  cudaFree(pDevY);
}

