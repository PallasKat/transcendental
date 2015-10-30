#include "mch_math.h"
#include "mch_cuda_math.h"
//#include "eval_op.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include <stdio.h>
//#include <cmath>

// ===============================================================
// DEVICE (+ ou -) FUNCTIONS
// ===============================================================

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

__global__ void test_cu(double* x, double* y, double* z, int n) {    
  int i = blockIdx.x;
  if (i < n) {
    z[i] = pow(x[i], y[i]) - __mch_cuda_pow(x[i], y[i]);
  }
}

// ===============================================================
// HOST FUNCTIONS
// ===============================================================
void test_host(double* x, double* y, double* z, int n) {    
  for (int i = 0; i < n; i++) {
    z[i] = mch_pow(x[i], y[i]);
    //z[i] = pow(x[i], y[i]);
  }
}


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
  int M = 100;
  
  double XX[M];
  double YY[M];
//  randomFill(XX, -1000, 1000, M);
//  randomFill(YY, -4, 4, M);
  
  const int N = 100; 
  // INPUTS
  double X[N];
  double* pDevX = NULL;  
  
  double Y[N];
  double* pDevY = NULL;  
  
  // OUTPUTS
  double Z[N];
  
  double R[N];
  double* pDevR = NULL;
  
  // VALUES
  //X[0] = XX[22443]; //3.0;
  //Y[0] = YY[22443]; //4.0;
    
  /*
  for (int k = 0; k < N; k++) {
    X[k] = static_cast<double>(k);
    Y[k] = static_cast<double>(k) - 10.0;
  }
  */
  
  
  randomFill(X, -10, 10, N);
  randomFill(Y, -10, 10, N);
  
  //X[0] = static_cast<double>(5);
  //Z[0] = 0.0;
  
  
  
  // ====================================================================
  // GPU
  // ====================================================================
  printf("\n======================\nGPU\n======================\n");
  sendToDevice(pDevX, X, N);
  sendToDevice(pDevY, Y, N);
  
  cudaMalloc((void**) &pDevR, N*sizeof(double));    
  test_cu<<<N,1>>>(pDevX, pDevY, pDevR, N);
  cudaMemcpy(R, pDevR, N*sizeof(double), cudaMemcpyDeviceToHost);    
  
  // ====================================================================
  // CPU
  // ====================================================================
  printf("\n======================\nCPU\n======================\n");
  test_host(X, Y, Z, N);
  
  // ====================================================================
  // ERRORS ?
  // ====================================================================
  
  std::ofstream f;
  f.open("pow_err.txt");
  
  printf("\n======================\nNumber of tested values: %i\n======================\n",N);
  int nn = 0;
  for (int i = 0; i < N; i++) {
    /*
    if (R[i] != Z[i]) {
      if (!(isnan(R[i]) && isnan(Z[i]))) {
        nn = nn + 1;
        std::cout << i 
                << ": " 
                << Z[i] 
                << " == " 
                << R[i] 
                << " ? " 
                << (R[i] == Z[i])
                << " err: "
                << abs(R[i] - Z[i]) 
                << std::endl;
        printDd(Z[i]);
        printDd(R[i]);
      }
    }
    */
    if (!(isnan(R[i]) && isnan(Z[i]))) {
      std::cout << i << ": " << X[i]
//            f << X[i]  
              << " " 
              << Y[i] 
              << " " 
//              << abs(R[i] - Z[i]) 
//              << " "
              << R[i]
              << " "
              << Z[i]
              << std::endl;
    }
  }
  printf("\n======================\nErrors: %i\n======================\n", nn);
  
  f.close();
  
  cudaFree(pDevX);
  cudaFree(pDevY);
}

