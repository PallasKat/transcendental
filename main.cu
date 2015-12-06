// cout
#include <iostream>
// io files
#include <fstream>
// printf
#include "stdio.h"
// tic toc
#include <stack>
#include <ctime>
// cuda
#include <cuda.h>
#include <cuda_runtime.h>
// mch math
#include "gpu_mch_math.h"
#include "cpu_mch_math.h"

// ----------------------------------------------------------------------
// CUDA ERROR CHECK
// ----------------------------------------------------------------------
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

// ----------------------------------------------------------------------
// IO UTILS
// ----------------------------------------------------------------------
std::ofstream outfile;
std::ofstream logErrfile;
std::ofstream powErrfile;
std::ofstream expErrfile;

// ----------------------------------------------------------------------
// TIME UTILS
// ----------------------------------------------------------------------
std::stack<clock_t> tictoc_stack;
cudaEvent_t start; 
cudaEvent_t stop;

void tic() {
  tictoc_stack.push(clock());
}

double toc(bool verbose, std::string msg) {
  double tt = ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;

  if (verbose) {
    std::cout << msg << " time elapsed: " << tt << std::endl;
    outfile << msg << " time elapsed: " << tt << "\n";
  }
  tictoc_stack.pop();
  return tt;
}

void gTic() {
  cudaEventRecord(start, 0);
}

float gToc(bool verbose, std::string msg) {
  float time;
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  if (verbose) {
    std::cout << msg << " time elapsed: " << time << std::endl;
    outfile << msg << " time elapsed: " << time << "\n";
  }
  return time;
}

// ----------------------------------------------------------------------
// DEVICE TESTS
// ----------------------------------------------------------------------

// ===============
// LOG
// ===============

__global__ void test_device_log(double* x, double* z, int n) {    
  int i = blockIdx.x;
  if (i < n) {
    z[i] = friendly_log(x[i]);
  }
}

__global__ void test_device_cuda_log(double* x, double* z, int n) {    
  int i = blockIdx.x;
  if (i < n) {
    z[i] = log(x[i]);
  }
}

__global__ void err_device_cuda_log(double* x, double* z, int n) {    
  int i = blockIdx.x;
  if (i < n) {
    z[i] = log(x[i]) - friendly_log(x[i]);
  }
}

// ===============
// POW
// ===============
__global__ void test_device_pow(double* x, double* y, double* z, int n) {    
  int i = blockIdx.x;
  if (i < n) {
    z[i] = friendly_pow(x[i], y[i]);
  }
}

__global__ void test_device_cuda_pow(double* x, double* y, double* z, int n) {    
  int i = blockIdx.x;
  if (i < n) {
    z[i] = pow(x[i], y[i]);
  }
}

__global__ void err_device_cuda_pow(double* x, double* y, double* z, int n) {    
  int i = blockIdx.x;
  if (i < n) {
    z[i] = pow(x[i], y[i]) - friendly_pow(x[i], y[i]);
  }
}

// ===============
// EXP
// ===============
__global__ void test_device_exp(double* x, double* z, int n) {    
  int i = blockIdx.x;
  if (i < n) {
    z[i] = friendly_exp(x[i]);
  }
}

__global__ void test_device_cuda_exp(double* x, double* z, int n) {    
  int i = blockIdx.x;
  if (i < n) {
    z[i] = exp(x[i]);
  }
}

__global__ void err_device_cuda_exp(double* x, double* z, int n) {    
  int i = blockIdx.x;
  if (i < n) {
    z[i] = exp(x[i]) - friendly_exp(x[i]);
  }
}

// ----------------------------------------------------------------------
// HOST TESTS
// ----------------------------------------------------------------------

// ===============
// LOG
// ===============
void test_host_log(double* x, double* z, int n) {    
  for (int i = 0; i < n; i++) {
    z[i] = cpu_friendly_log(x[i]);
  }
}

void test_host_lib_log(double* x, double* z, int n) {    
  for (int i = 0; i < n; i++) {
    z[i] = log(x[i]);
  }
}

void err_host_log(double* x, double* z, int n) {    
  for (int i = 0; i < n; i++) {
    z[i] = log(x[i]) - cpu_friendly_log(x[i]);
  }
}

// ===============
// POW
// ===============
void test_host_pow(double* x, double* y, double* z, int n) {    
  for (int i = 0; i < n; i++) {
    z[i] = cpu_friendly_pow(x[i], y[i]);
    //z[i] = cpu_friendly_log(y[i]*cpu_friendly_exp(x[i]));
  }
}

void test_host_lib_pow(double* x, double* y, double* z, int n) {    
  for (int i = 0; i < n; i++) {
    z[i] = pow(x[i], y[i]);
    //z[i] = log(y[i]*exp(x[i]));
  }
}

void err_host_pow(double* x, double* y, double* z, int n) {    
  for (int i = 0; i < n; i++) {
    z[i] = pow(x[i], y[i]) - cpu_friendly_pow(x[i], y[i]);
    //z[i] = cpu_friendly_log(y[i]*cpu_friendly_exp(x[i])) - log(y[i]*exp(x[i]));
  }
}

// ===============
// EXP
// ===============
void test_host_exp(double* x, double* z, int n) {    
  for (int i = 0; i < n; i++) {
    z[i] = cpu_friendly_exp(x[i]);
  }
}

void test_host_lib_exp(double* x, double* z, int n) {    
  for (int i = 0; i < n; i++) {
    z[i] = exp(x[i]);
  }
}

void err_host_exp(double* x, double* z, int n) {    
  for (int i = 0; i < n; i++) {
    z[i] = exp(x[i]) - cpu_friendly_exp(x[i]);
  }
}

// ----------------------------------------------------------------------

// Fill the array ary with random values in [a,b).
void randomFill(double* ary, int a, int b, int n) {    
  for (int i = 0; i < n; i++) {
    ary[i] = (b - a) * ((double) rand() / (double) RAND_MAX ) + a;
  }  
}

void randomFillTruncation(double* ary, double* ref, int a, int b, int n) {   
  randomFill(ary, a, b, n);
  for (int i = 0; i < n; i++) {
    if (ref[i] < 0) {
      ary[i] = trunc(ary[i]);
    }
  }  
}

// ----------------------------------------------------------------------
// CUDA HELPERS
// ----------------------------------------------------------------------

void sendToDevice(double*& pDev, double* X, int n) {
  gpuErrchk(cudaMalloc((void**) &pDev, n*sizeof(double)));
  gpuErrchk(cudaMemcpy(pDev, X, n*sizeof(double), cudaMemcpyHostToDevice));
}

void freeThem(double* pDevX, double* pDevY, double* pDevZ, double* pDevLibZ) {
  gpuErrchk(cudaFree(pDevX));
  gpuErrchk(cudaFree(pDevY));
  gpuErrchk(cudaFree(pDevZ)); 
  gpuErrchk(cudaFree(pDevLibZ));
}

int isBitRepro(double* x, double* c, double* g, int n) {
  int ok = 1;
  for (int i = 0; i < n; i++) {
    if (c[i] != g[i] && c[i] == c[i] && g[i] == g[i]) {
      outfile << i << ": " << x[i] << ": " << c[i] << " and " << g[i] << "\n";
      ok = 0;
    }
  }
  return ok;
}

// ----------------------------------------------------------------------
// ARRAY HELPERS
// ----------------------------------------------------------------------

void printary(double* ary, int n) {
  for (int i = 0; i < n; i++) {
    std::cout << ary[i] << ", ";
  }
  std::cout << std::endl;
}

bool testNan(double z, double r) {
  bool bothNan = (z != z) && (r != r);
  bool sameSign = ((z < 0.0 && r < 0.0) || (z > 0.0 && r > 0.0));
  return bothNan && sameSign;
}

void writeXYerr(
  double* X, 
  double* CZ, 
  double* CR,
  double* GZ, 
  double* GR,  
  int n, 
  std::ofstream& ofs
) {
  ofs << "i x cpu_mch_y cpu_lib_y cpu_err gpu_mch_y gpu_lib_y gpu_err\n";
  for (int i = 0; i < n; i++) {
    ofs << i << " " 
        << X[i] << " " 
        << CZ[i] << " " 
        << CR[i] << " ";
    if (testNan(CZ[i], CR[i])) {
      ofs << 0.0 << " ";
    } else {
      ofs << CZ[i] - CR[i] << " ";
    }
    
    ofs << GZ[i] << " " 
        << GR[i] << " ";
        
    if (testNan(GZ[i], GR[i])) {
      ofs << 0.0 << " ";
    } else {
      ofs << GZ[i] - GR[i] << "\n";
    }
  }
}

void writeXYZerr(
  double* X, 
  double* Y, 
  double* CZ, 
  double* CR,
  double* GZ, 
  double* GR,  
  int n, 
  std::ofstream& ofs
) {
  ofs << "i x y cpu_mch_z cpu_lib_z cpu_err gpu_mch_z gpu_lib_z gpu_err\n";
  for (int i = 0; i < n; i++) {
    ofs << i << " " 
        << X[i] << " "
        << Y[i] << " " 
        << CZ[i] << " " 
        << CR[i] << " ";
    if (testNan(CZ[i], CR[i])) {
      ofs << 0.0 << " ";
    } else {
      ofs << CZ[i] - CR[i] << " ";
    }
    
    ofs << GZ[i] << " " 
        << GR[i] << " ";
        
    if (testNan(GZ[i], GR[i])) {
      ofs << 0.0 << " ";
    } else {
      ofs << GZ[i] - GR[i] << "\n";
    }
  }
}

// ----------------------------------------------------------------------
// MAIN
// ----------------------------------------------------------------------

int main(void) {
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  outfile.open("dump.txt");
  
  logErrfile.open("log_err.txt");
  powErrfile.open("pow_err.txt");
  expErrfile.open("exp_err.txt");
  
  //int N = 4*1000*100; // 3.2 MB
  int N = 4*1000*1000; // 32 MB
  //int N = 4*1000*1000*10; // 320 MB
  //int N = 4*1000*1000*100; // 3.2 GB
  
  int zero = 0;
//  int left = -20;
//  int right = 20;
  int left = 0;
  int right = 1.4;
  
  // INPUTS
  double* X = new double[N];
  double* pDevX = NULL;  
  
  double* Y = new double[N];
  double* pDevY = NULL;  
  
  // OUTPUTS
  double* CZ = new double[N];
  double* libCZ = new double[N];  
  
  double* GZ = new double[N];  
  double* pDevZ = NULL;
  double* libGZ = new double[N];   
  double* pDevLibZ = NULL;
  
  // ==============================================================  
  // LOGARITHM
  // ==============================================================  
  
  bool br = 0;
  float mch_gpu_t = 0.0;
  float mch_cpu_t = 0.0;
  
  float lib_gpu_t = 0.0;
  float lib_cpu_t = 0.0;

  randomFill(X, zero, right, N);
  randomFill(Y, zero, right, N);
  sendToDevice(pDevX, X, N);
  sendToDevice(pDevY, Y, N);
  gpuErrchk(cudaMalloc((void**) &pDevZ, N*sizeof(double)));
  gpuErrchk(cudaMalloc((void**) &pDevLibZ, N*sizeof(double)));
  
  // -----------------------
  // MCH GPU
  // -----------------------  
  gTic();
  test_device_log<<<N,1>>>(pDevX, pDevZ, N);    
  gpuErrchk(cudaPeekAtLastError());
  mch_gpu_t += gToc(true, "Device mch log");
  gpuErrchk(cudaMemcpy(GZ, pDevZ, N*sizeof(double), cudaMemcpyDeviceToHost));  
  
  // -----------------------
  // MCH CPU
  // -----------------------  
  tic();
  test_host_log(X, CZ, N);    
  mch_cpu_t += toc(true, "Host mch log");
  
  br = isBitRepro(X, CZ, GZ, N);
  
  // -----------------------
  // LIB GPU
  // -----------------------
  gTic();
  test_device_cuda_log<<<N,1>>>(pDevX, pDevLibZ, N);
  lib_gpu_t += gToc(true, "Device lib log");
  gpuErrchk(cudaMemcpy(libGZ, pDevLibZ, N*sizeof(double), cudaMemcpyDeviceToHost));
  
  // -----------------------
  // LIB CPU
  // -----------------------
  tic();
  test_host_lib_log(X, libCZ, N);
  lib_cpu_t += toc(true, "Host lib log");
  
  freeThem(pDevX, pDevY, pDevZ, pDevLibZ);
  
  writeXYerr(X, CZ, libCZ, GZ, libGZ, N, logErrfile);
  //outfile << "Device cha log time elapsed: " << t << "\n";
  std::cout << "Is log bitrepro? " << br << std::endl;

  // ==============================================================  
  // POWER FUNCTION (EXPONENTIATION)
  // ==============================================================     
  
  mch_gpu_t = 0.0;
  mch_cpu_t = 0.0;
  
  lib_gpu_t = 0.0;
  lib_cpu_t = 0.0;
  
  randomFill(X, left, right, N);
  randomFillTruncation(Y, X, left, right, N);
  
  for (int i = 0; i < N; i++) {
    Y[i] = 287.05/1005.0;
  }
  
  /*
  N = 1;
  double* xx = new double[1];
  xx[0] = X[1];
  double* yy = new double[1];
  yy[0] = Y[1];
  */
  sendToDevice(pDevX, X, N);
  sendToDevice(pDevY, Y, N);
  gpuErrchk(cudaMalloc((void**) &pDevZ, N*sizeof(double)));
  gpuErrchk(cudaMalloc((void**) &pDevLibZ, N*sizeof(double)));
  
  // -----------------------
  // MCH GPU
  // -----------------------  
  gTic();
  test_device_pow<<<N,1>>>(pDevX, pDevY, pDevZ, N);    
  gpuErrchk(cudaPeekAtLastError());
  mch_gpu_t += gToc(true, "Device mch pow");
  gpuErrchk(cudaMemcpy(GZ, pDevZ, N*sizeof(double), cudaMemcpyDeviceToHost));  
  
  // -----------------------
  // MCH CPU
  // -----------------------  
  tic();
  test_host_pow(X, Y, CZ, N);    
  mch_cpu_t += toc(true, "Host mch pow");
  
  br = isBitRepro(X, CZ, GZ, N);
  
  // -----------------------
  // LIB GPU
  // -----------------------
  gTic();
  test_device_cuda_pow<<<N,1>>>(pDevX, pDevY, pDevLibZ, N);
  lib_gpu_t += gToc(true, "Device lib pow");
  gpuErrchk(cudaMemcpy(libGZ, pDevLibZ, N*sizeof(double), cudaMemcpyDeviceToHost));
  
  // -----------------------
  // LIB CPU
  // -----------------------
  tic();
  test_host_lib_pow(X, Y, libCZ, N);
  lib_cpu_t += toc(true, "Host lib pow");
  
  freeThem(pDevX, pDevY, pDevZ, pDevLibZ);
  
  writeXYZerr(X, Y, CZ, libCZ, GZ, libGZ, N, powErrfile);
  //outfile << "Device cha log time elapsed: " << t << "\n";
  std::cout << "Is pow bitrepro? " << br << std::endl;

  // ==============================================================  
  // EXPONENTIAL
  // ==============================================================   
  
  mch_gpu_t = 0.0;
  mch_cpu_t = 0.0;
  
  lib_gpu_t = 0.0;
  lib_cpu_t = 0.0;
  
  randomFill(X, left, right, N);
  randomFill(Y, left, right, N);
  sendToDevice(pDevX, X, N);
  sendToDevice(pDevY, Y, N);
  gpuErrchk(cudaMalloc((void**) &pDevZ, N*sizeof(double)));
  gpuErrchk(cudaMalloc((void**) &pDevLibZ, N*sizeof(double)));
  
  // -----------------------
  // MCH GPU
  // -----------------------    
  gTic();
  test_device_exp<<<N,1>>>(pDevX, pDevZ, N);
  mch_gpu_t += gToc(true, "Device mch exp");
  gpuErrchk(cudaMemcpy(GZ, pDevZ, N*sizeof(double), cudaMemcpyDeviceToHost));  
    
  // -----------------------
  // MCH CPU
  // -----------------------    
  tic();
  test_host_exp(X, CZ, N);    
  mch_cpu_t += toc(true, "Host mch exp");
  
  br = isBitRepro(X, CZ, GZ, N);
  
  // -----------------------
  // LIB GPU
  // -----------------------
  gTic();
  test_device_cuda_exp<<<N,1>>>(pDevX, pDevLibZ, N);
  lib_gpu_t += gToc(true, "Device lib exp");
  gpuErrchk(cudaMemcpy(libGZ, pDevLibZ, N*sizeof(double), cudaMemcpyDeviceToHost));
  
  // -----------------------
  // LIB CPU
  // -----------------------
  tic();
  test_host_lib_exp(X, libCZ, N);
  lib_cpu_t += toc(true, "Host lib exp");
  
  freeThem(pDevX, pDevY, pDevZ, pDevLibZ);
  
  writeXYerr(X, CZ, libCZ, GZ, libGZ, N, expErrfile);  
  std::cout << "Is exp bitrepro? " << br << std::endl;

  // =================================================================  
  // CLOSING AND CLEANING
  // =================================================================  
  
  outfile.close();  
  logErrfile.close();
  powErrfile.close();
  expErrfile.close();
  
  delete[] X;
  delete[] Y;
  delete[] CZ;
  delete[] libCZ;
  delete[] GZ;
  delete[] libGZ;
  
  return 0;
}

