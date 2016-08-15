// ptr
#include <memory>
// cuda helpers (ie sendToDevice)
#include "cuda_tools.h"
// assert
#include <assert.h>
// stdint definitions
#include <stdint.h>

// =============================================================================
// CUDA FUNCTORS AND FUNCTOR HELPERS
// =============================================================================

// -----------------------------------------------------------------------------
// Functor for the functions of the kind y = f(x), Double -> Double
// -----------------------------------------------------------------------------

template<class F> __global__
void GpuOperation1(const double* x, double* y, uint32_t n, F func) {  
  uint32_t i = blockDim.x*blockIdx.x + threadIdx.x;
  while (i < n) {
    y[i] = func(x[i]);
    i += blockDim.x*gridDim.x;
  }
}

template<class F>
std::vector<double> applyGpuOp1(
  const std::vector<double>& v,
  F functor
) {
  const uint32_t n = v.size();

  double* pDevX = NULL;
  sendToDevice(pDevX, v.data(), n);
  //std::vector<double> u = valueFill(0.0, n);
  //printVector(u);
  //sendToDevice(pDevX, u.data(), n);

  double* pDevY = NULL;
  std::vector<double> y(n);
  gpuErrchk(cudaMalloc((void**) &pDevY, n*sizeof(double)));

  // applying operations and copying the result back to the memory
  GpuOperation1<<<128, 128>>>(pDevX, pDevY, n, functor);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaMemcpy(&y[0], pDevY, n*sizeof(double), cudaMemcpyDeviceToHost));

  // freeing memory on the device
  gpuErrchk(cudaFree(pDevX));
  gpuErrchk(cudaFree(pDevY));

  // copy elision by the compiler ?
  return y;
}

template<class F>
std::vector<double> applyGpuBenchOp1(
  double* pDevX,
  double* pDevY,
  size_t n,
  F functor
) {
  // we want the host to wait on the device
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::vector<double> y(n);
  gpuErrchk(cudaMalloc((void**) &pDevY, n*sizeof(double)));

  // applying operations and copying the result back to the memory
  cudaEventRecord(start);
  GpuOperation1<<<128, 128>>>(pDevX, pDevY, n, functor);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // copy elision by the compiler ?
  return y;
}

// FLOATING POINTS

template<class F> __global__
void GpuOperation1(const float* x, float* y, uint32_t n, F func) {
  uint32_t i = blockDim.x*blockIdx.x + threadIdx.x;
  while (i < n) {
    y[i] = func(x[i]);
    i += blockDim.x*gridDim.x;
  }
}

template<class F>
std::vector<float> applyGpuOp1Float( 
  const std::vector<float>& v,
  F functor
) {
  const uint32_t n = v.size();

  float* pDevX = NULL;
  sendToDeviceFloat(pDevX, v.data(), n);
  //std::vector<float> u = valueFill(0.0, n);
  //printVector(u);
  //sendToDevice(pDevX, u.data(), n);

  float* pDevY = NULL;
  std::vector<float> y(n);
  gpuErrchk(cudaMalloc((void**) &pDevY, n*sizeof(float)));

  // applying operations and copying the result back to the memory
  GpuOperation1<<<128, 128>>>(pDevX, pDevY, n, functor);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaMemcpy(&y[0], pDevY, n*sizeof(float), cudaMemcpyDeviceToHost));

  // freeing memory on the device
  gpuErrchk(cudaFree(pDevX));
  gpuErrchk(cudaFree(pDevY));

  // copy elision by the compiler ?
     return y;
}

template<class F>
std::vector<float> applyGpuBenchOp1Float(
  float* pDevX,
  float* pDevY,
  size_t n,
  F functor
) {
  // we want the host to wait on the device
  cudaEvent_t startFloat, stopFloat;
  cudaEventCreate(&startFloat);
  cudaEventCreate(&stopFloat);

  std::vector<float> y(n);
  gpuErrchk(cudaMalloc((void**) &pDevY, n*sizeof(float)));

  // applying operations and copying the result back to the memory
  cudaEventRecord(startFloat);
  GpuOperation1<<<128, 128>>>(pDevX, pDevY, n, functor);
  cudaEventRecord(stopFloat);
  cudaEventSynchronize(stopFloat);

  // copy elision by the compiler ?
  return y;
}

// -----------------------------------------------------------------------------
// Functor for the function of the kind z = f(x, y), (Double x Double) -> Double
// -----------------------------------------------------------------------------

template<class F> __global__
void GpuOperation2(
  const double* x, 
  const double* y, 
  double* z, 
  uint32_t n, 
  F func
) {
  uint32_t i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n) {
    z[i] = func(x[i], y[i]);
  }
}

template<class F>
std::vector<double> applyGpuOp2(
  const std::vector<double>& v,
  const std::vector<double>& u,
  F functor
) {
  assert(v.size() == u.size());
  const uint32_t n = v.size();

  double* pDevX = NULL;
  sendToDevice(pDevX, v.data(), n);

  double* pDevY = NULL;
  sendToDevice(pDevY, u.data(), n);

  double* pDevZ = NULL;
  std::vector<double> z(n);
  gpuErrchk(cudaMalloc((void**) &pDevZ, n*sizeof(double)));

  // applying operations and copying the result back to the memory
  GpuOperation2<<<128, 128>>>(pDevX, pDevY, pDevZ, n, functor);
  gpuErrchk(cudaMemcpy(&z[0], pDevZ, n*sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaPeekAtLastError());

  // freeing memory on the device
  gpuErrchk(cudaFree(pDevX));
  gpuErrchk(cudaFree(pDevY));
  gpuErrchk(cudaFree(pDevZ));

  return z;
}

template<class F>
std::vector<double> applyGpuBenchOp2(
  double* pDevX,
  double* pDevY,
  double* pDevZ,
  size_t n,
  F functor
) {
  // we want the host to wait on the device
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::vector<double> z(n);
  gpuErrchk(cudaMalloc((void**) &pDevZ, n*sizeof(double)));

  // applying operations and copying the result back to the memory
  cudaEventRecord(start);
  GpuOperation2<<<128, 128>>>(pDevX, pDevY, pDevZ, n, functor);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // copy elision by the compiler ?
  return z;
}

// FLOATING POINTS

template<class F> __global__
void GpuOperation2(
  const float* x, 
  const float* y, 
  float* z, 
  uint32_t n, 
  F func
) {
  uint32_t i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n) {
    z[i] = func(x[i], y[i]);
  }
}

template<class F>
std::vector<float> applyGpuOp2Float(
  const std::vector<float>& v,
  const std::vector<float>& u,
  F functor
) {
  assert(v.size() == u.size());
  const uint32_t n = v.size();

  float* pDevX = NULL;
  sendToDeviceFloat(pDevX, v.data(), n);

  float* pDevY = NULL;
  sendToDeviceFloat(pDevY, u.data(), n);

  float* pDevZ = NULL;
  std::vector<float> z(n);
  gpuErrchk(cudaMalloc((void**) &pDevZ, n*sizeof(float)));

  // applying operations and copying the result back to the memory

  GpuOperation2<<<128, 128>>>(pDevX, pDevY, pDevZ, n, functor);
  gpuErrchk(cudaMemcpy(&z[0], pDevZ, n*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaPeekAtLastError());

  // freeing memory on the device
  gpuErrchk(cudaFree(pDevX));
  gpuErrchk(cudaFree(pDevY));
  gpuErrchk(cudaFree(pDevZ));

  return z;
}
/*
template<class F>
std::vector<float> applyGpuBenchOp2Float(
  float* pDevX,
  float* pDevY,
  float* pDevZ,
  size_t n,
  F functor
) {
  // we want the host to wait on the device
  //   cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::vector<float> z(n);
  gpuErrchk(cudaMalloc((void**) &pDevZ, n*sizeof(float)));

  // applying operations and copying the result back to the memory
  cudaEventRecord(start);
  GpuOperation2<<<128, 128>>>(pDevX, pDevY, pDevZ, n, functor);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // copy elision by the compiler ?
     return z;
}*/
