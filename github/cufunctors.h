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
