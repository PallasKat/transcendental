// ptr
#include <memory>
// cuda helpers (ie sendToDevice)
#include "cuda_tools.h"
// assert
#include <assert.h>
// stdint definitions
#include <stdint.h>

// =============================================================================
// FUNCTORS AND FUNCTOR HELPERS
// =============================================================================

// ----------------------------------------------------
// GPU FUNCTORS
// ----------------------------------------------------

// function of the kind y = f(x), Double -> Double
template<class F> __global__
void GpuOperation1(const double* x, double* y, uint32_t n, F func) {
  uint32_t i = blockDim.x*blockIdx.x + threadIdx.x;
  while (i < n) {
    y[i] = func(x[i]);
    i += blockDim.x*gridDim.x;
  }
}

// function of the kind z = f(x, y), (Double x Double) -> Double
template<class F> __global__
void GpuOperation2(const double* x, const double* y, double* z, uint32_t n, F func) {
  uint32_t i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n) {
    z[i] = func(x[i], y[i]);
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

// ----------------------------------------------------
// CPU FUNCTORS
// ----------------------------------------------------
template <class F>
std::vector<double> applyCpuOp1(
  const std::vector<double>& x,
  F functor
) {
  std::vector<double> y(x.size());
  for (auto i = 0; i < x.size(); i++) {
    y[i] = functor(x[i]);
  }
  return y;
}

template <class F>
std::vector<double> applyCpuOp2(
  const std::vector<double>& x,
  const std::vector<double>& y,
  F functor
) {
  assert(x.size() == y.size());
  std::vector<double> z(x.size());
  for (auto i = 0; i < x.size(); i++) {
    z[i] = functor(x[i], y[i]);
  }
  return z;
}

// ROTEN CODE
template<class F>
std::unique_ptr<std::vector<double>> applyGpuOpOld1(
  const std::vector<double>& v,
  F functor
) {
  const int n = v.size();
  const int blocks = n;
  const int threads = 1;

  double* pDevX = NULL;
  sendToDevice(pDevX, v.data(), n);

  double* pDevY = NULL;
  double* y = new double[n];
  gpuErrchk(cudaMalloc((void**) &pDevY, n*sizeof(double)));

  GpuOperation1<<<blocks, threads>>>(pDevX, pDevY, n, functor);
  gpuErrchk(cudaMemcpy(y, pDevY, n*sizeof(double), cudaMemcpyDeviceToHost));

  std::vector<double>* u = new std::vector<double>;
  u->assign(y, y + n);
  delete[] y;

  return std::unique_ptr<std::vector<double>>(u);
}
