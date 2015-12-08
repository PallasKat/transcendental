// ptr
#include <memory>
// cuda helpers (ie sendToDevice)
#include "cuda_tools.h"

// =============================================================================
// FUNCTORS AND FUNCTOR HELPERS
// =============================================================================

// --------------------------
// GPU FUNCTORS
// --------------------------
template<class F> __global__
void GpuOperation1(const double* x, double* y, unsigned int n, F func) {
  unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n) {
    y[i] = func(x[i]);
  }
}

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

template<class F>
std::vector<double> applyGpuOp1(
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

  std::vector<double> u(n);
  u.assign(y, y + n);
  delete[] y;

  // copy elision by the compiler ?
  return u;
}

// --------------------------
// CPU FUNCTORS
// --------------------------
template <class F>
std::unique_ptr<std::vector<double>> applyCpuOp1(
  const std::vector<double>& x,
  F functor
) {
  auto y = std::unique_ptr<std::vector<double>>(
    new std::vector<double>(x.size())
  );
  for (auto i = 0; i < x.size(); i++) {
    (*y)[i] = functor(x[i]);
  }
  return y;
}
