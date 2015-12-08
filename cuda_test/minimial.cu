// google test
#include "gtest/gtest.h"
// printf, scanf, puts, NULL
#include <stdio.h>
// test helpers
#include "test_tools.h"
// mch cpu math
#include "cpu_mch_math.h"
// mch gpu math
#include "gpu_mch_math.h"
// cuda helpers
#include "cuda_tools.h"
// lib math
#include <cmath>
// ptr
#include <memory>

class GpuLog {
  public:
    __device__ double operator() (double x) const
    {
      return friendly_log(x);
    }
};

class CpuLog {
  public:
    double operator() (double x) const
    {
      return cpu_friendly_log(x);
    }
};

class LibLog {
  public:
    double operator() (double x) const
    {
      return log(x);
    }
};

template<class F> __global__
void GpuOperation1(const double* x, double* y, unsigned int n, F func) {
  unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n) {
    y[i] = func(x[i]);
  }
}

template <class F>
double* applyGpuOp1(double* x, int n, F functor) {
  const int blocks = n;
  const int threads = 1;

  double* pDevX = NULL;
  sendToDevice(pDevX, x, n);

  double* pDevY = NULL;
  double* y = new double[n];
  gpuErrchk(cudaMalloc((void**) &pDevY, n*sizeof(double)));

  GpuOperation1<<<blocks, threads>>>(pDevX, pDevY, n, functor);
  gpuErrchk(cudaMemcpy(y, pDevY, n*sizeof(double), cudaMemcpyDeviceToHost));
  return y;
}

/*
template<class F> double* CpuOperation1(
  const double* x,
  unsigned int n,
  F func
) {
  double* y = new double[n];
  for (unsigned int i = 0; i < n; i++) {
    y[i] = func(x[i]);
  }
  return y;
}
*/
template <class F>
double* applyCpuOp1(const double* x, size_t n, F functor) {
  double* y = new double[n];
  for (size_t i = 0; i < n; i++) {
    y[i] = functor(x[i]);
  }
  return y;
}

template <class F>
std::unique_ptr<double[]> applyCpuOp1b(const double* x, size_t n, F functor) {
  std::unique_ptr<double[]> y(new double[n]);
  for (size_t i = 0; i < n; i++) {
    y[i] = functor(x[i]);
  }
  return y;
}


void verifyTol(
  const double* expectedValues,
  const double* values,
  size_t n,
  double tol
) {
  for (size_t i = 0; i < n; i++) {
    double err = relativeError(values[i], expectedValues[i]);
    if (isnan(expectedValues[i]) && isnan(values[i])) {
      SUCCEED();
    } else if (isinf(expectedValues[i]) && isinf(values[i])) {
      if (expectedValues[i] < 0 && values[i] < 0) {
        SUCCEED();
      } else {
        FAIL() << "One value is inf and the other is -inf.";
      }
    } else {
      ASSERT_LE(err, tol);
    }
  }
}

template <class F>
std::vector<double> applyCpuOp1c(const std::vector<double>& x, F functor) {
  std::vector<double> y(x.size());
  for (size_t i = 0; i < x.size(); i++) {
    y[i] = functor(x[i]);
  }
  return y;
}

template <class F>
std::unique_ptr<std::vector<double>> applyCpuOp1d(const std::vector<double>& x, F functor) {
  auto w = new std::vector<double>(x.size());
  std::unique_ptr<std::vector<double>> y(w);
  //auto y = std::unique_ptr<std::vector<double>>(new std::vector<double>(x.size()));
  for (auto i = 0; i < x.size(); i++) {
    (*y)[i] = functor(x[i]);
  }
  return y;
}


// Host code
int main()
{
  const int N = 10;

//  double* x = &x[0];
  double* x = new double[N];
  randomFill(x, N);
  /*
  double* y = new double[N];
  valueFill(y, 0, N);
  */
  double* y = applyGpuOp1(x, N, GpuLog());
  double* yy = applyCpuOp1(x, N, CpuLog());
  double* yyy = applyCpuOp1(x, N, LibLog());

  // reference
  std::vector<double> xx(N, 0);
  std::vector<double> yyyy = applyCpuOp1c(xx, LibLog());


  /*
  double* pDevX = NULL;
  sendToDevice(pDevX, x, N);
  double* pDevY = NULL;
  gpuErrchk(cudaMalloc((void**) &pDevY, N*sizeof(double)));

  GpuOperation1<<<blocks, threads>>>(pDevX, pDevY, N, GpuLog());
  gpuErrchk(cudaMemcpy(y, pDevY, N*sizeof(double), cudaMemcpyDeviceToHost));
  */
  for (unsigned int i = 0; i < N; i++) {
    fprintf(stderr, "%i: %f %f\n", i, x[i], y[i]);
  }
}
