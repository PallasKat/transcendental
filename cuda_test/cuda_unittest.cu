// google test
#include "gtest/gtest.h"
// printf, scanf, puts, NULL
#include <stdio.h>
// srand, rand
#include <stdlib.h>
// test helpers
#include "test_tools.h"
// cuda helpers
#include "cuda_tools.h"
// ptr
#include <memory>
// gpulog, cpulog, liblog
#include "log_func.h"

const int N = 10;
const double TOL = 10E-10;

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

template <class F>
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

template <class F>
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

// =============================================================================
// VERIFICATION HELPERS
// =============================================================================

void verifyTol(
  const std::vector<double>& expectedValues,
  const std::vector<double>& values,
  const double tol
) {
  if (expectedValues.size() != values.size()) {
    FAIL() << "The vectors have different lengths.";
  }

  for (auto i = 0; i < values.size(); i++) {
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

void verifyEq(
  const std::vector<double>& expectedValues,
  const std::vector<double>& values
) {
  if (expectedValues.size() != values.size()) {
    FAIL() << "The vectors have different lengths.";
  }

  for (int i = 0; i < values.size(); i++) {
    if (isnan(expectedValues[i]) && isnan(values[i])) {
      SUCCEED();
    } else {
      ASSERT_EQ(expectedValues[i], values[i]);
    }
  }
}

// =============================================================================
// TESTING THE LOGARITHM FUNCTIONS
// =============================================================================

TEST(LogCPUTest, PositiveValues) {
  std::unique_ptr<std::vector<double>> x = randomFill(0, 1, N);
  std::unique_ptr<std::vector<double>> expectedValues = applyCpuOp1(*x.get(), CpuLog());
  std::unique_ptr<std::vector<double>> crClimValues = applyCpuOp1(*x.get(), LibLog());
  verifyTol(*expectedValues.get(), *crClimValues.get(), TOL);
}

TEST(LogGPUTest, PositiveValues) {
  std::unique_ptr<std::vector<double>> x = randomFill(0, 1, N);
  std::vector<double> xx = *x.get();
  //std::unique_ptr<std::vector<double>> expectedValues = applyGpuOp1(xx, GpuLog());
  std::vector<double> expectedValues = applyGpuOp1(xx, GpuLog());
  std::unique_ptr<std::vector<double>> crClimValues = applyCpuOp1(xx, LibLog());
  verifyTol(expectedValues, *crClimValues.get(), TOL);
}

// =============================================================================
// ENTRY POINT
// =============================================================================
int main(int argc, char **argv) {
  //srand(time(NULL));
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
