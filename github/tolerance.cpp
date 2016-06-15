#include "tolerance.h"
// google test
#include "gtest/gtest.h"
// printf, scanf, puts, NULL
#include <stdio.h>
// std
#include <iostream>
// srand, rand
#include <stdlib.h>
// to use the functors on vectors
#include "functors.h"
// to use the cuda functors on vectors
#include "cufunctors.h"

void verifyTol(
  const std::vector<double>& expectedVals,
  const std::vector<double>& values,
  const std::vector<double>& x,
  const std::vector<double>& y,
  const double tol
) {
  //printf("[VERIF] %f => %f; %f\n", x[0], values[0], expectedVals[0]);
  if (expectedVals.size() != values.size()) {
    FAIL() << "The vectors have different lengths.";
  }

  for (auto i = 0; i < values.size(); i++) {
    double err = relativeError(values[i], expectedVals[i]);
    if (isnan(expectedVals[i]) && isnan(values[i])) {
      SUCCEED();
    } else if (isinf(expectedVals[i]) && isinf(values[i])) {
      ASSERT_EQ(expectedVals[i], values[i]);
    } else {
      if (!x.empty() && !y.empty()) {
        ASSERT_LE(err, tol) << "with x, y: " << x[i] << ", " << y[i]
                            << " and exp, val: "
                            << expectedVals[i] << ", " << values[i];
      } else if (!x.empty()) {
        ASSERT_LE(err, tol) << "with x: " << x[i]
                            << " and exp, val: "
                            << expectedVals[i] << ", " << values[i];
      } else {
        ASSERT_LE(err, tol) << "with: " << expectedVals[i] << ", " << values[i];
      }
    }
  }
}

void verifyTol(
  const std::vector<double>& expectedVals,
  const std::vector<double>& values,
  const std::vector<double>& x,
  const double tol
) {
  std::vector<double> emptiness;
  verifyTol(expectedVals, values, x, emptiness, tol);
}

void verifyTol(
  const std::vector<double>& expectedVals,
  const std::vector<double>& values,
  const double tol
) {
  std::vector<double> emptiness;
  verifyTol(expectedVals, values, emptiness, emptiness, tol);
}

template<class G, class R>
void testCpuTolOn(
  const std::vector<double> x,
  const double tol,
  G cpuFunctor,
  R refFunctor
) {
  std::vector<double> crClimValues = applyCpuOp1(x, cpuFunctor);
  std::vector<double> expectedValues = applyCpuOp1(x, refFunctor);
  verifyTol(expectedValues, crClimValues, x, tol);
}

template<class G, class R>
void testGpuTolOn(
  const std::vector<double> x,
  const double tol,
  G gpuFunctor,
  R refFunctor
) {
  std::vector<double> crClimValues = applyGpuOp1(x, gpuFunctor);
  std::vector<double> expectedValues = applyCpuOp1(x, refFunctor);
  verifyTol(expectedValues, crClimValues, x, tol);
}
