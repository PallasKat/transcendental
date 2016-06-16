// google test
#include "gtest/gtest.h"
// printf, scanf, puts, NULL
#include <stdio.h>
// std
#include <iostream>
// srand, rand
#include <stdlib.h>
// test helpers
#include "test_tools.h"
// gpu [log, exp, pow], cpu [log, ...], lib [log, ...]
#include "trans_functors.h"
// to use the functors on vectors
#include "functors.h"
// to use the cuda functors on vectors
#include "cufunctors.h"
// to use the tolerance checkers
#include "tolerance.h"
// to get infinity
#include <limits>

/*
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
*/
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

/*
template<class G, class R>
void testCpuEqOn(
  const std::vector<double> x,
  G cpuFunctor,
  R refFunctor
) {
	testCpuTolOn(x, 0.0, cpuFunctor, refFunctor);  
}

template<class G, class R>
void testGpuEqOn(
  const std::vector<double> x,
  G gpuFunctor,
  R refFunctor
) {
  testGpuTolOn(x, 0.0, gpuFunctor, refFunctor);
}
*/

// =============================================================================
// THE TESTS
// =============================================================================
const size_t N = 1000*10;
const double TOL = 10E-10;

// =============================================================================
// TESTING THE LOGARITHM FUNCTIONS
// =============================================================================

void testCpuLogTolOn(
  const std::vector<double> x,
  const double tol
) {
  testCpuTolOn(x, tol, CpuLog(), LibLog());
}

void testGpuLogTolOn(
  const std::vector<double> x,
  const double tol
) {
  testGpuTolOn(x, tol, GpuLog(), LibLog());
}

void testCpuLogEqOn(const std::vector<double> x) {
	//testCpuEqOn(x, CpuLog(), LibLog());
	testCpuLogTolOn(x, 0.0);
}

void testGpuLogEqOn(const std::vector<double> x) {
	//testGpuEqOn(x, GpuLog(), LibLog());
	testGpuLogTolOn(x, 0.0);
}

TEST(LogCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testCpuLogTolOn(x, TOL);
}

TEST(LogCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testCpuLogTolOn(x, TOL);
}

TEST(LogErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testCpuLogEqOn(x);
}

TEST(LogErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testCpuLogEqOn(x);
}

TEST(LogGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testGpuLogTolOn(x, TOL);
}

TEST(LogGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testGpuLogTolOn(x, TOL);
}

TEST(LogGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testGpuLogEqOn(x);
}

TEST(LogGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testGpuLogEqOn(x);
}

// =============================================================================
// ENTRY POINT
// =============================================================================
int main(int argc, char **argv) {
  //srand(time(NULL));
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
