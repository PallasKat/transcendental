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
// to get infinity
#include <limits>

// =============================================================================
// VERIFICATION HELPERS
// =============================================================================

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

void verifyEq(
  const std::vector<double>& expectedVals,
  const std::vector<double>& values,
  const std::vector<double>& x,
  const std::vector<double>& y
) {
  if (expectedVals.size() != values.size()) {
    FAIL() << "The vectors have different lengths.";
  }

  for (int i = 0; i < values.size(); i++) {
    if (isnan(expectedVals[i]) && isnan(values[i])) {
      SUCCEED();
    } else {
      if (!x.empty() && !y.empty()) {
        ASSERT_EQ(expectedVals[i], values[i]) << "with x, y: "
                                              << x[i] << ", " << y[i];
      } else if (!x.empty()) {
        ASSERT_EQ(expectedVals[i], values[i]) << "with x: " << x[i];
      } else {
        ASSERT_EQ(expectedVals[i], values[i]);
      }
    }
  }
}

void verifyEq(
  const std::vector<double>& expectedVals,
  const std::vector<double>& values,
  const std::vector<double>& x
) {
  std::vector<double> emptiness;
  verifyEq(expectedVals, values, x, emptiness);
}

void verifyEq(
  const std::vector<double>& expectedVals,
  const std::vector<double>& values
) {
  std::vector<double> emptiness;
  verifyEq(expectedVals, values, emptiness, emptiness);
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
void testCpuTolOn(
  const std::vector<double> x,
  const std::vector<double> y,
  const double tol,
  G cpuFunctor,
  R refFunctor
) {
  std::vector<double> crClimValues = applyCpuOp2(x, y, cpuFunctor);
  std::vector<double> expectedValues = applyCpuOp2(x, y, refFunctor);
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

// =============================================================================
// THE TESTS
// =============================================================================
const size_t N = 1000*1000*10;
const double TOL = 10E-10;

// =============================================================================
// TESTING THE LOGARITHM FUNCTIONS
// =============================================================================

// -----------------
// CPU
// -----------------

void testCpuLogTolOn(
  const std::vector<double> x,
  const double tol
) {
  testCpuTolOn(x, tol, CpuLog(), LibLog());
  /*
  std::vector<double> crClimValues = applyCpuOp1(x, CpuLog());
  std::vector<double> expectedValues = applyCpuOp1(x, LibLog());
  verifyTol(expectedValues, crClimValues, x, TOL);
  */
}

void testCpuLogEqOn(const std::vector<double> x) {
  std::vector<double> crClimValues = applyCpuOp1(x, CpuLog());
  std::vector<double> expectedValues = applyCpuOp1(x, LibLog());
  verifyEq(expectedValues, crClimValues, x);
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

// -----------------
// GPU
// -----------------
void testGpuLogTolOn(
  const std::vector<double> x,
  const double tol
) {
  testGpuTolOn(x, tol, GpuLog(), LibLog());
  /*
  std::vector<double> crClimValues = applyGpuOp1(x, GpuLog());
  std::vector<double> expectedValues = applyCpuOp1(x, LibLog());
  verifyTol(expectedValues, crClimValues, x, tol);
  */
}

void testGpuLogEqOn(const std::vector<double> x) {
  std::vector<double> crClimValues = applyGpuOp1(x, GpuLog());
  std::vector<double> expectedValues = applyCpuOp1(x, LibLog());
  verifyEq(expectedValues, crClimValues, x);
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
// TESTING THE EXPONENTIAL FUNCTIONS
// =============================================================================

// -----------------
// CPU
// -----------------
void testCpuExpTolOn(
  const std::vector<double> x,
  const double tol
) {
  testCpuTolOn(x, tol, CpuExp(), LibExp());
  /*
  std::vector<double> crClimValues = applyCpuOp1(x, CpuExp());
  std::vector<double> expectedValues = applyCpuOp1(x, LibExp());
  verifyTol(expectedValues, crClimValues, TOL);
  */
}

void testCpuExpEqOn(const std::vector<double> x) {
  std::vector<double> crClimValues = applyCpuOp1(x, CpuExp());
  std::vector<double> expectedValues = applyCpuOp1(x, LibExp());
  verifyEq(expectedValues, crClimValues);
}

TEST(ExpCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testCpuExpTolOn(x, TOL);
}

TEST(ExpCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testCpuExpTolOn(x, TOL);
}

TEST(ExpCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testCpuExpEqOn(x);
}

TEST(ExpCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testCpuExpEqOn(x);
}

// -----------------
// GPU
// -----------------
void testGpuExpTolOn(
  const std::vector<double> x,
  const double tol
) {
  std::vector<double> crClimValues = applyGpuOp1(x, GpuExp());
  std::vector<double> expectedValues = applyCpuOp1(x, LibExp());
  verifyTol(expectedValues, crClimValues, x, TOL);
}

void testGpuExpEqOn(const std::vector<double> x) {
  std::vector<double> crClimValues = applyGpuOp1(x, GpuExp());
  std::vector<double> expectedValues = applyCpuOp1(x, LibExp());
  verifyEq(expectedValues, crClimValues, x);
}

TEST(ExpGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testGpuExpTolOn(x, TOL);
}

TEST(ExpGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testGpuExpTolOn(x, TOL);
}

TEST(ExpGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testGpuExpEqOn(x);
}

TEST(ExpGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testGpuExpEqOn(x);
}

// =============================================================================
// TESTING THE EXPONENTIATION (z = x^y) FUNCTIONS
// =============================================================================

// -----------------
// CPU
// -----------------
void testCpuPowTolOn(
  const std::vector<double> x,
  const std::vector<double> y,
  const double tol
) {
  testCpuTolOn(x, y, tol, CpuPow(), LibPow());
  /*
  std::vector<double> crClimValues = applyCpuOp2(x, y, CpuPow());
  std::vector<double> expectedValues = applyCpuOp2(x, y, LibPow());
  verifyTol(expectedValues, crClimValues, x, y, TOL);
  */
}

void testCpuPowEqOn(const std::vector<double> x, const std::vector<double> y) {
  std::vector<double> crClimValues = applyCpuOp2(x, y, CpuPow());
  std::vector<double> expectedValues = applyCpuOp2(x, y, LibPow());
  verifyEq(expectedValues, crClimValues, x, y);
}

TEST(PowCPUTest, PositiveBasePositiveExponentValues) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = randomFill(0, 100, N);
  testCpuPowTolOn(x, y, TOL);
}

TEST(PowCPUTest, PositiveBaseNegativeExponentValues) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = randomFill(-100, 0, N);
  testCpuPowTolOn(x, y, TOL);
}

TEST(PowCPUTest, NegativeBasePositiveEvenExponentValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  std::vector<double> y = randomEvenFill(0, 100, N);
  testCpuPowTolOn(x, y, TOL);
}

TEST(PowCPUTest, NegativeBaseNegativeEvenExponentValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  std::vector<double> y = randomEvenFill(-100, 0, N);
  testCpuPowTolOn(x, y, TOL);
}

TEST(PowCPUTest, NegativeBaseAnyExponentValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  std::vector<double> y = randomFill(-100, 100, N);
  testCpuPowTolOn(x, y, TOL);
}

TEST(PowCPUTest, zeroExponentValues) {
  std::vector<double> x = randomFill(-100, 100, N);
  std::vector<double> y = zeroFill(N);
  testCpuPowEqOn(x, y);
}

TEST(PowCPUTest, zeroBaseValues) {
  std::vector<double> x = zeroFill(N);
  std::vector<double> y = randomFill(-100, 100, N);
  testCpuPowEqOn(x, y);
}

TEST(PowCPUTest, zeroBaseZeroInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = zeroFill(3);
  std::vector<double> y = {0.0, posInf, negInf};
  testCpuPowEqOn(x, y);
}

TEST(PowCPUTest, posInfBaseZeroInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = valueFill(posInf, 3);
  std::vector<double> y = {0.0, posInf, negInf};
  testCpuPowEqOn(x, y);
}

TEST(PowCPUTest, negInfBaseZeroInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = valueFill(negInf, 3);
  std::vector<double> y = {0.0, posInf, negInf};
  testCpuPowEqOn(x, y);
}

TEST(PowCPUTest, zeroToOneBasePosInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  std::vector<double> x = randomFill(N);
  std::vector<double> y = valueFill(posInf, N);
  testCpuPowEqOn(x, y);
}

TEST(PowCPUTest, zeroToOneBaseNegInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = randomFill(N);
  std::vector<double> y = valueFill(negInf, N);
  testCpuPowEqOn(x, y);
}

TEST(PowCPUTest, anyBaseAnyIntExponentValues) {
  std::vector<double> x = randomFill(-100, 100, N);
  std::vector<double> y = randomIntFill(-100, 100, N);
  testCpuPowTolOn(x, y, TOL);
}

// -----------------
// GPU
// -----------------
void testGpuPowTolOn(
  const std::vector<double> x,
  const std::vector<double> y,
  const double tol
) {
  std::vector<double> crClimValues = applyGpuOp2(x, y, GpuPow());
  std::vector<double> expectedValues = applyCpuOp2(x, y, LibPow());
  verifyTol(expectedValues, crClimValues, x, y, TOL);
}

void testGpuPowEqOn(const std::vector<double> x, const std::vector<double> y) {
  std::vector<double> crClimValues = applyGpuOp2(x, y, GpuPow());
  std::vector<double> expectedValues = applyCpuOp2(x, y, LibPow());
  verifyEq(expectedValues, crClimValues, x, y);
}

TEST(PowGPUTest, PositiveBasePositiveExponentValues) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = randomFill(0, 100, N);
  testGpuPowTolOn(x, y, TOL);
}

TEST(PowGPUTest, PositiveBaseNegativeExponentValues) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = randomFill(-100, 0, N);
  testGpuPowTolOn(x, y, TOL);
}

TEST(PowGPUTest, NegativeBasePositiveEvenExponentValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  std::vector<double> y = randomEvenFill(0, 100, N);
  testGpuPowTolOn(x, y, TOL);
}

TEST(PowGPUTest, NegativeBaseNegativeEvenExponentValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  std::vector<double> y = randomEvenFill(-100, 0, N);
  testGpuPowTolOn(x, y, TOL);
}

TEST(PowGPUTest, NegativeBaseAnyExponentValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  std::vector<double> y = randomFill(-100, 100, N);
  testCpuPowTolOn(x, y, TOL);
}

TEST(PowGPUTest, zeroExponentValues) {
  std::vector<double> x = randomFill(-100, 100, N);
  std::vector<double> y = zeroFill(N);
  testGpuPowEqOn(x, y);
}

TEST(PowGPUTest, zeroBaseValues) {
  std::vector<double> x = zeroFill(N);
  std::vector<double> y = randomFill(-100, 100, N);
  testGpuPowEqOn(x, y);
}

TEST(PowGPUTest, zeroBaseZeroInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = zeroFill(3);
  std::vector<double> y = {0.0, posInf, negInf};
  testGpuPowEqOn(x, y);
}

TEST(PowGPUTest, posInfBaseZeroInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = valueFill(posInf, 3);
  std::vector<double> y = {0.0, posInf, negInf};
  testGpuPowEqOn(x, y);
}

TEST(PowGPUTest, negInfBaseZeroInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = valueFill(negInf, 3);
  std::vector<double> y = {0.0, posInf, negInf};
  testGpuPowEqOn(x, y);
}

TEST(PowGPUTest, zeroToOneBasePosInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  std::vector<double> x = randomFill(N);
  std::vector<double> y = valueFill(posInf, N);
  testGpuPowEqOn(x, y);
}

TEST(PowGPUTest, zeroToOneBaseNegInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = randomFill(N);
  std::vector<double> y = valueFill(negInf, N);
  testGpuPowEqOn(x, y);
}

TEST(PowGPUTest, anyBaseAnyIntExponentValues) {
  std::vector<double> x = randomFill(-100, 100, N);
  std::vector<double> y = randomIntFill(-100, 100, N);
  testGpuPowTolOn(x, y, TOL);
}

// =============================================================================
// ENTRY POINT
// =============================================================================
int main(int argc, char **argv) {
  //srand(time(NULL));
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
