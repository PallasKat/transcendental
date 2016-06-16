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
  const std::vector<double> y,
  const double tol,
  G gpuFunctor,
  R refFunctor
) {
  std::vector<double> crClimValues = applyGpuOp2(x, y, gpuFunctor);
  std::vector<double> expectedValues = applyCpuOp2(x, y, refFunctor);
  verifyTol(expectedValues, crClimValues, x, tol);
}

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
	testCpuLogTolOn(x, 0.0);
}

void testGpuLogEqOn(const std::vector<double> x) {
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
// TESTING THE EXPONENTIAL FUNCTIONS
// =============================================================================

void testCpuExpTolOn(
  const std::vector<double> x,
  const double tol
) {
  testCpuTolOn(x, tol, CpuLog(), LibExp());
}

void testGpuExpTolOn(
  const std::vector<double> x,
  const double tol
) {
  testGpuTolOn(x, tol, GpuExp(), LibExp());
}

void testCpuExpEqOn(const std::vector<double> x) {
	testCpuExpTolOn(x, 0.0);
}

void testGpuExpEqOn(const std::vector<double> x) {
	testGpuExpTolOn(x, 0.0);
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

void testCpuPowTolOn(
  const std::vector<double> x,
  const std::vector<double> y,
  const double tol
) {
  testCpuTolOn(x, y, tol, CpuPow(), LibPow());
}

void testGpuPowTolOn(
  const std::vector<double> x,
  const std::vector<double> y,
  const double tol
) {
  testGpuTolOn(x, y, tol, GpuPow(), LibPow());
}

void testCpuPowEqOn(
	const std::vector<double> x,
	const std::vector<double> y
) {
	testCpuPowTolOn(x, y, 0.0);
}

void testGpuPowEqOn(
	const std::vector<double> x,
	const std::vector<double> y
) {
	testGpuPowTolOn(x, y, 0.0);
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
