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
void testReproOn(
  const std::vector<double> x,
  G gpuFunctor,
  R cpuFunctor
) {  
  std::vector<double> cpuValues = applyCpuOp1(x, cpuFunctor);
  std::vector<double> gpuValues = applyGpuOp1(x, gpuFunctor);
  verifyEq(cpuValues, gpuValues, x);
}

template<class G, class R>
void testReproOn(
  const std::vector<double> x,
  const std::vector<double> y,
  G gpuFunctor,
  R cpuFunctor
) {  
  std::vector<double> cpuValues = applyCpuOp2(x, y, cpuFunctor);
  std::vector<double> gpuValues = applyGpuOp2(x, y, gpuFunctor);
  verifyEq(cpuValues, gpuValues, x);
}

// =============================================================================
// THE TESTS
// =============================================================================
const size_t N = 10*1000;

// =============================================================================
// TESTING THE LOGARITHM FUNCTIONS
// =============================================================================

void testLogReproOn(const std::vector<double> x) {
  testReproOn(x, GpuLog(), CpuLog());
}

TEST(LogTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testLogReproOn(x);
}

TEST(LogCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testLogReproOn(x);
}

TEST(LogErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testLogReproOn(x);
}

TEST(LogErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testLogReproOn(x);
}

TEST(LogGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testLogReproOn(x);
}

TEST(LogGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testLogReproOn(x);
}

TEST(LogGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testLogReproOn(x);
}

TEST(LogGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testLogReproOn(x);
}

// =============================================================================
// TESTING THE EXPONENTIAL FUNCTIONS
// =============================================================================

void testExpReproOn(const std::vector<double> x) {
  testReproOn(x, GpuExp(), CpuExp());
}

TEST(ExpCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testExpReproOn(x);
}

TEST(ExpCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testExpReproOn(x);
}

TEST(ExpCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testExpReproOn(x);
}

TEST(ExpCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testExpReproOn(x);
}

TEST(ExpGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testExpReproOn(x);
}

TEST(ExpGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testExpReproOn(x);
}

TEST(ExpGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testExpReproOn(x);
}

TEST(ExpGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testExpReproOn(x);
}

// =============================================================================
// TESTING THE EXPONENTIATION (z = x^y) FUNCTIONS
// =============================================================================

void testPowReproOn(
  const std::vector<double> x,
  const std::vector<double> y
) {
  testReproOn(x, y, GpuPow(), CpuPow());
}

TEST(PowCPUTest, PositiveBasePositiveExponentValues) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = randomFill(0, 100, N);
  testPowReproOn(x, y);
}

TEST(PowCPUTest, PositiveBaseNegativeExponentValues) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = randomFill(-100, 0, N);
  testPowReproOn(x, y);
}

TEST(PowCPUTest, NegativeBasePositiveEvenExponentValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  std::vector<double> y = randomEvenFill(0, 100, N);
  testPowReproOn(x, y);
}

TEST(PowCPUTest, NegativeBaseNegativeEvenExponentValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  std::vector<double> y = randomEvenFill(-100, 0, N);
  testPowReproOn(x, y);
}

TEST(PowCPUTest, NegativeBaseAnyExponentValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  std::vector<double> y = randomFill(-100, 100, N);
  testPowReproOn(x, y);
}

TEST(PowCPUTest, zeroExponentValues) {
  std::vector<double> x = randomFill(-100, 100, N);
  std::vector<double> y = zeroFill(N);
  testPowReproOn(x, y);
}

TEST(PowCPUTest, zeroBaseValues) {
  std::vector<double> x = zeroFill(N);
  std::vector<double> y = randomFill(-100, 100, N);
  testPowReproOn(x, y);
}

TEST(PowCPUTest, zeroBaseZeroInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = zeroFill(3);
  std::vector<double> y = {0.0, posInf, negInf};
  testPowReproOn(x, y);
}

TEST(PowCPUTest, posInfBaseZeroInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = valueFill(posInf, 3);
  std::vector<double> y = {0.0, posInf, negInf};
  testPowReproOn(x, y);
}

TEST(PowCPUTest, negInfBaseZeroInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = valueFill(negInf, 3);
  std::vector<double> y = {0.0, posInf, negInf};
  testPowReproOn(x, y);
}

TEST(PowCPUTest, zeroToOneBasePosInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  std::vector<double> x = randomFill(N);
  std::vector<double> y = valueFill(posInf, N);
  testPowReproOn(x, y);
}

TEST(PowCPUTest, zeroToOneBaseNegInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = randomFill(N);
  std::vector<double> y = valueFill(negInf, N);
  testPowReproOn(x, y);
}

TEST(PowCPUTest, anyBaseAnyIntExponentValues) {
  std::vector<double> x = randomFill(-100, 100, N);
  std::vector<double> y = randomIntFill(-100, 100, N);
  testPowReproOn(x, y);
}

TEST(PowGPUTest, PositiveBasePositiveExponentValues) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = randomFill(0, 100, N);
  testPowReproOn(x, y);
}

TEST(PowGPUTest, PositiveBaseNegativeExponentValues) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = randomFill(-100, 0, N);
  testPowReproOn(x, y);
}

TEST(PowGPUTest, NegativeBasePositiveEvenExponentValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  std::vector<double> y = randomEvenFill(0, 100, N);
  testPowReproOn(x, y);
}

TEST(PowGPUTest, NegativeBaseNegativeEvenExponentValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  std::vector<double> y = randomEvenFill(-100, 0, N);
  testPowReproOn(x, y);
}

TEST(PowGPUTest, NegativeBaseAnyExponentValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  std::vector<double> y = randomFill(-100, 100, N);
  testPowReproOn(x, y);
}

TEST(PowGPUTest, zeroExponentValues) {
  std::vector<double> x = randomFill(-100, 100, N);
  std::vector<double> y = zeroFill(N);
  testPowReproOn(x, y);
}

TEST(PowGPUTest, zeroBaseValues) {
  std::vector<double> x = zeroFill(N);
  std::vector<double> y = randomFill(-100, 100, N);
  testPowReproOn(x, y);
}

TEST(PowGPUTest, zeroBaseZeroInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = zeroFill(3);
  std::vector<double> y = {0.0, posInf, negInf};
  testPowReproOn(x, y);
}

TEST(PowGPUTest, posInfBaseZeroInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = valueFill(posInf, 3);
  std::vector<double> y = {0.0, posInf, negInf};
  testPowReproOn(x, y);
}

TEST(PowGPUTest, negInfBaseZeroInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = valueFill(negInf, 3);
  std::vector<double> y = {0.0, posInf, negInf};
  testPowReproOn(x, y);
}

TEST(PowGPUTest, zeroToOneBasePosInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  std::vector<double> x = randomFill(N);
  std::vector<double> y = valueFill(posInf, N);
  testPowReproOn(x, y);
}

TEST(PowGPUTest, zeroToOneBaseNegInfExponentValues) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = randomFill(N);
  std::vector<double> y = valueFill(negInf, N);
  testPowReproOn(x, y);
}

TEST(PowGPUTest, anyBaseAnyIntExponentValues) {
  std::vector<double> x = randomFill(-100, 100, N);
  std::vector<double> y = randomIntFill(-100, 100, N);
  testPowReproOn(x, y);
}

// =============================================================================
// ENTRY POINT
// =============================================================================
int main(int argc, char **argv) {
  //srand(time(NULL));
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
