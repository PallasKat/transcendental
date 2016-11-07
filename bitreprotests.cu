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

//Double
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

//FLOATING POINTS.
template<class G, class R>
void testReproOnFloat(
  const std::vector<float> x,
  G gpuFunctor,
  R cpuFunctor
) {
  std::vector<float> cpuValues = applyCpuOp1Float(x, cpuFunctor);
  std::vector<float> gpuValues = applyGpuOp1Float(x, gpuFunctor);
  verifyEqFloat(cpuValues, gpuValues, x);
}

template<class G, class R>
void testReproOnFloat(
  const std::vector<float> x,
  const std::vector<float> y,
  G gpuFunctor,
  R cpuFunctor
) {
  std::vector<float> cpuValues = applyCpuOp2Float(x, y, cpuFunctor);
  std::vector<float> gpuValues = applyGpuOp2Float(x, y, gpuFunctor);
  verifyEqFloat(cpuValues, gpuValues, x);
}


// =============================================================================
// THE TESTS
// =============================================================================
const size_t N = 10*1000;

// =============================================================================
// DOUBLE -  TESTING THE LOGARITHM FUNCTIONS
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
/*
// =============================================================================
// FLOATING POINTS -  TESTING THE LOGARITHM FUNCTIONS
// =============================================================================

void testLogReproOnFloat(const std::vector<float> x) {
  testReproOnFloat(x, GpuLog(), CpuLog());
}

TEST(LogTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testLogReproOnFloat(x);
}

TEST(LogCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testLogReproOnFloat(x);
}

TEST(LogErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testLogReproOnFloat(x);
}

TEST(LogErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testLogReproOnFloat(x);
}

TEST(LogGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testLogReproOnFloat(x);
}

TEST(LogGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testLogReproOnFloat(x);
}

TEST(LogGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testLogReproOnFloat(x);
}

TEST(LogGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testLogReproOnFloat(x);
}
*/
// =============================================================================
// TESTING THE EXPONENTIAL FUNCTIONS WITH DOUBLE
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
/*
// =============================================================================
// TESTING THE EXPONENTIAL FUNCTIONS WITH FLOAT
// =============================================================================

void testExpReproOnFloat(const std::vector<float> x) {
  testReproOnFloat(x, GpuExp(), CpuExp());
}

TEST(ExpCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testExpReproOnFloat(x);
}

TEST(ExpCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testExpReproOnFloat(x);
}

TEST(ExpCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testExpReproOnFloat(x);
}

TEST(ExpCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testExpReproOnFloat(x);
}

TEST(ExpGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testExpReproOnFloat(x);
}

TEST(ExpGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testExpReproOnFloat(x);
}

TEST(ExpGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testExpReproOnFloat(x);
}

TEST(ExpGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testExpReproOnFloat(x);
}
*/
// =============================================================================
// DOUBLE -  TESTING THE EXPONENTIATION (z = x^y) FUNCTIONS
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
/*
// =============================================================================
// FLOATING POINTS - TESTING THE EXPONENTIATION (z = x^y) FUNCTIONS
// =============================================================================

void testPowReproOnFloat(
  const std::vector<float> x,
  const std::vector<float> y
) {
  testReproOnFloat(x, y, GpuPow(), CpuPow());
}

TEST(PowCPUTestFloat, PositiveBasePositiveExponentValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = randomFillFloat(0, 100, N);
  testPowReproOnFloat(x, y);
}

TEST(PowCPUTestFloat, PositiveBaseNegativeExponentValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = randomFillFloat(-100, 0, N);
  testPowReproOnFloat(x, y);
}

TEST(PowCPUTestFloat, NegativeBasePositiveEvenExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  std::vector<float> y = randomEvenFillFloat(0, 100, N);
  testPowReproOnFloat(x, y);
}

TEST(PowCPUTestFloat, NegativeBaseNegativeEvenExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  std::vector<float> y = randomEvenFillFloat(-100, 0, N);
  testPowReproOnFloat(x, y);
}

TEST(PowCPUTestFloat, NegativeBaseAnyExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  std::vector<float> y = randomFillFloat(-100, 100, N);
  testPowReproOnFloat(x, y);
}

TEST(PowCPUTestFloat, zeroExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 100, N);
  std::vector<float> y = zeroFillFloat(N);
  testPowReproOnFloat(x, y);
}

TEST(PowCPUTestFloat, zeroBaseValues) {
  std::vector<float> x = zeroFillFloat(N);
  std::vector<float> y = randomFillFloat(-100, 100, N);
  testPowReproOnFloat(x, y);
}

TEST(PowCPUTestFloat, zeroBaseZeroInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = zeroFillFloat(3);
  std::vector<float> y = {0.0, posInf, negInf};
  testPowReproOnFloat(x, y);
}

TEST(PowCPUTestFloat, posInfBaseZeroInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = valueFillFloat(posInf, 3);
  std::vector<float> y = {0.0, posInf, negInf};
  testPowReproOnFloat(x, y);
}

TEST(PowCPUTestFloat, negInfBaseZeroInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = valueFillFloat(negInf, 3);
  std::vector<float> y = {0.0, posInf, negInf};
  testPowReproOnFloat(x, y);
}

TEST(PowCPUTestFloat, zeroToOneBasePosInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  std::vector<float> x = randomFillFloat(N);
  std::vector<float> y = valueFillFloat(posInf, N);
  testPowReproOnFloat(x, y);
}

TEST(PowCPUTestFloat, zeroToOneBaseNegInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = randomFillFloat(N);
  std::vector<float> y = valueFillFloat(negInf, N);
  testPowReproOnFloat(x, y);
}

TEST(PowCPUTestFloat, anyBaseAnyIntExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 100, N);
  std::vector<float> y = randomIntFillFloat(-100, 100, N);
  testPowReproOnFloat(x, y);
}

TEST(PowGPUTestFloat, PositiveBasePositiveExponentValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = randomFillFloat(0, 100, N);
  testPowReproOnFloat(x, y);
}

TEST(PowGPUTestFloat, PositiveBaseNegativeExponentValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = randomFillFloat(-100, 0, N);
  testPowReproOnFloat(x, y);
}

TEST(PowGPUTestFloat, NegativeBasePositiveEvenExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  std::vector<float> y = randomEvenFillFloat(0, 100, N);
  testPowReproOnFloat(x, y);
}

TEST(PowGPUTestFloat, NegativeBaseNegativeEvenExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  std::vector<float> y = randomEvenFillFloat(-100, 0, N);
  testPowReproOnFloat(x, y);
}

TEST(PowGPUTestFloat, NegativeBaseAnyExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  std::vector<float> y = randomFillFloat(-100, 100, N);
  testPowReproOnFloat(x, y);
}

TEST(PowGPUTestFloat, zeroExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 100, N);
  std::vector<float> y = zeroFillFloat(N);
  testPowReproOnFloat(x, y);
}

TEST(PowGPUTestFloat, zeroBaseValues) {
  std::vector<float> x = zeroFillFloat(N);
  std::vector<float> y = randomFillFloat(-100, 100, N);
  testPowReproOnFloat(x, y);
}

TEST(PowGPUTestFloat, zeroBaseZeroInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = zeroFillFloat(3);
  std::vector<float> y = {0.0, posInf, negInf};
  testPowReproOnFloat(x, y);
}

TEST(PowGPUTestFloat, posInfBaseZeroInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = valueFillFloat(posInf, 3);
  std::vector<float> y = {0.0, posInf, negInf};
  testPowReproOnFloat(x, y);
}

TEST(PowGPUTestFloat, negInfBaseZeroInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = valueFillFloat(negInf, 3);
  std::vector<float> y = {0.0, posInf, negInf};
  testPowReproOnFloat(x, y);
}

TEST(PowGPUTestFloat, zeroToOneBasePosInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  std::vector<float> x = randomFillFloat(N);
  std::vector<float> y = valueFillFloat(posInf, N);
  testPowReproOnFloat(x, y);
}

TEST(PowGPUTestFloat, zeroToOneBaseNegInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = randomFillFloat(N);
  std::vector<float> y = valueFillFloat(negInf, N);
  testPowReproOnFloat(x, y);
}

TEST(PowGPUTestFloat, anyBaseAnyIntExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 100, N);
  std::vector<float> y = randomIntFillFloat(-100, 100, N);
  testPowReproOnFloat(x, y);
}
*/
// =============================================================================
// TESTING THE SIN FUNCTIONS WITH DOUBLES
// =============================================================================

void testSinReproOn(const std::vector<double> x) {
  testReproOn(x, GpuSin(), CpuSin());
}

TEST(SinCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testSinReproOn(x);
}

TEST(SinCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testSinReproOn(x);
}

TEST(SinErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testSinReproOn(x);
}

TEST(SinErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testSinReproOn(x);
}

TEST(SinGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testSinReproOn(x);
}

TEST(SinGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testSinReproOn(x);
}

TEST(SinGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testSinReproOn(x);
}

TEST(SinGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testSinReproOn(x);
}
/*
// =============================================================================
// TESTING THE SIN FUNCTIONS WITH FLOATING POINTS
// =============================================================================

void testSinReproOnFloat(const std::vector<float> x) {
  testReproOnFloat(x, GpuSin(), CpuSin());
}

TEST(SinCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testSinReproOnFloat(x);
}

TEST(SinCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testSinReproOnFloat(x);
}

TEST(SinErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testSinReproOnFloat(x);
}

TEST(SinErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testSinReproOnFloat(x);
}

TEST(SinGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testSinReproOnFloat(x);
}

TEST(SinGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testSinReproOnFloat(x);
}

TEST(SinGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testSinReproOnFloat(x);
}

TEST(SinGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testSinReproOnFloat(x);
}
*/
// =============================================================================
// TESTING THE COS FUNCTIONS WITH DOUBLES
// =============================================================================

void testCosReproOn(const std::vector<double> x) {
  testReproOn(x, GpuCos(), CpuCos());
}

TEST(CosCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testCosReproOn(x);
}

TEST(CosCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testCosReproOn(x);
}

TEST(CosErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testCosReproOn(x);
}

TEST(CosErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testCosReproOn(x);
}

TEST(CosGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testCosReproOn(x);
}

TEST(CosGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testCosReproOn(x);
}

TEST(CosGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testCosReproOn(x);
}

TEST(CosGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testCosReproOn(x);
}
/*
// =============================================================================
// TESTING THE COS FUNCTIONS WITH FLOATING POINTS
// =============================================================================

void testCosReproOnFloat(const std::vector<float> x) {
  testReproOnFloat(x, GpuCos(), CpuCos());
}

TEST(CosCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testCosReproOnFloat(x);
}

TEST(CosCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testCosReproOnFloat(x);
}

TEST(CosErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testCosReproOnFloat(x);
}

TEST(CosErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testCosReproOnFloat(x);
}

TEST(CosGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testCosReproOnFloat(x);
}

TEST(CosGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testCosReproOnFloat(x);
}

TEST(CosGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testCosReproOnFloat(x);
}

TEST(CosGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testCosReproOnFloat(x);
}
*/

// =============================================================================
// TESTING THE TAN FUNCTIONS WITH DOUBLES
// =============================================================================
void testTanReproOn(const std::vector<double> x) {
  testReproOn(x, GpuTan(), CpuTan());
}

TEST(TanCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testTanReproOn(x);
}

TEST(TanCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testTanReproOn(x);
}

TEST(TanErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testTanReproOn(x);
}

TEST(TanErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testTanReproOn(x);
}

TEST(TanGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testTanReproOn(x);
}

TEST(TanGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testTanReproOn(x);
}

TEST(TanGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testTanReproOn(x);
}

TEST(TanGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testTanReproOn(x);
}
/*
// =============================================================================
// TESTING THE TAN FUNCTIONS WITH FLOATING POINTS
// =============================================================================
void testTanReproOnFloat(const std::vector<float> x) {
  testReproOnFloat(x, GpuTan(), CpuTan());
}

TEST(TanCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testTanReproOnFloat(x);
}

TEST(TanCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testTanReproOnFloat(x);
}

TEST(TanErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testTanReproOnFloat(x);
}

TEST(TanErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testTanReproOnFloat(x);
}

TEST(TanGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testTanReproOnFloat(x);
}

TEST(TanGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testTanReproOnFloat(x);
}

TEST(TanGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testTanReproOnFloat(x);
}

TEST(TanGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testTanReproOnFloat(x);
}
*/
// =============================================================================
// TESTING THE ASIN FUNCTIONS WITH DOUBLES
// =============================================================================
void testAsinReproOn(const std::vector<double> x) {
  testReproOn(x, GpuAsin(), CpuAsin());
}

TEST(AsinCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testAsinReproOn(x);
}

TEST(AsinCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testAsinReproOn(x);
}

TEST(AsinErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testAsinReproOn(x);
}

TEST(AsinErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testAsinReproOn(x);
}

TEST(AsinGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testAsinReproOn(x);
}

TEST(AsinGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testAsinReproOn(x);
}

TEST(AsinGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testAsinReproOn(x);
}

TEST(AsinGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testAsinReproOn(x);
}
/*
// =============================================================================
// TESTING THE ASIN FUNCTIONS WITH FLOATING POINTS
// =============================================================================
void testAsinReproOnFloat(const std::vector<float> x) {
  testReproOnFloat(x, GpuAsin(), CpuAsin());
}

TEST(AsinCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testAsinReproOnFloat(x);
}

TEST(AsinCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testAsinReproOnFloat(x);
}

TEST(AsinErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testAsinReproOnFloat(x);
}

TEST(AsinErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testAsinReproOnFloat(x);
}

TEST(AsinGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testAsinReproOnFloat(x);
}

TEST(AsinGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testAsinReproOnFloat(x);
}

TEST(AsinGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testAsinReproOnFloat(x);
}

TEST(AsinGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testAsinReproOnFloat(x);
}
*/
// =============================================================================
// TESTING THE ACOS FUNCTIONS WITH DOUBLES
// =============================================================================
void testAcosReproOn(const std::vector<double> x) {
  testReproOn(x, GpuAcos(), CpuAcos());
}

TEST(AcosCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testAcosReproOn(x);
}

TEST(AcosCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testAcosReproOn(x);
}

TEST(AcosErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testAcosReproOn(x);
}

TEST(AcosErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testAcosReproOn(x);
}

TEST(AcosGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testAcosReproOn(x);
}

TEST(AcosGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testAcosReproOn(x);
}

TEST(AcosGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testAcosReproOn(x);
}

TEST(AcosGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testAcosReproOn(x);
}
/*
// =============================================================================
// TESTING THE ACOS FUNCTIONS WITH FLOATING POINTS
// =============================================================================
void testAcosReproOnFloat(const std::vector<float> x) {
  testReproOnFloat(x, GpuAcos(), CpuAcos());
}

TEST(AcosCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testAcosReproOnFloat(x);
}

TEST(AcosCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testAcosReproOnFloat(x);
}

TEST(AcosErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testAcosReproOnFloat(x);
}

TEST(AcosErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testAcosReproOnFloat(x);
}

TEST(AcosGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testAcosReproOnFloat(x);
}

TEST(AcosGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testAcosReproOnFloat(x);
}

TEST(AcosGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testAcosReproOnFloat(x);
}

TEST(AcosGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testAcosReproOnFloat(x);
}
*/
// =============================================================================
// TESTING THE ATAN FUNCTIONS WITH DOUBLES
// =============================================================================
void testAtanReproOn(const std::vector<double> x) {
  testReproOn(x, GpuAtan(), CpuAtan());
}

TEST(AtanCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testAtanReproOn(x);
}

TEST(AtanCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testAtanReproOn(x);
}

TEST(AtanErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testAtanReproOn(x);
}

TEST(AtanErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testAtanReproOn(x);
}

TEST(AtanGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testAtanReproOn(x);
}

TEST(AtanGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testAtanReproOn(x);
}

TEST(AtanGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testAtanReproOn(x);
}

TEST(AtanGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testAtanReproOn(x);
}
/*
// =============================================================================
// TESTING THE ATAN FUNCTIONS WITH FLOATING POINTS
// =============================================================================
void testAtanReproOnFloat(const std::vector<float> x) {
  testReproOnFloat(x, GpuAtan(), CpuAtan());
}

TEST(AtanCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testAtanReproOnFloat(x);
}

TEST(AtanCPUTestfFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testAtanReproOnFloat(x);
}

TEST(AtanErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testAtanReproOnFloat(x);
}

TEST(AtanErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testAtanReproOnFloat(x);
}

TEST(AtanGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testAtanReproOnFloat(x);
}

TEST(AtanGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testAtanReproOnFloat(x);
}

TEST(AtanGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testAtanReproOnFloat(x);
}

TEST(AtanGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testAtanReproOnFloat(x);
}
*/
// =============================================================================
// TESTING THE SINH FUNCTIONS WITH DOUBLES
// =============================================================================
void testSinhReproOn(const std::vector<double> x) {
  testReproOn(x, GpuSinh(), CpuSinh());
}

TEST(SinhCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testSinhReproOn(x);
}

TEST(SinhCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testSinhReproOn(x);
}

TEST(SinhErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testSinhReproOn(x);
}

TEST(SinhErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testSinhReproOn(x);
}

TEST(SinhGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testSinhReproOn(x);
}

TEST(SinhGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testSinhReproOn(x);
}

TEST(SinhGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testSinhReproOn(x);
}

TEST(SinhGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testSinhReproOn(x);
}
/*
// =============================================================================
// TESTING THE SINH FUNCTIONS WITH FLOATING POINTS
// =============================================================================
void testSinhReproOnFloat(const std::vector<float> x) {
  testReproOnFloat(x, GpuSinh(), CpuSinh());
}

TEST(SinhCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testSinhReproOnFloat(x);
}

TEST(SinhCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testSinhReproOnFloat(x);
}

TEST(SinhErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testSinhReproOnFloat(x);
}

TEST(SinhErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testSinhReproOnFloat(x);
}

TEST(SinhGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testSinhReproOnFloat(x);
}

TEST(SinhGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testSinhReproOnFloat(x);
}

TEST(SinhGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testSinhReproOnFloat(x);
}

TEST(SinhGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testSinhReproOnFloat(x);
}
*/
// =============================================================================
// TESTING THE COSH FUNCTIONS WITH DOUBLES
// =============================================================================
void testCoshReproOn(const std::vector<double> x) {
  testReproOn(x, GpuCosh(), CpuCosh());
}

TEST(CoshCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testCoshReproOn(x);
}

TEST(CoshCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testCoshReproOn(x);
}

TEST(CoshErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testCoshReproOn(x);
}

TEST(CoshErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testCoshReproOn(x);
}

TEST(CoshGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testCoshReproOn(x);
}

TEST(CoshGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testCoshReproOn(x);
}

TEST(CoshGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testAtanReproOn(x);
}

TEST(CoshGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testCoshReproOn(x);
}
/*
// =============================================================================
// TESTING THE COSH FUNCTIONS WITH FLOATING POINTS
// =============================================================================
void testCoshReproOnFloat(const std::vector<float> x) {
  testReproOnFloat(x, GpuCosh(), CpuCosh());
}

TEST(CoshCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testCoshReproOnFloat(x);
}

TEST(CoshCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testCoshReproOnFloat(x);
}

TEST(CoshErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testCoshReproOnFloat(x);
}

TEST(CoshErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testCoshReproOnFloat(x);
}

TEST(CoshGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testCoshReproOnFloat(x);
}

TEST(CoshGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testCoshReproOnFloat(x);
}

TEST(CoshGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testAtanReproOnFloat(x);
}

TEST(CoshGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testCoshReproOnFloat(x);
}
*/
// =============================================================================
// TESTING THE TANH FUNCTIONS WITH DOUBLES
// =============================================================================
void testTanhReproOn(const std::vector<double> x) {
  testReproOn(x, GpuTanh(), CpuTanh());
}

TEST(TanhCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testTanhReproOn(x);
}

TEST(TanhCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testTanhReproOn(x);
}

TEST(TanhErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testTanhReproOn(x);
}

TEST(TanhErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testTanhReproOn(x);
}

TEST(TanhGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testTanhReproOn(x);
}

TEST(TanhGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testTanhReproOn(x);
}

TEST(TanhGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testTanhReproOn(x);
}

TEST(TanhGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testTanhReproOn(x);
}
/*
// =============================================================================
// TESTING THE TANH FUNCTIONS WITH FLOATING POINTS
// =============================================================================
void testTanhReproOnFloat(const std::vector<float> x) {
  testReproOnFloat(x, GpuTanh(), CpuTanh());
}

TEST(TanhCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testTanhReproOnFloat(x);
}

TEST(TanhCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testTanhReproOnFloat(x);
}

TEST(TanhErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testTanhReproOnFloat(x);
}

TEST(TanhErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testTanhReproOnFloat(x);
}

TEST(TanhGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testTanhReproOnFloat(x);
}

TEST(TanhGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testTanhReproOnFloat(x);
}

TEST(TanhGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testTanhReproOnFloat(x);
}

TEST(TanhGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<double>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testTanhReproOnFloat(x);
}
*/
// =============================================================================
// TESTING THE ASINH FUNCTIONS WITH DOUBLES
// =============================================================================
void testAsinhReproOn(const std::vector<double> x) {
  testReproOn(x, GpuAsinh(), CpuAsinh());
}

TEST(AsinhCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testAsinhReproOn(x);
}

TEST(AsinhCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testAsinhReproOn(x);
}

TEST(AsinhErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testAsinhReproOn(x);
}

TEST(AsinhErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testAsinhReproOn(x);
}

TEST(AsinhGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testAsinhReproOn(x);
}

TEST(AsinhGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testAsinhReproOn(x);
}

TEST(AsinhGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testAsinhReproOn(x);
}

TEST(AsinhGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testAsinhReproOn(x);
}
/*
// =============================================================================
// TESTING THE ASINH FUNCTIONS WITH FLOATING POINTS
// =============================================================================
void testAsinhReproOnFloat(const std::vector<float> x) {
  testReproOnFloat(x, GpuAsinh(), CpuAsinh());
}

TEST(AsinhCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testAsinhReproOnFloat(x);
}

TEST(AsinhCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testAsinhReproOnFloat(x);
}

TEST(AsinhErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testAsinhReproOnFloat(x);
}

TEST(AsinhErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testAsinhReproOnFloat(x);
}

TEST(AsinhGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testAsinhReproOnFloat(x);
}

TEST(AsinhGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testAsinhReproOnFloat(x);
}

TEST(AsinhGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testAsinhReproOnFloat(x);
}

TEST(AsinhGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testAsinhReproOnFloat(x);
}
*/
// =============================================================================
// TESTING THE ACOSH FUNCTIONS WITH DOUBLES
// =============================================================================
void testAcoshReproOn(const std::vector<double> x) {
  testReproOn(x, GpuAcosh(), CpuAcosh());
}

TEST(AcoshCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testAcoshReproOn(x);
}

TEST(AcoshCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testAcoshReproOn(x);
}

TEST(AcoshErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testAcoshReproOn(x);
}

TEST(AcoshErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testAcoshReproOn(x);
}

TEST(AcoshGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testAcoshReproOn(x);
}

TEST(AcoshGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testAcoshReproOn(x);
}

TEST(AcoshGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testAcoshReproOn(x);
}

TEST(AcoshGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testAcoshReproOn(x);
}
/*
// =============================================================================
// TESTING THE ACOSH FUNCTIONS WITH FLOATING POINTS
// =============================================================================
void testAcoshReproOnFloat(const std::vector<float> x) {
  testReproOnFloat(x, GpuAcosh(), CpuAcosh());
}

TEST(AcoshCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testAcoshReproOnFloat(x);
}

TEST(AcoshCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testAcoshReproOnFloat(x);
}

TEST(AcoshErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testAcoshReproOnFloat(x);
}

TEST(AcoshErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testAcoshReproOnFloat(x);
}

TEST(AcoshGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testAcoshReproOnFloat(x);
}

TEST(AcoshGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testAcoshReproOnFloat(x);
}

TEST(AcoshGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testAcoshReproOnFloat(x);
}

TEST(AcoshGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testAcoshReproOnFloat(x);
}
*/
// =============================================================================
// TESTING THE ATANH FUNCTIONS WITH DOUBLES
// =============================================================================
void testAtanhReproOn(const std::vector<double> x) {
  testReproOn(x, GpuAtanh(), CpuAtanh());
}

TEST(SinCheck, ExactValue){
 std::vector<double> x={1};
 testAtanhReproOn(x);
}

TEST(AtanhCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testAtanhReproOn(x);
}

TEST(AtanhCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testAtanhReproOn(x);
}

TEST(AtanhErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testAtanhReproOn(x);
}

TEST(AtanhErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testAtanhReproOn(x);
}

TEST(AtanhGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testAtanhReproOn(x);
}

TEST(AtanhGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testAtanhReproOn(x);
}

TEST(AtanhGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testAtanhReproOn(x);
}

TEST(AtanhGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testAtanhReproOn(x);
}
/*
// =============================================================================
// TESTING THE ATANH FUNCTIONS WITH FLOATING POINTS
// =============================================================================
void testAtanhReproOnFloat(const std::vector<float> x) {
  testReproOnFloat(x, GpuAtanh(), CpuAtanh());
}

TEST(AtanhCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testAtanhReproOnFloat(x);
}

TEST(AtanhCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testAtanhReproOnFloat(x);
}

TEST(AtanhErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testAtanhReproOnFloat(x);
}

TEST(AtanhErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testAtanhReproOnFloat(x);
}

TEST(AtanhGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testAtanhReproOnFloat(x);
}

TEST(AtanhGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testAtanhReproOnFloat(x);
}

TEST(AtanhGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testAtanhReproOnFloat(x);
}

TEST(AtanhGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testAtanhReproOnFloat(x);
}
*/
// =============================================================================
// ENTRY POINT
// =============================================================================
int main(int argc, char **argv) {
  //srand(time(NULL));
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
