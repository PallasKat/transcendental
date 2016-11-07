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
#include <iostream>

// =============================================================================
// TEMPLATE WITH DOUBLE
// =============================================================================

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
// TEMPLATE WITH FLOAT
// =============================================================================

template<class G, class R>
void testCpuTolOnFloat(
  const std::vector<float> x,
  const float tol,
  G cpuFunctor,
  R refFunctor
) {
  std::vector<float> crClimValues = applyCpuOp1Float(x, cpuFunctor);
  std::vector<float> expectedValues = applyCpuOp1Float(x, refFunctor);
  verifyTolFloat(expectedValues, crClimValues, x, tol);
}

template<class G, class R>
void testGpuTolOnFloat(
  const std::vector<float> x,
  const float tol,
  G gpuFunctor,
  R refFunctor
) {
  std::vector<float> crClimValues = applyGpuOp1Float(x, gpuFunctor);
  std::vector<float> expectedValues = applyCpuOp1Float(x, refFunctor);
  verifyTolFloat(expectedValues, crClimValues, x, tol);
}

template<class G, class R>
void testCpuTolOnFloat(
  const std::vector<float> x,
  const std::vector<float> y,
  const float tol,
  G cpuFunctor,
  R refFunctor
) {
  std::vector<float> crClimValues = applyCpuOp2Float(x, y, cpuFunctor);
  std::vector<float> expectedValues = applyCpuOp2Float(x, y, refFunctor);
  verifyTolFloat(expectedValues, crClimValues, x, tol);
}

template<class G, class R>
void testGpuTolOnFloat(
  const std::vector<float> x,
  const std::vector<float> y,
  const float tol,
  G gpuFunctor,
  R refFunctor
) {
  std::vector<float> crClimValues = applyGpuOp2Float(x, y, gpuFunctor);
  std::vector<float> expectedValues = applyCpuOp2Float(x, y, refFunctor);
  verifyTolFloat(expectedValues, crClimValues, x, tol);
}


// =============================================================================
// THE TESTS
// =============================================================================
const size_t N = 1000*10;
const double TOL = 10E-10;
//const float TOLL = 10E-10;

// =============================================================================
// TESTING THE LOGARITHM FUNCTIONS WITH DOUBLES
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
/*
// =============================================================================
// TESTING THE LOGARITHM FUNCTIONS WITH FLOATING POINTS
// =============================================================================

void testCpuLogTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testCpuTolOnFloat(x, tol, CpuLog(), LibLog());
}

void testGpuLogTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testGpuTolOnFloat(x, tol, GpuLog(), LibLog());
}

void testCpuLogEqOnFloat(const std::vector<float> x) {
	testCpuLogTolOnFloat(x, 0.0);
}

void testGpuLogEqOnFloat(const std::vector<float> x) {
	testGpuLogTolOnFloat(x, 0.0);
}

TEST(LogCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testCpuLogTolOnFloat(x, TOLL);
}

TEST(LogCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testCpuLogTolOnFloat(x, TOLL);
}

TEST(LogErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testCpuLogEqOnFloat(x);
}

TEST(LogErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testCpuLogEqOnFloat(x);
}

TEST(LogGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testGpuLogTolOnFloat(x, TOLL);
}

TEST(LogGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testGpuLogTolOnFloat(x, TOLL);
}

TEST(LogGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testGpuLogEqOnFloat(x);
}

TEST(LogGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testGpuLogEqOnFloat(x);
}
*/

// =============================================================================
// TESTING THE EXPONENTIAL FUNCTIONS WITH DOUBLES
// =============================================================================

void testCpuExpTolOn(
  const std::vector<double> x,
  const double tol
) {
  testCpuTolOn(x, tol, CpuExp(), LibExp());
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
/*
// =============================================================================
// TESTING THE EXPONENTIAL FUNCTIONS WITH FLOATING POINTS 
// =============================================================================
void testCpuExpTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testCpuTolOnFloat(x, tol, CpuExp(), LibExp());
}

void testGpuExpTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testGpuTolOnFloat(x, tol, GpuExp(), LibExp());
}

void testCpuExpEqOnFloat(const std::vector<float> x) {
        testCpuExpTolOnFloat(x, 0.0);
}

void testGpuExpEqOnFloat(const std::vector<float> x) {
        testGpuExpTolOnFloat(x, 0.0);
}

TEST(ExpCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testCpuExpTolOnFloat(x, TOLL);
}

TEST(ExpCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testCpuExpTolOnFloat(x, TOLL);
}

TEST(ExpCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testCpuExpEqOnFloat(x);
}

TEST(ExpCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testCpuExpEqOnFloat(x);
}

TEST(ExpGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testGpuExpTolOnFloat(x, TOLL);
}

TEST(ExpGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testGpuExpTolOnFloat(x, TOLL);
}

TEST(ExpGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testGpuExpEqOnFloat(x);
}

TEST(ExpGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testGpuExpEqOnFloat(x);
}
*/
// =============================================================================
// TESTING THE EXPONENTIATION (z = x^y) FUNCTIONS WITH DOUBLES
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
/*
// =============================================================================
// TESTING THE EXPONENTIATION (z = x^y) FUNCTIONS WITH FLOATING POINTS
// =============================================================================

void testCpuPowTolOnFloat(
  const std::vector<float> x,
  const std::vector<float> y,
  const float tol
) {
  testCpuTolOnFloat(x, y, tol, CpuPow(), LibPow());
}

void testGpuPowTolOnFloat(
  const std::vector<float> x,
  const std::vector<float> y,
  const float tol
) {
  testGpuTolOnFloat(x, y, tol, GpuPow(), LibPow());
}

void testCpuPowEqOnFloat(
	const std::vector<float> x,
	const std::vector<float> y
) {
	testCpuPowTolOnFloat(x, y, 0.0);
}

void testGpuPowEqOnFloat(
	const std::vector<float> x,
	const std::vector<float> y
) {
	testGpuPowTolOnFloat(x, y, 0.0);
}

TEST(PowCPUTestFloat, PositiveBasePositiveExponentValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = randomFillFloat(0, 100, N);
  testCpuPowTolOnFloat(x, y, TOLL);
}

TEST(PowCPUTestFloat, PositiveBaseNegativeExponentValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = randomFillFloat(-100, 0, N);
  testCpuPowTolOnFloat(x, y, TOLL);
}

TEST(PowCPUTestFloat, NegativeBasePositiveEvenExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  std::vector<float> y = randomEvenFillFloat(0, 100, N);
  testCpuPowTolOnFloat(x, y, TOLL);
}

TEST(PowCPUTestFloat, NegativeBaseNegativeEvenExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  std::vector<float> y = randomEvenFillFloat(-100, 0, N);
  testCpuPowTolOnFloat(x, y, TOLL);
}

TEST(PowCPUTestFloat, NegativeBaseAnyExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  std::vector<float> y = randomFillFloat(-100, 100, N);
  testCpuPowTolOnFloat(x, y, TOLL);
}

TEST(PowCPUTestFloat, zeroExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 100, N);
  std::vector<float> y = zeroFillFloat(N);
  testCpuPowEqOnFloat(x, y);
}

TEST(PowCPUTestFloat, zeroBaseValues) {
  std::vector<float> x = zeroFillFloat(N);
  std::vector<float> y = randomFillFloat(-100, 100, N);
  testCpuPowEqOnFloat(x, y);
}

TEST(PowCPUTestFloat, zeroBaseZeroInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = zeroFillFloat(3);
  std::vector<float> y = {0.0, posInf, negInf};
  testCpuPowEqOnFloat(x, y);
}

TEST(PowCPUTestFloat, posInfBaseZeroInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = valueFillFloat(posInf, 3);
  std::vector<float> y = {0.0, posInf, negInf};
  testCpuPowEqOnFloat(x, y);
}

TEST(PowCPUTestFloat, negInfBaseZeroInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = valueFillFloat(negInf, 3);
  std::vector<float> y = {0.0, posInf, negInf};
  testCpuPowEqOnFloat(x, y);
}

TEST(PowCPUTestFloat, zeroToOneBasePosInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  std::vector<float> x = randomFillFloat(N);
  std::vector<float> y = valueFillFloat(posInf, N);
  testCpuPowEqOnFloat(x, y);
}

TEST(PowCPUTestFloat, zeroToOneBaseNegInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = randomFillFloat(N);
  std::vector<float> y = valueFillFloat(negInf, N);
  testCpuPowEqOnFloat(x, y);
}

TEST(PowCPUTestFloat, anyBaseAnyIntExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 100, N);
  std::vector<float> y = randomIntFillFloat(-100, 100, N);
  testCpuPowTolOnFloat(x, y, TOLL);
}

TEST(PowGPUTestFloat, PositiveBasePositiveExponentValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = randomFillFloat(0, 100, N);
  testGpuPowTolOnFloat(x, y, TOLL);
}

TEST(PowGPUTestFloat, PositiveBaseNegativeExponentValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = randomFillFloat(-100, 0, N);
  testGpuPowTolOnFloat(x, y, TOLL);
}

TEST(PowGPUTestFloat, NegativeBasePositiveEvenExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  std::vector<float> y = randomEvenFillFloat(0, 100, N);
  testGpuPowTolOnFloat(x, y, TOLL);
}

TEST(PowGPUTestFloat, NegativeBaseNegativeEvenExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  std::vector<float> y = randomEvenFillFloat(-100, 0, N);
  testGpuPowTolOnFloat(x, y, TOLL);
}

TEST(PowGPUTestFloat, NegativeBaseAnyExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  std::vector<float> y = randomFillFloat(-100, 100, N);
  testCpuPowTolOnFloat(x, y, TOLL);
}

TEST(PowGPUTestfloat, zeroExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 100, N);
  std::vector<float> y = zeroFillFloat(N);
  testGpuPowEqOnFloat(x, y);
}

TEST(PowGPUTestFloat, zeroBaseValues) {
  std::vector<float> x = zeroFillFloat(N);
  std::vector<float> y = randomFillFloat(-100, 100, N);
  testGpuPowEqOnFloat(x, y);
}

TEST(PowGPUTestFloat, zeroBaseZeroInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = zeroFillFloat(3);
  std::vector<float> y = {0.0, posInf, negInf};
  testGpuPowEqOnFloat(x, y);
}

TEST(PowGPUTestFloat, posInfBaseZeroInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = valueFillFloat(posInf, 3);
  std::vector<float> y = {0.0, posInf, negInf};
  testGpuPowEqOnFloat(x, y);
}

TEST(PowGPUTestFloat, negInfBaseZeroInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = valueFillFloat(negInf, 3);
  std::vector<float> y = {0.0, posInf, negInf};
  testGpuPowEqOnFloat(x, y);
}

TEST(PowGPUTestFloat, zeroToOneBasePosInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  std::vector<float> x = randomFillFloat(N);
  std::vector<float> y = valueFillFloat(posInf, N);
  testGpuPowEqOnFloat(x, y);
}

TEST(PowGPUTestFloat, zeroToOneBaseNegInfExponentValues) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = randomFillFloat(N);
  std::vector<float> y = valueFillFloat(negInf, N);
  testGpuPowEqOnFloat(x, y);
}

TEST(PowGPUTestFloat, anyBaseAnyIntExponentValues) {
  std::vector<float> x = randomFillFloat(-100, 100, N);
  std::vector<float> y = randomIntFillFloat(-100, 100, N);
  testGpuPowTolOnFloat(x, y, TOLL);
}
*/
// =============================================================================
// TESTING THE SINE FUNCTIONS WITH DOUBLES
// =============================================================================

void testCpuSinTolOn(
  const std::vector<double> x,
  const double tol
) {
  testCpuTolOn(x, tol, CpuSin(), LibSin());
}

void testGpuSinTolOn(
  const std::vector<double> x,
  const double tol
) {
  testGpuTolOn(x, tol, GpuSin(), LibSin());
}

void testCpuSinEqOn(const std::vector<double> x) {
	testCpuSinTolOn(x, 0.0);
}

void testGpuSinEqOn(const std::vector<double> x) {
	testGpuSinTolOn(x, 0.0);
}

TEST(SinCPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testCpuSinEqOn(x);
}

TEST(SinErrCPUTest, Positive_NAN) {
  std::vector<double> x = {NAN};
  testCpuSinEqOn(x);
}

TEST(SinErrCPUTest, Negative_NAN) {
  std::vector<double> x = {-NAN};
  testCpuSinEqOn(x);
}


TEST(SinCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testCpuSinTolOn(x, TOL);
}

TEST(SinCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testCpuSinTolOn(x, TOL);
}

TEST(SinErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testCpuSinEqOn(x);
}

TEST(SinErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testCpuSinEqOn(x);
}

TEST(SinGPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testGpuSinEqOn(x);
}

TEST(SinGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testGpuSinTolOn(x, TOL);
}

TEST(SinGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testGpuSinTolOn(x, TOL);
}

TEST(SinGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testGpuSinEqOn(x);
}

TEST(SinGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testGpuSinEqOn(x);
}
/*
// =============================================================================
// TESTING THE SINE FUNCTIONS WITH FLOATING POINTS
// =============================================================================
void testCpuSinTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testCpuTolOnFloat(x, tol, CpuSin(), LibSin());
}

void testGpuSinTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testGpuTolOnFloat(x, tol, GpuSin(), LibSin());
}

void testCpuSinEqOnFloat(const std::vector<float> x) {
        testCpuSinTolOnFloat(x, 0.0);
}

void testGpuSinEqOnFloat(const std::vector<float> x) {
        testGpuSinTolOnFloat(x, 0.0);
}

TEST(SinCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testCpuSinTolOnFloat(x, TOL);
}

TEST(SinCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testCpuSinTolOnFloat(x, TOL);
}

TEST(SinErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testCpuSinEqOnFloat(x);
}

TEST(SinErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testCpuSinEqOnFloat(x);
}

TEST(SinGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testGpuSinTolOnFloat(x, TOL);
}

TEST(SinGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testGpuSinTolOnFloat(x, TOL);
}

TEST(SinGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testGpuSinEqOnFloat(x);
}

TEST(SinGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testGpuSinEqOnFloat(x);
}
*/
// =============================================================================
// TESTING THE COSINE FUNCTIONS WITH DOUBLE
// =============================================================================

void testCpuCosTolOn(
  const std::vector<double> x,
  const double tol
) {
  testCpuTolOn(x, tol, CpuCos(), LibCos());
}

void testGpuCosTolOn(
  const std::vector<double> x,
  const double tol
) {
  testGpuTolOn(x, tol, GpuCos(), LibCos());
}

void testCpuCosEqOn(const std::vector<double> x) {
	testCpuCosTolOn(x, 0.0);
}

void testGpuCosEqOn(const std::vector<double> x) {
	testGpuCosTolOn(x, 0.0);
}

TEST(CosCPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testCpuCosEqOn(x);
}


TEST(CosCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testCpuCosTolOn(x, TOL);
}

TEST(CosCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testCpuCosTolOn(x, TOL);
}

TEST(CosErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testCpuCosEqOn(x);
}

TEST(CosErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testCpuCosEqOn(x);
}

TEST(CosGPUTest, nan) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testGpuCosEqOn(x);
}

TEST(CosGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testGpuCosTolOn(x, TOL);
}

TEST(CosGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testGpuCosTolOn(x, TOL);
}

TEST(CosGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testGpuCosEqOn(x);
}

TEST(CosGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testGpuCosEqOn(x);
}
/*
// =============================================================================
// TESTING THE COSINE FUNCTIONS WITH FLOATING POINTS
// =============================================================================

void testCpuCosTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testCpuTolOnFloat(x, tol, CpuCos(), LibCos());
}

void testGpuCosTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testGpuTolOnFloat(x, tol, GpuCos(), LibCos());
}

void testCpuCosEqOnFloat(const std::vector<float> x) {
        testCpuCosTolOnFloat(x, 0.0);
}

void testGpuCosEqOnFloat(const std::vector<float> x) {
        testGpuCosTolOnFloat(x, 0.0);
}

TEST(CosCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testCpuCosTolOnFloat(x, TOL);
}

TEST(CosCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testCpuCosTolOnFloat(x, TOL);
}

TEST(CosErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testCpuCosEqOnFloat(x);
}

TEST(CosErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testCpuCosEqOnFloat(x);
}

TEST(CosGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testGpuCosTolOnFloat(x, TOL);
}

TEST(CosGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testGpuCosTolOnFloat(x, TOL);
}

TEST(CosGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testGpuCosEqOnFloat(x);
}

TEST(CosGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testGpuCosEqOnFloat(x);
}
*/
// =============================================================================
// TESTING THE TANGENT FUNCTIONS WITH DOUBLES
// =============================================================================
void testCpuTanTolOn(
  const std::vector<double> x,
  const double tol
) {
  testCpuTolOn(x, tol, CpuTan(), LibTan());
}

void testGpuTanTolOn(
  const std::vector<double> x,
  const double tol
) {
  testGpuTolOn(x, tol, GpuTan(), LibTan());
}

void testCpuTanEqOn(const std::vector<double> x) {
	testCpuTanTolOn(x, 0.0);
}

void testGpuTanEqOn(const std::vector<double> x) {
	testGpuTanTolOn(x, 0.0);
}

TEST(TanCPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testCpuTanEqOn(x);
}

TEST(TanCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testCpuTanTolOn(x, TOL);
}

TEST(TanCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testCpuTanTolOn(x, TOL);
}

TEST(TanErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testCpuTanEqOn(x);
}

TEST(TanErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testCpuTanEqOn(x);
}

TEST(TanGPUTest, NANs) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testGpuTanEqOn(x);
}

TEST(TanGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testGpuTanTolOn(x, TOL);
}

TEST(TanGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testGpuTanTolOn(x, TOL);
}

TEST(TanGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testGpuTanEqOn(x);
}

TEST(TanGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testGpuTanEqOn(x);
}
/*
// =============================================================================
// TESTING THE TANGENT FUNCTIONS WITH FLOATING POINTS
// =============================================================================
void testCpuTanTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testCpuTolOnFloat(x, tol, CpuTan(), LibTan());
}

void testGpuTanTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testGpuTolOnFloat(x, tol, GpuTan(), LibTan());
}

void testCpuTanEqOnFloat(const std::vector<float> x) {
        testCpuTanTolOnFloat(x, 0.0);
}

void testGpuTanEqOnFloat(const std::vector<float> x) {
        testGpuTanTolOnFloat(x, 0.0);
}

TEST(TanCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testCpuTanTolOnFloat(x, TOL);
}

TEST(TanCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testCpuTanTolOnFloat(x, TOL);
}

TEST(TanErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testCpuTanEqOnFloat(x);
}

TEST(TanErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testCpuTanEqOnFloat(x);
}

TEST(TanGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testGpuTanTolOnFloat(x, TOL);
}

TEST(TanGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testGpuTanTolOnFloat(x, TOL);
}

TEST(TanGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testGpuTanEqOnFloat(x);
}

TEST(TanGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testGpuTanEqOnFloat(x);
}
*/
// =============================================================================
// TESTING THE INVERSE SINE FUNCTIONS WITH DOUBLES
// =============================================================================
void testCpuAsinTolOn(
  const std::vector<double> x,
  const double tol
) {
  testCpuTolOn(x, tol, CpuAsin(), LibAsin());
}

void testGpuAsinTolOn(
  const std::vector<double> x,
  const double tol
) {
  testGpuTolOn(x, tol, GpuAsin(), LibAsin());
}

void testCpuAsinEqOn(const std::vector<double> x) {
	testCpuAsinTolOn(x, 0.0);
}

void testGpuAsinEqOn(const std::vector<double> x) {
	testGpuAsinTolOn(x, 0.0);
}

TEST(AsinCPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testCpuAsinEqOn(x);
}

TEST(AsinCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testCpuAsinTolOn(x, TOL);
}

TEST(AsinCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testCpuAsinTolOn(x, TOL);
}

TEST(AsinErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testCpuAsinEqOn(x);
}

TEST(AsinErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testCpuAsinEqOn(x);
}

TEST(AsinGPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testGpuAsinEqOn(x);
}

TEST(AsinGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testGpuAsinTolOn(x, TOL);
}

TEST(AsinGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testGpuAsinTolOn(x, TOL);
}

TEST(AsinGPUTest, Zero) {
  std::vector<double> x = {0};
  testGpuAsinEqOn(x);
}

TEST(AsinGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testGpuAsinEqOn(x);
}

/*
// =============================================================================
// TESTING THE INVERSE SINE FUNCTIONS WITH FlOATING POINTS
// =============================================================================
void testCpuAsinTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testCpuTolOnFloat(x, tol, CpuAsin(), LibAsin());
}

void testGpuAsinTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testGpuTolOnFloat(x, tol, GpuAsin(), LibAsin());
}

void testCpuAsinEqOnFloat(const std::vector<float> x) {
        testCpuAsinTolOnFloat(x, 0.0);
}

void testGpuAsinEqOnFloat(const std::vector<float> x) {
        testGpuAsinTolOnFloat(x, 0.0);
}

TEST(AsinCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testCpuAsinTolOnFloat(x, TOL);
}

TEST(AsinCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testCpuAsinTolOnFloat(x, TOL);
}

TEST(AsinErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testCpuAsinEqOnFloat(x);
}

TEST(AsinErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testCpuAsinEqOnFloat(x);
}

TEST(AsinGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testGpuAsinTolOnFloat(x, TOL);
}


TEST(AsinGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testGpuAsinTolOnFloat(x, TOL);
}

TEST(AsinGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testGpuAsinEqOnFloat(x);
}

TEST(AsinGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testGpuAsinEqOnFloat(x);
}
*/
// =============================================================================
// TESTING THE INVERSE COSINE FUNCTIONS WITH DOUBLES
// =============================================================================

void testCpuAcosTolOn(
  const std::vector<double> x,
  const double tol
) {
  testCpuTolOn(x, tol, CpuAcos(), LibAcos());
}

void testGpuAcosTolOn(
  const std::vector<double> x,
  const double tol
) {
  testGpuTolOn(x, tol, GpuAcos(), LibAcos());
}

void testCpuAcosEqOn(const std::vector<double> x) {
	testCpuAcosTolOn(x, 0.0);
}

void testGpuAcosEqOn(const std::vector<double> x) {
	testGpuAcosTolOn(x, 0.0);
}

TEST(AcosCPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testCpuAcosEqOn(x);
}

TEST(AcosErrCPUTest, negativeONE) {
  std::vector<double> x = {-1};
  testCpuAcosEqOn(x);
}

TEST(AcosCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testCpuAcosTolOn(x, TOL);
}

TEST(AcosCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testCpuAcosTolOn(x, TOL);
}

TEST(AcosErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testCpuAcosEqOn(x);
}

TEST(AcosErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testCpuAcosEqOn(x);
}

TEST(AcosGPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testGpuAcosEqOn(x);
}


TEST(AcosGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testGpuAcosTolOn(x, TOL);
}

TEST(AcosGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testGpuAcosTolOn(x, TOL);
}

TEST(AcosGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testGpuAcosEqOn(x);
}

TEST(AcosGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testGpuAcosEqOn(x);
}
/*
// =============================================================================
// TESTING THE INVERSE COSINE FUNCTIONS WITH FLOATING POINTS
// =============================================================================

void testCpuAcosTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testCpuTolOnFloat(x, tol, CpuAcos(), LibAcos());
}

void testGpuAcosTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testGpuTolOnFloat(x, tol, GpuAcos(), LibAcos());
}

void testCpuAcosEqOnFloat(const std::vector<float> x) {
        testCpuAcosTolOnFloat(x, 0.0);
}

void testGpuAcosEqOnFloat(const std::vector<float> x) {
        testGpuAcosTolOnFloat(x, 0.0);
}

TEST(AcosCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testCpuAcosTolOnFloat(x, TOL);
}

TEST(AcosCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testCpuAcosTolOnFloat(x, TOL);
}

TEST(AcosErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testCpuAcosEqOnFloat(x);
}

TEST(AcosErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testCpuAcosEqOnFloat(x);
}

TEST(AcosGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testGpuAcosTolOnFloat(x, TOL);
}

TEST(AcosGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testGpuAcosTolOnFloat(x, TOL);
}

TEST(AcosGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testGpuAcosEqOnFloat(x);
}

TEST(AcosGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testGpuAcosEqOnFloat(x);
}
*/
// =============================================================================
// TESTING THE INVERSE TANGENT FUNCTIONS WITH DOUBLES
// =============================================================================

void testCpuAtanTolOn(
  const std::vector<double> x,
  const double tol
) {
  testCpuTolOn(x, tol, CpuAtan(), LibAtan());
}

void testGpuAtanTolOn(
  const std::vector<double> x,
  const double tol
) {
  testGpuTolOn(x, tol, GpuAtan(), LibAtan());
}

void testCpuAtanEqOn(const std::vector<double> x) {
	testCpuAtanTolOn(x, 0.0);
}

void testGpuAtanEqOn(const std::vector<double> x) {
	testGpuAtanTolOn(x, 0.0);
}

TEST(AtanCPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testCpuAtanEqOn(x);
}

TEST(AtanCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testCpuAtanTolOn(x, TOL);
}

TEST(AtanCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testCpuAtanTolOn(x, TOL);
}

TEST(AtanErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testCpuAtanEqOn(x);
}

TEST(AtanErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testCpuAtanEqOn(x);
}

TEST(AtanGPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testGpuAtanEqOn(x);
}

TEST(AtanGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testGpuAtanTolOn(x, TOL);
}

TEST(AtanGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testGpuAtanTolOn(x, TOL);
}

TEST(AtanGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testGpuAtanEqOn(x);
}

TEST(AtanGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testGpuAtanEqOn(x);
}
/*
// =============================================================================
// TESTING THE INVERSE TANGENT FUNCTIONS WITH FLOATING POINTS
// =============================================================================

void testCpuAtanTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testCpuTolOnFloat(x, tol, CpuAtan(), LibAtan());
}

void testGpuAtanTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testGpuTolOnFloat(x, tol, GpuAtan(), LibAtan());
}

void testCpuAtanEqOnFloat(const std::vector<float> x) {
        testCpuAtanTolOnFloat(x, 0.0);
}

void testGpuAtanEqOnFloat(const std::vector<float> x) {
        testGpuAtanTolOnFloat(x, 0.0);
}

TEST(AtanCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testCpuAtanTolOnFloat(x, TOL);
}

TEST(AtanCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testCpuAtanTolOnFloat(x, TOL);
}

TEST(AtanErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testCpuAtanEqOnFloat(x);
}

TEST(AtanErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testCpuAtanEqOnFloat(x);
}

TEST(AtanGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testGpuAtanTolOnFloat(x, TOL);
}

TEST(AtanGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testGpuAtanTolOnFloat(x, TOL);
}

TEST(AtanGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testGpuAtanEqOnFloat(x);
}

TEST(AtanGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testGpuAtanEqOnFloat(x);
}
*/
// =============================================================================
// TESTING THE HYPERBOLIC SINE FUNCTIONS WITH DOUBLES
// =============================================================================

void testCpuSinhTolOn(
  const std::vector<double> x,
  const double tol
) {
  testCpuTolOn(x, tol, CpuSinh(), LibSinh());
}

void testGpuSinhTolOn(
  const std::vector<double> x,
  const double tol
) {
  testGpuTolOn(x, tol, GpuSinh(), LibSinh());
}

void testCpuSinhEqOn(const std::vector<double> x) {
	testCpuSinhTolOn(x, 0.0);
}

void testGpuSinhEqOn(const std::vector<double> x) {
	testGpuSinhTolOn(x, 0.0);
}

TEST(SinhCPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testCpuSinhEqOn(x);
}

TEST(SinhCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testCpuSinhTolOn(x, TOL);
}

TEST(SinhCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testCpuSinhTolOn(x, TOL);
}

TEST(SinhErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testCpuSinhEqOn(x);
}

TEST(SinhErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testCpuSinhEqOn(x);
}

TEST(SinhGPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testGpuSinhEqOn(x);
}


TEST(SinhGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testGpuSinhTolOn(x, TOL);
}

TEST(SinhGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testGpuSinhTolOn(x, TOL);
}

TEST(SinhGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testGpuSinhEqOn(x);
}

TEST(SinhGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testGpuSinhEqOn(x);
}
/*
// =============================================================================
// TESTING THE HYPERBOLIC SINE FUNCTIONS WITH FLOATING POINTS
// =============================================================================

void testCpuSinhTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testCpuTolOnFloat(x, tol, CpuSinh(), LibSinh());
}

void testGpuSinhTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testGpuTolOnFloat(x, tol, GpuSinh(), LibSinh());
}

void testCpuSinhEqOnFloat(const std::vector<float> x) {
        testCpuSinhTolOnFloat(x, 0.0);
}

void testGpuSinhEqOnFloat(const std::vector<float> x) {
        testGpuSinhTolOnFloat(x, 0.0);
}

TEST(SinhCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testCpuSinhTolOnFloat(x, TOL);
}

TEST(SinhCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testCpuSinhTolOnFloat(x, TOL);
}

TEST(SinhErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testCpuSinhEqOnFloat(x);
}

TEST(SinhErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testCpuSinhEqOnFloat(x);
}

TEST(SinhGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testGpuSinhTolOnFloat(x, TOL);
}

TEST(SinhGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testGpuSinhTolOnFloat(x, TOL);
}

TEST(SinhGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testGpuSinhEqOnFloat(x);
}

TEST(SinhGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testGpuSinhEqOnFloat(x);
}
*/
// =============================================================================
// TESTING THE HYPERBOLIC COSINE FUNCTIONS WITH DOUBLES
// =============================================================================

void testCpuCoshTolOn(
  const std::vector<double> x,
  const double tol
) {
  testCpuTolOn(x, tol, CpuCosh(), LibCosh());
}

void testGpuCoshTolOn(
  const std::vector<double> x,
  const double tol
) {
  testGpuTolOn(x, tol, GpuCosh(), LibCosh());
}

void testCpuCoshEqOn(const std::vector<double> x) {
	testCpuCoshTolOn(x, 0.0);
}

void testGpuCoshEqOn(const std::vector<double> x) {
	testGpuCoshTolOn(x, 0.0);
}

TEST(CoshCPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testCpuCoshEqOn(x);
}

TEST(CoshCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testCpuCoshTolOn(x, TOL);
}

TEST(CoshCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testCpuCoshTolOn(x, TOL);
}

TEST(CoshErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testCpuCoshEqOn(x);
}

TEST(CoshErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testCpuCoshEqOn(x);
}

TEST(CoshGPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testGpuCoshEqOn(x);
}

TEST(CoshGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testGpuCoshTolOn(x, TOL);
}

TEST(CoshGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testGpuCoshTolOn(x, TOL);
}

TEST(CoshGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testGpuCoshEqOn(x);
}

TEST(CoshGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testGpuCoshEqOn(x);
}
/*
// =============================================================================
// TESTING THE HYPERBOLIC COSINE FUNCTIONS WITH FLOATING POINTS
// =============================================================================

void testCpuCoshTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testCpuTolOnFloat(x, tol, CpuCosh(), LibCosh());
}

void testGpuCoshTolOnFloat(
  const std::vector<float> x,
  const double tol
) {
  testGpuTolOnFloat(x, tol, GpuCosh(), LibCosh());
}

void testCpuCoshEqOnFloat(const std::vector<float> x) {
        testCpuCoshTolOnFloat(x, 0.0);
}

void testGpuCoshEqOnFloat(const std::vector<float> x) {
        testGpuCoshTolOnFloat(x, 0.0);
}

TEST(CoshCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testCpuCoshTolOnFloat(x, TOL);
}

TEST(CoshCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testCpuCoshTolOnFloat(x, TOL);
}

TEST(CoshErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testCpuCoshEqOnFloat(x);
}

TEST(CoshErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testCpuCoshEqOnFloat(x);
}

TEST(CoshGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testGpuCoshTolOnFloat(x, TOL);
}

TEST(CoshGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testGpuCoshTolOnFloat(x, TOL);
}

TEST(CoshGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testGpuCoshEqOnFloat(x);
}

TEST(CoshGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testGpuCoshEqOnFloat(x);
}
*/
// =============================================================================
// TESTING THE HYPERBOLIC TANGENT FUNCTIONS WITH DOUBLES
// =============================================================================
void testCpuTanhTolOn(
  const std::vector<double> x,
  const double tol
) {
  testCpuTolOn(x, tol, CpuTanh(), LibTanh());
}

void testGpuTanhTolOn(
  const std::vector<double> x,
  const double tol
) {
  testGpuTolOn(x, tol, GpuTanh(), LibTanh());
}

void testCpuTanhEqOn(const std::vector<double> x) {
	testCpuTanhTolOn(x, 0.0);
}

void testGpuTanhEqOn(const std::vector<double> x) {
	testGpuTanhTolOn(x, 0.0);
}

TEST(TanhCPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testCpuTanhEqOn(x);
}

TEST(TanhCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testCpuTanhTolOn(x, TOL);
}

TEST(TanhCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testCpuTanhTolOn(x, TOL);
}

TEST(TanhErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testCpuTanhEqOn(x);
}

TEST(TanhErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testCpuTanhEqOn(x);
}

TEST(TanhGPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testGpuTanhEqOn(x);
}

TEST(TanhGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testGpuTanhTolOn(x, TOL);
}

TEST(TanhGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testGpuTanhTolOn(x, TOL);
}

TEST(TanhGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testGpuTanhEqOn(x);
}

TEST(TanhGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testGpuTanhEqOn(x);
}
/*
// =============================================================================
// TESTING THE HYPERBOLIC TANGENT FUNCTIONS WITH FLOATING POINTS
// =============================================================================
void testCpuTanhTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testCpuTolOnFloat(x, tol, CpuTanh(), LibTanh());
}

void testGpuTanhTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testGpuTolOnFloat(x, tol, GpuTanh(), LibTanh());
}

void testCpuTanhEqOnFloat(const std::vector<float> x) {
        testCpuTanhTolOnFloat(x, 0.0);
}

void testGpuTanhEqOnFloat(const std::vector<float> x) {
        testGpuTanhTolOnFloat(x, 0.0);
}

TEST(TanhCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testCpuTanhTolOnFloat(x, TOL);
}

TEST(TanhCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testCpuTanhTolOnFloat(x, TOL);
}

TEST(TanhErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testCpuTanhEqOnFloat(x);
}

TEST(TanhErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testCpuTanhEqOnFloat(x);
}

TEST(TanhGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testGpuTanhTolOnFloat(x, TOL);
}

TEST(TanhGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testGpuTanhTolOnFloat(x, TOL);
}

TEST(TanhGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testGpuTanhEqOnFloat(x);
}

TEST(TanhGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testGpuTanhEqOnFloat(x);
}
*/

// =============================================================================
// TESTING THE INVERSE HYPERBOLIC SINE FUNCTIONS WITH DOUBLES
// =============================================================================
void testCpuAsinhTolOn(
  const std::vector<double> x,
  const double tol
) {
  testCpuTolOn(x, tol, CpuAsinh(), LibAsinh());
}

void testGpuAsinhTolOn(
  const std::vector<double> x,
  const double tol
) {
  testGpuTolOn(x, tol, GpuAsinh(), LibAsinh());
}

void testCpuAsinhEqOn(const std::vector<double> x) {
	testCpuAsinhTolOn(x, 0.0);
}

void testGpuAsinhEqOn(const std::vector<double> x) {
	testGpuAsinhTolOn(x, 0.0);
}

TEST(AsinhCPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testCpuAsinhEqOn(x);
}

TEST(AsinhCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testCpuAsinhTolOn(x, TOL);
}

TEST(AsinhCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testCpuAsinhTolOn(x, TOL);
}

TEST(AsinhErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testCpuAsinhEqOn(x);
}

TEST(AsinhErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testCpuAsinhEqOn(x);
}

TEST(AsinhGPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testGpuAsinhEqOn(x);
}

TEST(AsinhGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testGpuAsinhTolOn(x, TOL);
}

TEST(AsinhGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testGpuAsinhTolOn(x, TOL);
}

TEST(AsinhGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testGpuAsinhEqOn(x);
}

TEST(AsinhGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testGpuAsinhEqOn(x);
}
/*
// =============================================================================
// TESTING THE INVERSE HYPERBOLIC SINE FUNCTIONS WITH FLOATING POINTS
// =============================================================================
void testCpuAsinhTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testCpuTolOnFloat(x, tol, CpuAsinh(), LibAsinh());
}

void testGpuAsinhTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testGpuTolOnFloat(x, tol, GpuAsinh(), LibAsinh());
}

void testCpuAsinhEqOnFloat(const std::vector<float> x) {
        testCpuAsinhTolOnFloat(x, 0.0);
}

void testGpuAsinhEqOnFloat(const std::vector<float> x) {
        testGpuAsinhTolOnFloat(x, 0.0);
}

TEST(AsinhCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testCpuAsinhTolOnFloat(x, TOL);
}

TEST(AsinhCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testCpuAsinhTolOnFloat(x, TOL);
}

TEST(AsinhErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testCpuAsinhEqOnFloat(x);
}

TEST(AsinhErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testCpuAsinhEqOnFloat(x);
}

TEST(AsinhGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testGpuAsinhTolOnFloat(x, TOL);
}

TEST(AsinhGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testGpuAsinhTolOnFloat(x, TOL);
}

TEST(AsinhGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testGpuAsinhEqOnFloat(x);
}

TEST(AsinhGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testGpuAsinhEqOnFloat(x);
}
*/

// =============================================================================
// TESTING THE INVERSE HYPERBOLIC COSINE FUNCTIONS WITH DOUBLES
// =============================================================================
void testCpuAcoshTolOn(
  const std::vector<double> x,
  const double tol
) {
  testCpuTolOn(x, tol, CpuAcosh(), LibAcosh());
}

void testGpuAcoshTolOn(
  const std::vector<double> x,
  const double tol
) {
  testGpuTolOn(x, tol, GpuAcosh(), LibAcosh());
}

void testCpuAcoshEqOn(const std::vector<double> x) {
	testCpuAcoshTolOn(x, 0.0);
}

void testGpuAcoshEqOn(const std::vector<double> x) {
	testGpuAcoshTolOn(x, 0.0);
}

TEST(AcoshCPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testCpuAcoshEqOn(x);
}

TEST(AcoshCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testCpuAcoshTolOn(x, TOL);
}

TEST(AcoshCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testCpuAcoshTolOn(x, TOL);
}

TEST(AcoshErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testCpuAcoshEqOn(x);
}

TEST(AcoshErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testCpuAcoshEqOn(x);
}

TEST(AcoshGPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testGpuAcoshEqOn(x);
}

TEST(AcoshGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testGpuAcoshTolOn(x, TOL);
}

TEST(AcoshGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testGpuAcoshTolOn(x, TOL);
}

TEST(AcoshGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testGpuAcoshEqOn(x);
}

TEST(AcoshGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testGpuAcoshEqOn(x);
}
/*
// =============================================================================
// TESTING THE INVERSE HYPERBOLIC COSINE FUNCTIONS WITH FLOATING POINTS
// =============================================================================
void testCpuAcoshTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testCpuTolOnFloat(x, tol, CpuAcosh(), LibAcosh());
}

void testGpuAcoshTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testGpuTolOnFloat(x, tol, GpuAcosh(), LibAcosh());
}

void testCpuAcoshEqOnFloat(const std::vector<float> x) {
        testCpuAcoshTolOnFloat(x, 0.0);
}

void testGpuAcoshEqOnFloat(const std::vector<float> x) {
        testGpuAcoshTolOnFloat(x, 0.0);
}

TEST(AcoshCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testCpuAcoshTolOnFloat(x, TOL);
}

TEST(AcoshCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testCpuAcoshTolOnFloat(x, TOL);
}

TEST(AcoshErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testCpuAcoshEqOnFloat(x);
}

TEST(AcoshErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testCpuAcoshEqOnFloat(x);
}

TEST(AcoshGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testGpuAcoshTolOnFloat(x, TOL);
}

TEST(AcoshGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testGpuAcoshTolOnFloat(x, TOL);
}

TEST(AcoshGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testGpuAcoshEqOnFloat(x);
}

TEST(AcoshGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testGpuAcoshEqOnFloat(x);
}
*/

// =============================================================================
// TESTING THE INVERSE HYPERBOLIC TANGENT FUNCTIONS WITH DOUBLES
// =============================================================================
void testCpuAtanhTolOn(
  const std::vector<double> x,
  const double tol
) {
  testCpuTolOn(x, tol, CpuAtanh(), LibAtanh());
}

void testGpuAtanhTolOn(
  const std::vector<double> x,
  const double tol
) {
  testGpuTolOn(x, tol, GpuAtanh(), LibAtanh());
}

void testCpuAtanhEqOn(const std::vector<double> x) {
	testCpuAtanhTolOn(x, 0.0);
}

void testGpuAtanhEqOn(const std::vector<double> x) {
	testGpuAtanhTolOn(x, 0.0);
}

TEST(AtanhCPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testCpuAtanhEqOn(x);
}

TEST(AtanhCPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testCpuAtanhTolOn(x, TOL);
}

TEST(AtanhCPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testCpuAtanhTolOn(x, TOL);
}

TEST(AtanhErrCPUTest, Zero) {
  std::vector<double> x = {0.0};
  testCpuAtanhEqOn(x);
}

TEST(AtanhErrCPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testCpuAtanhEqOn(x);
}

TEST(AtanhGPUTest, Nans) {
  double posNAN = NAN;
  double negNAN = -posNAN;
  std::vector<double> x = {negNAN, posNAN};
  testGpuAtanhEqOn(x);
}

TEST(AtanhGPUTest, PositiveValues) {
  std::vector<double> x = randomFill(0, 100, N);
  testGpuAtanhTolOn(x, TOL);
}

TEST(AtanhGPUTest, NegativeValues) {
  std::vector<double> x = randomFill(-100, 0, N);
  testGpuAtanhTolOn(x, TOL);
}

TEST(AtanhGPUTest, Zero) {
  std::vector<double> x = {0.0};
  testGpuAtanhEqOn(x);
}

TEST(AtanhGPUTest, Infinity) {
  double posInf = std::numeric_limits<double>::infinity();
  double negInf = -posInf;
  std::vector<double> x = {negInf, posInf};
  testGpuAtanhEqOn(x);
}
/*
// =============================================================================
// TESTING THE INVERSE HYPERBOLIC TANGENT FUNCTIONS WITH FLOATING POINTS
// =============================================================================
void testCpuAtanhTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testCpuTolOnFloat(x, tol, CpuAtanh(), LibAtanh());
}

void testGpuAtanhTolOnFloat(
  const std::vector<float> x,
  const float tol
) {
  testGpuTolOnFloat(x, tol, GpuAtanh(), LibAtanh());
}

void testCpuAtanhEqOnFloat(const std::vector<float> x) {
        testCpuAtanhTolOnFloat(x, 0.0);
}

void testGpuAtanhEqOnFloat(const std::vector<float> x) {
        testGpuAtanhTolOnFloat(x, 0.0);
}

TEST(AtanhCPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testCpuAtanhTolOnFloat(x, TOL);
}

TEST(AtanhCPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testCpuAtanhTolOnFloat(x, TOL);
}

TEST(AtanhErrCPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testCpuAtanhEqOnFloat(x);
}

TEST(AtanhErrCPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testCpuAtanhEqOnFloat(x);
}

TEST(AtanhGPUTestFloat, PositiveValues) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  testGpuAtanhTolOnFloat(x, TOL);
}

TEST(AtanhGPUTestFloat, NegativeValues) {
  std::vector<float> x = randomFillFloat(-100, 0, N);
  testGpuAtanhTolOnFloat(x, TOL);
}

TEST(AtanhGPUTestFloat, Zero) {
  std::vector<float> x = {0.0};
  testGpuAtanhEqOnFloat(x);
}

TEST(AtanhGPUTestFloat, Infinity) {
  float posInf = std::numeric_limits<float>::infinity();
  float negInf = -posInf;
  std::vector<float> x = {negInf, posInf};
  testGpuAtanhEqOnFloat(x);
}

*/

// ENTRY POINT
// =============================================================================
int main(int argc, char **argv) {
  //srand(time(NULL));
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
