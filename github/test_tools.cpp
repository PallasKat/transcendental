#include "test_tools.h"
// isnan, isinf
#include <cmath>
// srand, rand
#include <stdlib.h>
#include <iostream>

// =============================================================================
// VECTOR HELPER
// =============================================================================

void printVector(const std::vector<double>& v) {
  const size_t n = v.size();
  for (auto i = 0; i < n; i++) {
    std::cout << v[i] << " ";
  }
  std::cout << std::endl;
}

// =============================================================================
// ARRAY FILL HELPERS:
// This is mainly used to fill inputs for the test functions
// =============================================================================

// -----------------------------------------------------------------------------
// Random real values
// -----------------------------------------------------------------------------

// return a random double in [a,b)
inline double randomDouble(int a, int b) {
  return (b - a)*((double) rand()/(double) RAND_MAX) + a;
}

// Fill the array ary of size n with random values in [a,b)
std::vector<double> randomFill(int a, int b, size_t n) {
  std::vector<double> y(n);
  for (size_t i = 0; i < n; i++) {
    y[i] = randomDouble(a, b);
  }
  return y;
}

// Fill the array ary of size n random values in [0,1)
std::vector<double> randomFill(size_t n) {
  return randomFill(0, 1, n);
}

// Fill the array ary of size n with random values x in [a,b)
// whose truncated value is even
std::vector<double> randomEvenFill(int a, int b, size_t n) {
  std::vector<double> y(n);
  for (size_t i = 0; i < n; i++) {
    double t = randomDouble(a, b);
    while (((int) trunc(t)) % 2 != 0) {
      t = randomDouble(a, b);
    }
    y[i] = t;
  }
  return y;
}

// -----------------------------------------------------------------------------
// Random integer values (as double)
// -----------------------------------------------------------------------------

// return a random integer (as double) in [a,b]
inline double randomInteger(int a, int b) {
  int x = a + (rand() % (int)(b - a + 1));
  return (double) x;
}

// Fill the array ary of size n with random integer values in [a,b)
std::vector<double> randomIntFill(int a, int b, size_t n) {
  std::vector<double> y(n);
  for (size_t i = 0; i < n; i++) {
    y[i] = randomInteger(a, b);
  }
  return y;
}

// -----------------------------------------------------------------------------
// Fixed values
// -----------------------------------------------------------------------------

// Fill the array ary of size n with a given value
std::vector<double> valueFill(double d, size_t n) {
  std::vector<double> ary(n);
  for (size_t i = 0; i < n; i++) {
    ary[i] = d;
  }
  return ary;
}

// Fill the array ary of size n with a given value
std::vector<double> zeroFill(size_t n) {
  return valueFill(0, n);
}

// =============================================================================
// ERROR HELPERS:
// Mainly used to compute the error betweem a reference and an expected value
// in the test functions
// =============================================================================

// Compute the absolute error between val and expected
double absoluteError(double val, double trueValue) {
  return std::abs(val - trueValue);
}

// Compute the relative error between val and expected
double relativeError(double val, double trueValue) {
  // this should be discussed
  if (trueValue == 0) {
    return absoluteError(val, trueValue);
  } else {
    return absoluteError(val, trueValue)/trueValue;
  }
}
