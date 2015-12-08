#include "test_tools.h"
// isnan, isinf
#include <cmath>
// srand, rand
#include <stdlib.h>
// ptr
#include <memory>

// =============================================================================
// ARRAY FILL HELPERS
// =============================================================================
// Fill the array ary of size n with random values in [a,b)
void randomFill(int a, int b, double* ary, int n) {
  for (unsigned i = 0; i < n; i++) {
    ary[i] = (b - a) * ((double) rand()/(double) RAND_MAX) + a;
  }
}

// Fill the array ary of size n with random values in [a,b)
std::unique_ptr<std::vector<double>> randomFill(int a, int b, int n) {
  auto y = std::unique_ptr<std::vector<double>>(new std::vector<double>(n));
  for (unsigned i = 0; i < n; i++) {
    (*y)[i] = (b - a) * ((double) rand()/(double) RAND_MAX) + a;
  }
  return y;
}

// Fill the array ary of size n random values in [0,1)
void randomFill(double* ary, int n) {
  return randomFill(0, 1, ary, n);
}

// Fill the array ary of size n with a given value
void valueFill(double* ary, double d, int n) {
  for (unsigned i = 0; i < n; i++) {
    ary[i] = d;
  }
}

// Fill the array ary of size n with a given value
void zeroFill(double* ary, int n) {
  valueFill(ary, 0, n);
}

// =============================================================================
// ERROR HELPERS
// =============================================================================

// Compute the absolute error between val and expected
double absoluteError(double val, double expected) {
  return abs(val - expected);
}

// Compute the relative error between val and expected
double relativeError(double val, double expected) {
  return absoluteError(val, expected)/val;
}
