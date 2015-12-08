// ptr
#include <memory>
// vector
#include <vector>

// Fill the array ary of size n with random values in [a,b)
void randomFill(int a, int b, double* ary, int n);

// Fill the array ary of size n with random values in [a,b)
std::unique_ptr<std::vector<double>> randomFill(int a, int b, int n);

// Fill the array ary of size n random values in [0,1)
void randomFill(double* ary, int n);

// Fill the array ary of size n with a given value
void valueFill(double* ary, double d, int n);

// Fill the array ary of size n with a given value
void zeroFill(double* ary, int n);

// Compute the absolute error between val and expected
double absoluteError(double val, double expected);

// Compute the relative error between val and expected
double relativeError(double val, double expected);

void verifyTol(double* expectedValues, double* values, int n, double tol);

void verifyEq(double* expectedValues, double* values, int n);
