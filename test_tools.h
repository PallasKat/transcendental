// ptr
#include <memory>
// vector
#include <vector>

// Print a vector on standard output
void printVector(const std::vector<double>& v);

// Fill the array ary of size n with random values in [a,b)
std::vector<double> randomFill(int a, int b, size_t n);

// Fill the array ary of size n random values in [0,1)
std::vector<double> randomFill(size_t n);

// Fill the array ary of size n with a given value
std::vector<double> valueFill(double d, size_t n);

// Fill the array ary of size n with a given value
std::vector<double> zeroFill(size_t n);

// Fill the array ary of size n with random values x in [a,b)
// where the floor of x, if x > 0, is even and if x < 0 the
// ceil is even
std::vector<double> randomEvenFill(int a, int b, size_t n) ;

// Fill the array ary of size n with random integer values in [a,b)
std::vector<double> randomIntFill(int a, int b, size_t n);

// Compute the absolute error between val and expected
double absoluteError(double val, double expected);

// Compute the relative error between val and expected
double relativeError(double val, double expected);

//FLOATING POINTS 

// Print a vector on standard output
void printVector(const std::vector<double>& v);

// Fill the array ary of size n with random values in [a,b)
std::vector<float> randomFillFloat(int a, int b, size_t n);

// Fill the array ary of size n random values in [0,1)
std::vector<float> randomFillFloat(size_t n);

// Fill the array ary of size n with a given value
std::vector<float> valueFillFloat(float d, size_t n);

// Fill the array ary of size n with a given value
std::vector<float> zeroFillFloat(size_t n);

// Fill the array ary of size n with random values x in [a,b)
// where the floor of x, if x > 0, is even and if x < 0 the
// // ceil is even
std::vector<float> randomEvenFillFloat(int a, int b, size_t n) ;

// Fill the array ary of size n with random integer values in [a,b)
std::vector<float> randomIntFillFloat(int a, int b, size_t n);

// Compute the absolute error between val and expected
float absoluteErrorFloat(double val, double expected);

// Compute the relative error between val and expected
 float relativeErrorFloat(double val, double expected);

