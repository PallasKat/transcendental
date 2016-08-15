// ptr
#include <memory>
// assert
#include <assert.h>
// stdint definitions
#include <stdint.h>

// =============================================================================
// FUNCTORS AND FUNCTOR HELPERS
// =============================================================================

// -----------------------------------------------------------------------------
// Functor for the functions of the kind y = f(x), Double -> Double
// -----------------------------------------------------------------------------

template <class F>
std::vector<double> applyCpuOp1(
  const std::vector<double>& x,
  F functor
) {
  std::vector<double> y(x.size());
  for (auto i = 0; i < x.size(); i++) {
    y[i] = functor(x[i]);
  }

  return y;
}

template <class F>
std::vector<float> applyCpuOp1Float(
  const std::vector<float>& x,
  F functor
) {
  std::vector<float> y(x.size());
  for (auto i = 0; i < x.size(); i++) {
    y[i] = functor(x[i]);
  }

  return y;
}

// -----------------------------------------------------------------------------
// Functor for the function of the kind z = f(x, y), (Double x Double) -> Double
// -----------------------------------------------------------------------------

template <class F>
std::vector<double> applyCpuOp2(
  const std::vector<double>& x,
  const std::vector<double>& y,
  F functor
) {
  assert(x.size() == y.size());
  std::vector<double> z(x.size());
  for (auto i = 0; i < x.size(); i++) {
    z[i] = functor(x[i], y[i]);
  }
  return z;
}

template <class F>
std::vector<float> applyCpuOp2Float(
  const std::vector<float>& x,
  const std::vector<float>& y,
  F functor
) {
  assert(x.size() == y.size());
  std::vector<float> z(x.size());
  for (auto i = 0; i < x.size(); i++) {
    z[i] = functor(x[i], y[i]);
  }
  return z;
}

