// mch cpu math
//#include "cpu_mch_math.h"
// mch gpu math
//#include "gpu_mch_math.h"
// lib math
#include <cmath>

namespace cm {
  double log(const double x);
  double exp(double x);
  double pow(const double x, const double y);
}

namespace gm {
  __device__ double log(const double x);
  __device__ double exp(double x);
  __device__ double pow(const double x, const double y);
}

// =============================================================================
// LOG FUNCTORS
// =============================================================================
class GpuLog {
  public:
    __device__ double operator() (double x) const
    {
      double y = gm::log(x);
      //double y = friendly_log(x);
      //printf("[GPU FUNCTOR] %f -> %f\n", x, y);
      return y;
    }
};

class CpuLog {
  public:
    double operator() (double x) const
    {
      return cm::log(x);
      //return cpu_friendly_log(x);
    }
};

class LibLog {
  public:
    double operator() (double x) const
    {
      return log(x);
    }
};

// =============================================================================
// EXP FUNCTORS
// =============================================================================
class GpuExp {
  public:
    __device__ double operator() (double x) const
    {
      return gm::exp(x);
      //return friendly_exp(x);
    }
};

class CpuExp {
  public:
    double operator() (double x) const
    {
      return cm::exp(x);
      //return cpu_friendly_exp(x);
    }
};

class LibExp {
  public:
    double operator() (double x) const
    {
      return exp(x);
    }
};

// =============================================================================
// POW FUNCTORS
// =============================================================================
class GpuPow {
  public:
    __device__ double operator() (double x, double y) const
    {
      return gm::pow(x, y);
      //return friendly_pow(x, y);
    }
};

class CpuPow {
  public:
    double operator() (double x, double y) const
    {
      return cm::pow(x, y);
      //return cpu_friendly_pow(x, y);
    }
};

class LibPow {
  public:
    double operator() (double x, double y) const
    {
      return pow(x, y);
    }
};
