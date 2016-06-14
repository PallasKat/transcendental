// mch cpu math
#include "cpu_mch_math.h"
// mch gpu math
#include "gpu_mch_math.h"
// lib math
#include <cmath>

// =============================================================================
// LOG FUNCTORS
// =============================================================================
class GpuLog {
  public:
    __device__ double operator() (double x) const
    {
      double y = friendly_log(x);
      //printf("[GPU FUNCTOR] %f -> %f\n", x, y);
      return y;
    }
};

class CpuLog {
  public:
    double operator() (double x) const
    {
      return cpu_friendly_log(x);
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
      return friendly_exp(x);
    }
};

class CpuExp {
  public:
    double operator() (double x) const
    {
      return cpu_friendly_exp(x);
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
      return friendly_pow(x, y);
    }
};

class CpuPow {
  public:
    double operator() (double x, double y) const
    {
      return cpu_friendly_pow(x, y);
    }
};

class LibPow {
  public:
    double operator() (double x, double y) const
    {
      return pow(x, y);
    }
};
