// mch cpu math
#include "cpu_mch_math.h"
// mch gpu math
#include "gpu_mch_math.h"
// lib math
#include <cmath>

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
