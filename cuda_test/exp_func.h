// mch cpu math
#include "cpu_mch_math.h"
// mch gpu math
#include "gpu_mch_math.h"
// lib math
#include <cmath>

class GpuExp {
  public:
    __device__ double operator() (double x) const
    {
      return friendly_log(x);
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
