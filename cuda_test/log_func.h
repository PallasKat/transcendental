// mch cpu math
#include "cpu_mch_math.h"
// mch gpu math
#include "gpu_mch_math.h"
// lib math
#include <cmath>

class GpuLog {
  public:
    __device__ double operator() (double x) const
    {
      return friendly_log(x);
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
