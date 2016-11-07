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
  double sin(const double x);	
  double cos(const double x);
  double tan(const double x);
  double asin(const double x);
  double acos(const double x);
  double atan(const double x);
  double sinh(const double x);
  double cosh(const double x);
  double tanh(const double x);
  double asinh(const double x);
  double acosh(const double x);
  double atanh(const double x);
// Floating points
  float logFloat(const float x);
  float exp(float x);
  float pow(const float x, const float y);
  float sin(const float x);
  float cos(const float x);
  float tan(const float x);
  float asin(const float x);
  float acos(const float x);
  float atan(const float x);
  float sinh(const float x);
  float cosh(const float x);
  float tanh(const float x);
  float asinh(const float x);
  float acosh(const float x);
  float atanh(const float x);

}

namespace gm {
  __device__ double log(const double x);
  __device__ double exp(double x);
  __device__ double pow(const double x, const double y);
  __device__ double sin(const double x);	
  __device__ double cos(const double x);
  __device__ double tan(const double x);
  __device__ double asin(const double x);
  __device__ double acos(const double x);
  __device__ double atan(const double x);
  __device__ double sinh(const double x);
  __device__ double cosh(const double x);
  __device__ double tanh(const double x);
  __device__ double asinh(const double x);
  __device__ double acosh(const double x);
  __device__ double atanh(const double x);
// FLOATING POINTS
  __device__ float logFloat(const float x);
  __device__ float exp(float x);
  __device__ float pow(const float x, const float y);
  __device__ float sin(const float x);
  __device__ float cos(const float x);
  __device__ float tan(const float x);
  __device__ float asin(const float x);
  __device__ float acos(const float x);
  __device__ float atan(const float x);
  __device__ float sinh(const float x);
  __device__ float cosh(const float x);
  __device__ float tanh(const float x);
  __device__ float asinh(const float x);
  __device__ float acosh(const float x);
  __device__ float atanh(const float x);


}

// =============================================================================
// LOG FUNCTORS WITH DOUBLES
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
/*
// =============================================================================
//  LOG FUNCTORS WITH FLOATING POINTS
// =============================================================================
class GpuLogFloat {
  public:
    __device__ float operator() (float x) const
    {
      float y = gm::logFloat(x);
      //double y = friendly_log(x);
      //printf("[GPU FUNCTOR] %f -> %f\n", x, y);
      return y;
    }
};

class CpuLogFloat {
  public:
    float operator() (float x) const
    {
      return cm::logFloat(x);
      //return cpu_friendly_log(x);
    }
};

class LibLogFloat {
  public:
    float operator() (float x) const
    {
      return log(x);
    }
};
*/

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

// =============================================================================
// SINE FUNCTORS
// =============================================================================
class GpuSin{
  public:
   __device__ double operator()(double x) const
   {
     return gm::sin(x);
     //retun friendly_sin(x);
   }		
};

class CpuSin{
  public:
   double operator()(double x) const
   {
     return cm::sin(x);
     //retun cpu_friendly_sin(x);
   }
}; 
class LibSin {
  public:
   double operator() (double x) const
   {
     return sin(x);
   }
};

// =============================================================================
//  COSINE FUNCTORS
// =============================================================================
class GpuCos{
  public:
   __device__ double operator()(double x) const
   {
     return gm::cos(x);
     //retun friendly_cos(x);
   }
};

class CpuCos{
  public:
   double operator()(double x) const
   {
     return cm::cos(x);
     //retun cpu_friendly_cos(x);
   }
};
class LibCos {
  public:
   double operator() (double x) const
   {
     return cos(x);
   }
};

// =============================================================================
//   TANGENT FUNCTORS
// =============================================================================
class GpuTan{
  public:
   __device__ double operator()(double x) const
   {
     return gm::tan(x);
     //retun friendly_tan(x);
   }
};

class CpuTan{
  public:
   double operator()(double x) const
   {
     return cm::tan(x);
     //retun cpu_friendly_tan(x);
   }
};
class LibTan {
  public:
   double operator() (double x) const
   {
     return tan(x);
   }
};


// =============================================================================
// INVERSE OF  SINE FUNCTORS
// =============================================================================

class GpuAsin{
  public:
   __device__ double operator()(double x) const
   {
     return gm::asin(x);
     //return gm::acos(x);
     //retun friendly_acos(x);
   }
};
   
class CpuAsin{
  public:
   double operator()(double x) const
   {
     return cm::asin(x);
     //retun cpu_friendly_Asin(x);
   }
};

class LibAsin {
  public:
   double operator() (double x) const
   {
     return asin(x);
   }
};

// =============================================================================
//  INVERSE OF  COSINE FUNCTORS
// =============================================================================
class GpuAcos{
  public:
   __device__ double operator()(double x) const
   {
     return gm::acos(x);
     //retun friendly_acos(x);
  }
};

class CpuAcos{
  public:
   double operator()(double x) const
   {
     return cm::acos(x);
     //retun cpu_friendly_acos(x);
   }
};
class LibAcos {
  public:
   double operator() (double x) const
   {
     return acos(x);
   }
};


// =============================================================================
//   INVERSE OF TANGENT FUNCTORS
// =============================================================================
class GpuAtan{
  public:
   __device__ double operator()(double x) const
   {
     return gm::atan(x);
     //retun friendly_atan(x);
  }
};

class CpuAtan{
  public:
   double operator()(double x) const
   {
     return cm::atan(x);
     //retun cpu_friendly_atan(x);
   }
};
class LibAtan {
  public:
   double operator() (double x) const
   {
     return atan(x);
   }
};

// =============================================================================
//    HYPERBOLIC SINE FUNCTORS
// =============================================================================
class GpuSinh{
  public:
   __device__ double operator()(double x) const
   {
     return gm::sinh(x);
      //return friendly_sinh(x);
    }
};

class CpuSinh {
  public:
    double operator() (double x) const
    {
      return cm::sinh(x);
      //return cpu_friendly_sinh(x);
    }
};

class LibSinh {
  public:
    double operator() (double x) const
    {
      return sinh(x);
    }
};

// =============================================================================
//     HYPERBOLIC COSINE FUNCTORS
// =============================================================================
class GpuCosh{
  public:
   __device__ double operator()(double x) const
   {
     return gm::cosh(x);
      //return friendly_cosh(x);
    }
};

class CpuCosh {
  public:
    double operator() (double x) const
    {
      return cm::cosh(x);
      //return cpu_friendly_cosh(x);
    }
};

class LibCosh {
  public:
    double operator() (double x) const
    {
      return cosh(x);
    }
};

// =============================================================================
//     HYPERBOLIC TANGENT FUNCTORS
// =============================================================================
class GpuTanh{
  public:
   __device__ double operator()(double x) const
   {
     return gm::tanh(x);
      //return friendly_tanh(x);
    }
};

class CpuTanh {
  public:
    double operator() (double x) const
    {
      return cm::tanh(x);
      //return cpu_friendly_tanh(x);
    }
};

class LibTanh {
  public:
    double operator() (double x) const
    {
      return tanh(x);
    }
};

// =============================================================================
//   INVERSE HYPERBOLIC SINE FUNCTORS
// =============================================================================
class GpuAsinh{
  public:
   __device__ double operator()(double x) const
   {
     return gm::asinh(x);
      //return friendly_asinh(x);
    }
};

class CpuAsinh {
  public:
    double operator() (double x) const
    {
      return cm::asinh(x);
      //return cpu_friendly_asinh(x);
    }
};

class LibAsinh {
  public:
    double operator() (double x) const
    {
      return asinh(x);
    }
};

// =============================================================================
//   INVERSE HYPERBOLIC COSINE FUNCTORS
// =============================================================================
class GpuAcosh{
  public:
   __device__ double operator()(double x) const
   {
     return gm::acosh(x);
      //return friendly_acosh(x);
    }
};

class CpuAcosh {
  public:
    double operator() (double x) const
    {
      return cm::acosh(x);
      //return cpu_friendly_acosh(x);
    }
};

class LibAcosh {
  public:
    double operator() (double x) const
    {
      return acosh(x);
    }
};


// =============================================================================
//   INVERSE HYPERBOLIC TANGENT FUNCTORS
// =============================================================================
class GpuAtanh{
  public:
   __device__ double operator()(double x) const
   {
     return gm::atanh(x);
      //return friendly_atanh(x);
    }
};

class CpuAtanh {
  public:
    double operator() (double x) const
    {
      return cm::atanh(x);
      //return cpu_friendly_atanh(x);
    }
};

class LibAtanh {
  public:
    double operator() (double x) const
    {
      return atanh(x);
    }
};

