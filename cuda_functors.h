// =============================================================================
// CUDA Math functors With Doubles
// =============================================================================

class CudaLog {
  public:
    __device__ double operator() (double x) const {
      return log(x);
    }
};

class CudaExp {
  public:
    __device__ double operator() (double x) const {
      return exp(x);
    }
};

class CudaPow {
  public:
    __device__ double operator() (double x, double y) const {
      return pow(x, y);
    }
};

class CudaSin {
  public:
    __device__ double operator() (double x) const {
      return sin(x);
    }
};

class CudaCos {
  public:
    __device__ double operator() (double x) const {
      return cos(x);
    }
};

class CudaTan {
  public:
    __device__ double operator() (double x) const {
      return tan(x);
    }
};

class CudaAsin {
  public:
    __device__ double operator() (double x) const {
      return asin(x);
    }
};

class CudaAcos {
  public:
    __device__ double operator() (double x) const {
      return acos(x);
    }
};

class CudaAtan {
  public:
    __device__ double operator() (double x) const {
      return atan(x);
    }
};

class CudaSinh {
  public:
    __device__ double operator() (double x) const {
      return sinh(x);
    }
};

class CudaCosh {
  public:
    __device__ double operator() (double x) const {
      return cosh(x);
    }
};

class CudaTanh {
  public:
    __device__ double operator() (double x) const {
      return tanh(x);
    }
};

class CudaAsinh {
  public:
    __device__ double operator() (double x) const {
      return asinh(x);
    }
};

class CudaAcosh {
  public:
    __device__ double operator() (double x) const {
      return acosh(x);
    }
};

class CudaAtanh {
  public:
    __device__ double operator() (double x) const {
      return atanh(x);
    }
};

// =============================================================================
//  CUDA Math functors With Floats
// =============================================================================

class CudaLogFloat {
  public:
    __device__ float operator() (float x) const {
      return log(x);
    }
};

class CudaExpFloat {
  public:
    __device__ float operator() (float x) const {
      return exp(x);
    }
};

class CudaPowFloat {
  public:
    __device__ float operator() (float x, float y) const {
      return pow(x, y);
    }
};

class CudaSinFloat {
  public:
    __device__ float operator() (float x) const {
      return sin(x);
    }
};

class CudaCosFloat {
  public:
    __device__ float operator() (float x) const {
      return cos(x);
    }
};

class CudaTanFloat {
  public:
    __device__ float operator() (float x) const {
      return tan(x);
    }
};

class CudaAsinFloat {
  public:
    __device__ float operator() (float x) const {
      return asin(x);
    }
};

class CudaAcosFloat {
  public:
    __device__ float operator() (float x) const {
      return acos(x);
    }
};

class CudaAtanFloat {
  public:
    __device__ float operator() (float x) const {
      return atan(x);
    }
};

class CudaSinhFloat {
  public:
    __device__ float operator() (float x) const {
      return sinh(x);
    }
};

class CudaCoshFloat {
  public:
    __device__ float operator() (float x) const {
      return cosh(x);
    }
};

class CudaTanhFloat {
  public:
    __device__ float operator() (float x) const {
      return tanh(x);
    }
};

class CudaAsinhFloat {
  public:
    __device__ float operator() (float x) const {
      return asinh(x);
    }
};

class CudaAcoshFloat {
  public:
    __device__ float operator() (float x) const {
      return acosh(x);
    }
};

class CudaAtanhFloat {
  public:
    __device__ float operator() (float x) const {
      return atanh(x);
    }
};

