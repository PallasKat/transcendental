// =============================================================================
// CUDA Math functors
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
