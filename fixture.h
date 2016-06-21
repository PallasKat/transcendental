// ========================================================
// Fixture to avoid taking into account memory copy 
// operations in the benchmark
// ========================================================

// ====================================================
// For functions of the type y -> f(x)
// ====================================================

template<int NA, int NB, int NN>
class Params: public ::hayai::Fixture {
public:  
  int A = NA;
  int B = NB;
  int N = NN;

  // data structure on which computation will be applied
  std::vector<double> x = randomFill(A, B, N);
  double* pDevX = NULL;
  double* pDevY = NULL;

  Params() {}

  virtual void SetUp() {    
    // copy data to device memory
    sendToDevice(pDevX, x.data(), N);
  }

  virtual void TearDown() {  
    // freeing memory on the device
    gpuErrchk(cudaFree(pDevX));
    gpuErrchk(cudaFree(pDevY));
  }  
};

class Random_100: public Params<0,100,1000> {};

class Random_1000: public Params<0,100,10000> {};

// ====================================================
// For functions of the type z -> f(x, y)
// ====================================================

template<int NAx, int NBx, int NAy, int NBy, int NN>
class Params2: public ::hayai::Fixture {
public:  
  int Ax = NAx;
  int Bx = NBx;
  int Ay = NAy;
  int By = NBy;
  int N = NN;

  // data structure on which computation will be applied
  std::vector<double> x = randomFill(Ax, Bx, N);
  std::vector<double> y = randomFill(Ay, By, N);

  double* pDevX = NULL;
  double* pDevY = NULL;
  double* pDevZ = NULL;

  Params2() {}

  virtual void SetUp() {    
    // copy data to device memory
    sendToDevice(pDevX, x.data(), N);
    sendToDevice(pDevY, y.data(), N);
  }

  virtual void TearDown() {  
    // freeing memory on the device
    gpuErrchk(cudaFree(pDevX));
    gpuErrchk(cudaFree(pDevY));
    gpuErrchk(cudaFree(pDevZ));
  }  
};

class Random2_1000: public Params2<0,100,-100,100,10000> {};
