// std
#include <iostream>
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
    sendToDevice(pDevX, x.data(), N);
  }

  virtual void TearDown() {  
    // freeing memory on the device
    gpuErrchk(cudaFree(pDevX));
    gpuErrchk(cudaFree(pDevY));
  }  
};

class Random_0_100: public Params<0,100,1000> {};

class Random_0_1000: public Params<0,100,10000> {};
