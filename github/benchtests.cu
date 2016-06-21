#include <hayai.hpp>
// gpu [log, exp, pow], cpu [log, ...], lib [log, ...]
#include "trans_functors.h"
// to use the functors on vectors
#include "functors.h"
// to use the cuda functors on vectors
#include "cufunctors.h"
// to use the cuda functors on vectors
#include "cuda_functors.h"
// test helpers
#include "test_tools.h"
// fixture
#include "fixture.h"

const size_t N = 10*1000;
int A = 0;
int B = 100;
int NN = N;

int* pA = &A;
int* pB = &B;
int* pN = &NN;

template<class G>
void benchGPU(
  const std::vector<double> x,
  G gpuFunctor
) {  
  std::vector<double> y = applyGpuOp1(x, gpuFunctor);
}

/*
BENCHMARK(CpuNaturalLogarithm, Random_0100, 10, 100) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, CpuLog());
}

BENCHMARK(LibNaturalLogarithm, Random_0100, 10, 100) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, LibLog());
}

BENCHMARK(GpuNaturalLogarithm, Random_0100, 10, 100) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyGpuOp1(x, GpuLog());
}

BENCHMARK(CpuExponential, Random_100100, 10, 100) {
  std::vector<double> x = randomFill(-100, 100, N);
  std::vector<double> y = applyCpuOp1(x, CpuExp());
}

BENCHMARK(LibExponential, Random_100100, 10, 100) {
  std::vector<double> x = randomFill(-100, 100, N);
  std::vector<double> y = applyCpuOp1(x, LibExp());
}

BENCHMARK(GpuExponential, Random_100100, 10, 100) {
  std::vector<double> x = randomFill(-100, 100, N);
  std::vector<double> y = applyGpuOp1(x, GpuExp());
}

BENCHMARK(GpuExponential10, Random_100100, 10, 100) {
  std::vector<double> x = randomFill(-100, 100, 10*N);
  std::vector<double> y = applyGpuOp1(x, GpuExp());
}
*/

/*
BENCHMARK(GpuExponentialBench, Random_100100, 10, 100) {
  std::vector<double> x = randomFill(-100, 100, N);
  std::vector<double> y = applyGpuBenchOp1(x, GpuExp());
}

BENCHMARK(GpuExponential10Bench, Random_100100, 10, 100) {
  std::vector<double> x = randomFill(-100, 100, 10*N);
  std::vector<double> y = applyGpuBenchOp1(x, GpuExp());
}
*/
/*
int* Random_0_100::pA = pA;
int* Random_0_100::pB = pB;
int* Random_0_100::pN = pN;
*/

BENCHMARK_F(Random_0_1000 , GpuExponentialBench, 10, 100) {
  std::vector<double> y = applyGpuBenchOp1(
  	pDevX, 
  	pDevY,
  	N,
  	GpuExp()
  );
}

BENCHMARK_F(Random_0_1000, GpuExponentialBench2, 10, 100) {
  std::vector<double> y = applyGpuBenchOp1(
  	pDevX, 
  	pDevY,
  	N,
  	CudaExp()
  );
}

int main() {
  hayai::ConsoleOutputter consoleOutputter;
 
  hayai::Benchmarker::AddOutputter(consoleOutputter);
  hayai::Benchmarker::RunAllTests();
  return 0;
}
