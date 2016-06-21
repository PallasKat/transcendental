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

// Parameters for the benchmarks
const size_t N = 10*1000;
const int nRep = 10;
const int nIter = 100;

// ==========================================================
//  Bencharks on the CPU (vs. CMath)
// ==========================================================
BENCHMARK(CpuNaturalLogarithm, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, CpuLog());
}

BENCHMARK(LibNaturalLogarithm, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, LibLog());
}

BENCHMARK(CpuExponential, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(-100, 100, N);
  std::vector<double> y = applyCpuOp1(x, CpuExp());
}

BENCHMARK(LibExponential, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(-100, 100, N);
  std::vector<double> y = applyCpuOp1(x, LibExp());
}

BENCHMARK(CpuPower, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(-100, 100, N);
  std::vector<double> y = randomFill(0, 100, N);
  std::vector<double> z = applyCpuOp2(x, y, CpuPow());
}

BENCHMARK(LibPower, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(-100, 100, N);
  std::vector<double> y = randomFill(0, 100, N);
  std::vector<double> z = applyCpuOp2(x, y, LibPow());
}

// ==========================================================
//  Bencharks on the GPU (vs. CUDAMath)
// ==========================================================
BENCHMARK_F(Random_1000 , GpuExponentialBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
  	pDevX,
  	pDevY,
  	N,
  	GpuExp()
  );
}

BENCHMARK_F(Random_1000, CudaExponentialBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
  	pDevX, 
  	pDevY,
  	N,
  	CudaExp()
  );
}

BENCHMARK_F(Random_1000 , GpuNaturalLogarithmBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
  	pDevX, 
  	pDevY,
  	N,
  	GpuLog()
  );
}

BENCHMARK_F(Random_1000, CudaNaturalLogarithmBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
  	pDevX, 
  	pDevY,
  	N,
  	CudaLog()
  );
}

BENCHMARK_F(Random2_1000 , GpuNaturalLogarithmBench, nRep, nIter) {
  std::vector<double> z = applyGpuBenchOp2(
  	pDevX, 
  	pDevY,
  	pDevZ,
  	N,
  	GpuPow()
  );
}

BENCHMARK_F(Random2_1000, CudaNaturalLogarithmBench, nRep, nIter) {
  std::vector<double> z = applyGpuBenchOp2(
  	pDevX, 
  	pDevY,
  	pDevZ,
  	N,
  	CudaPow()
  );
}

int main() {
  hayai::ConsoleOutputter consoleOutputter;
 
  hayai::Benchmarker::AddOutputter(consoleOutputter);
  hayai::Benchmarker::RunAllTests();
  return 0;
}
