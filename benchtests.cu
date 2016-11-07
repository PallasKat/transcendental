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
// DOUBLES -  Bencharks on the CPU (vs. CMath)
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

BENCHMARK(CpuSin, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, CpuSin());
}

BENCHMARK(LibSin, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, LibSin());
}

BENCHMARK(CpuCos, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, CpuCos());
}

BENCHMARK(LibCos, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, LibCos());
}

BENCHMARK(CpuTan, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, CpuTan());
}

BENCHMARK(LibTan, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, LibTan());
}

BENCHMARK(CpuAsin, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, CpuAsin());
}

BENCHMARK(LibAsin, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, LibAsin());
}

BENCHMARK(CpuAcos, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, CpuAcos());
}

BENCHMARK(LibAcos, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, LibAcos());
}

BENCHMARK(CpuAtan, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, CpuAtan());
}

BENCHMARK(LibAtan, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, LibAtan());
}
BENCHMARK(CpuSinh, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, CpuSinh());
}

BENCHMARK(LibSinh, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, LibSinh());
}

BENCHMARK(CpuCosh, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, CpuCosh());
}

BENCHMARK(LibCosh, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, LibCosh());
}

BENCHMARK(CpuTanh, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, CpuTanh());
}

BENCHMARK(LibTanh, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, LibTanh());
}

BENCHMARK(CpuAsinh, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, CpuAsinh());
}

BENCHMARK(LibAsinh, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, LibAsinh());
}


BENCHMARK(CpuAcosh, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, CpuAcosh());
}

BENCHMARK(LibAcosh, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, LibAcosh());
}

BENCHMARK(CpuAtanh, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, CpuAtanh());
}

BENCHMARK(LibAtanh, Random_1000, nRep, nIter) {
  std::vector<double> x = randomFill(0, 100, N);
  std::vector<double> y = applyCpuOp1(x, LibAtanh());
}

// ==========================================================
//  Bencharks on the CPU (vs. CMath) FLOATING POINTS
// ==========================================================

/*
BENCHMARK(CpuNaturalLogarithmFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, CpuLog());
}

BENCHMARK(LibNaturalLogarithmFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, LibLog());
}

BENCHMARK(CpuExponentialFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(-100, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, CpuExp());
}

BENCHMARK(LibExponentialFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(-100, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, LibExp());
}

BENCHMARK(CpuPowerFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(-100, 100, N);
  std::vector<float> y = randomFillFloat(0, 100, N);
  std::vector<float> z = applyCpuOp2Float(x, y, CpuPow());
}

BENCHMARK(LibPowerFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(-100, 100, N);
  std::vector<float> y = randomFillFloat(0, 100, N);
  std::vector<float> z = applyCpuOp2Float(x, y, LibPow());
}

BENCHMARK(CpuSinFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, CpuSin());
}

BENCHMARK(LibSinFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, LibSin());
}

BENCHMARK(CpuCosFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, CpuCos());
}

BENCHMARK(LibCosFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, LibCos());
}

BENCHMARK(CpuTanFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, CpuTan());
}

BENCHMARK(LibTanFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, LibTan());
}


BENCHMARK(CpuAsinFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, CpuAsin());
}

BENCHMARK(LibAsinFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, LibAsin());
}

BENCHMARK(CpuAcosFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, CpuAcos());
}

BENCHMARK(LibAcosFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, LibAcos());
}

BENCHMARK(CpuAtanFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, CpuAtan());
}

BENCHMARK(LibAtanFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, LibAtan());
}
BENCHMARK(CpuSinhFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, CpuSinh());
}

BENCHMARK(LibSinhFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, LibSinh());
}

BENCHMARK(CpuCoshFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, CpuCosh());
}

BENCHMARK(LibCoshFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, LibCosh());
}

BENCHMARK(CpuTanhFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, CpuTanh());
}

BENCHMARK(LibTanhFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, LibTanh());
}

BENCHMARK(CpuAsinhFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, CpuAsinh());
}

BENCHMARK(LibAsinhFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, LibAsinh());
}


BENCHMARK(CpuAcoshFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, CpuAcosh());
}

BENCHMARK(LibAcoshFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, LibAcosh());
}

BENCHMARK(CpuAtanhFloat, Random_1000, nRep, nIter) {
  std::vector<float> x = randomFillFloat(0, 100, N);
  std::vector<float> y = applyCpuOp1Float(x, CpuAtanh());
}
*/

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

BENCHMARK_F(Random2_1000 , GpuPowerBench, nRep, nIter) {
  std::vector<double> z = applyGpuBenchOp2(
  	pDevX, 
  	pDevY,
  	pDevZ,
  	N,
  	GpuPow()
  );
}

BENCHMARK_F(Random2_1000, CudaPowerBench, nRep, nIter) {
  std::vector<double> z = applyGpuBenchOp2(
  	pDevX, 
  	pDevY,
  	pDevZ,
  	N,
  	CudaPow()
  );
}

BENCHMARK_F(Random_1000 , GpuSinBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
  	pDevX,
  	pDevY,
  	N,
  	GpuSin()
  );
}

BENCHMARK_F(Random_1000, CudaSinBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
  	pDevX, 
  	pDevY,
  	N,
  	CudaSin()
  );
}

BENCHMARK_F(Random_1000 , GpuCosBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        GpuCos()
  );
}

BENCHMARK_F(Random_1000, CudaCosBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        CudaCos()
  );
}
BENCHMARK_F(Random_1000 , GpuTanBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        GpuTan()
  );
} 
        
BENCHMARK_F(Random_1000, CudaTanBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        CudaTan()
  );
} 

BENCHMARK_F(Random_1000 , GpuAsinBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        GpuAsin()
  );
} 
        
BENCHMARK_F(Random_1000, CudaAsinBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        CudaAsin()
  );
}
 
BENCHMARK_F(Random_1000 , GpuAcosBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        GpuAcos()
  );
} 
        
BENCHMARK_F(Random_1000, CudaAcosBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        CudaAcos()
  );
} 
BENCHMARK_F(Random_1000 , GpuAtanBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        GpuAtan()
  );
} 
        
BENCHMARK_F(Random_1000, CudaAtanBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        CudaAtan()
  );
} 

BENCHMARK_F(Random_1000 , GpuSinhBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        GpuSinh()
  );
} 
        
BENCHMARK_F(Random_1000, CudaSinhBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        CudaSinh()
  );
} 

BENCHMARK_F(Random_1000 , GpuCoshBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        GpuCosh()
  );
} 
        
BENCHMARK_F(Random_1000, CudaCoshBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        CudaCosh()
  );
} 

BENCHMARK_F(Random_1000 , GpuTanhBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        GpuTanh()
  );
} 
        
BENCHMARK_F(Random_1000, CudaTanhBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        CudaTanh()
  );
} 

BENCHMARK_F(Random_1000 , GpuAsinhBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        GpuAsinh()
  );
} 
        
BENCHMARK_F(Random_1000, CudaAsinhBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        CudaAsinh()
  );
} 

BENCHMARK_F(Random_1000 , GpuAcoshBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        GpuAcosh()
  );
} 
        
BENCHMARK_F(Random_1000, CudaAcoshBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        CudaAcosh()
  );
} 

BENCHMARK_F(Random_1000 , GpuAtanhBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        GpuAtanh()
  );
} 
        
BENCHMARK_F(Random_1000, CudaAtanhBench, nRep, nIter) {
  std::vector<double> y = applyGpuBenchOp1(
        pDevX,
        pDevY,
        N,
        CudaAtanh()
  );
} 

// ==========================================================
//  Bencharks on the GPU (vs. CUDAMath) FLOATING POINTS
// ==========================================================

/*
BENCHMARK_F(Random_1000Float , GpuNaturalLogarithmBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        GpuLog()
  );
}

BENCHMARK_F(Random_1000Float, CudaNaturalLogarithmBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        CudaLog()
  );
}

BENCHMARK_F(Random_1000Float, GpuExponentialBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        GpuExp()
  );
}

BENCHMARK_F(Random_1000Float, CudaExponentialBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        CudaExp()
  );
}
*/

/*
BENCHMARK_F(Random2_1000Float, GpuNaturalLogarithmBenchFloat, nRep, nIter) {
  std::vector<float> z = applyGpuBenchOp2Float(
        pDevX,
        pDevY,
        pDevZ,
        N,
        GpuPow()
  );
}

BENCHMARK_F(Random2_1000Float, CudaNaturalLogarithmBenchFloat, nRep, nIter) {
  std::vector<float> z = applyGpuBenchOp2Float(
        pDevX,
        pDevY,
        pDevZ,
        N,
        CudaPow()
  );
}
*/

/*
BENCHMARK_F(Random_1000Float, GpuSinBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        GpuSin()
  );
}

BENCHMARK_F(Random_1000Float, CudaSinBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        CudaSin()
  );
}

BENCHMARK_F(Random_1000Float, GpuCosBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        GpuCos()
  );
}

BENCHMARK_F(Random_1000Float, CudaCosBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        CudaCos()
  );
}
BENCHMARK_F(Random_1000Float, GpuTanBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        GpuTan()
  );
}

BENCHMARK_F(Random_1000Float, CudaTanBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        CudaTan()
  );
}

BENCHMARK_F(Random_1000Float, GpuAsinBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        GpuAsin()
  );
} 
        
BENCHMARK_F(Random_1000Float, CudaAsinBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        CudaAsin()
  );
}

BENCHMARK_F(Random_1000Float, GpuAcosBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        GpuAcos()
  );
}

BENCHMARK_F(Random_1000Float, CudaAcosBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        CudaAcos()
  );
}
BENCHMARK_F(Random_1000Float, GpuAtanBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        GpuAtan()
  );
}

BENCHMARK_F(Random_1000Float, CudaAtanBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        CudaAtan()
  );
}

BENCHMARK_F(Random_1000Float, GpuSinhBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        GpuSinh()
  );
}

BENCHMARK_F(Random_1000Float, CudaSinhBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        CudaSinh()
  );
}

BENCHMARK_F(Random_1000Float, GpuCoshBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        GpuCosh()
  );
}

BENCHMARK_F(Random_1000Float, CudaCoshBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        CudaCosh()
  );
}

BENCHMARK_F(Random_1000Float, GpuTanhBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        GpuTanh()
  );
}

BENCHMARK_F(Random_1000Float, CudaTanhBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        CudaTanh()
  );
}

BENCHMARK_F(Random_1000Float, GpuAsinhBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        GpuAsinh()
  );
}

BENCHMARK_F(Random_1000Float, CudaAsinhBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        CudaAsinh()
  );
}

BENCHMARK_F(Random_1000Float, GpuAcoshBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        GpuAcosh()
  );
}

BENCHMARK_F(Random_1000Float, CudaAcoshBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        CudaAcosh()
  );
}

BENCHMARK_F(Random_1000Float, GpuAtanhBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        GpuAtanh()
  );
}

BENCHMARK_F(Random_1000Float, CudaAtanhBenchFloat, nRep, nIter) {
  std::vector<float> y = applyGpuBenchOp1Float(
        pDevX,
        pDevY,
        N,
        CudaAtanh()
  );
}
*/

int main() {
  hayai::ConsoleOutputter consoleOutputter;
 
  hayai::Benchmarker::AddOutputter(consoleOutputter);
  hayai::Benchmarker::RunAllTests();
  return 0;
}
