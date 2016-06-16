#include <vector>

void verifyTol(
  const std::vector<double>& expectedVals,
  const std::vector<double>& values,
  const std::vector<double>& x,
  const std::vector<double>& y,
  const double tol
);

void verifyTol(
  const std::vector<double>& expectedVals,
  const std::vector<double>& values,
  const std::vector<double>& x,
  const double tol
);

void verifyTol(
  const std::vector<double>& expectedVals,
  const std::vector<double>& values,
  const double tol
);

void verifyEq(
  const std::vector<double>& expectedVals,
  const std::vector<double>& values,
  const std::vector<double>& x,
  const std::vector<double>& y
);

void verifyEq(
  const std::vector<double>& expectedVals,
  const std::vector<double>& values,
  const std::vector<double>& x
);

void verifyEq(
  const std::vector<double>& expectedVals,
  const std::vector<double>& values
);

/*
template<class G, class R>
void testCpuTolOn(
  const std::vector<double> x,
  const double tol,
  G cpuFunctor,
  R refFunctor
);

template<class G, class R>
void testGpuTolOn(
  const std::vector<double> x,
  const double tol,
  G gpuFunctor,
  R refFunctor
);

void testCpuLogTolOn(
  const std::vector<double> x,
  const double tol
);

void testGpuLogTolOn(
  const std::vector<double> x,
  const double tol
);
*/