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
