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

// VerifyTol and VerifyEq float functions.

void verifyTolFloat(
  const std::vector<float>& expectedVals,
  const std::vector<float>& values,
  const std::vector<float>& x,
  const std::vector<float>& y,
  const float tol
);

void verifyTolFloat(
  const std::vector<float>& expectedVals,
  const std::vector<float>& values,
  const std::vector<float>& x,
  const float tol
);

void verifyTolFloat(
  const std::vector<float>& expectedVals,
  const std::vector<float>& values,
  const float tol
);

void verifyEqFloat(
  const std::vector<float>& expectedVals,
  const std::vector<float>& values,
  const std::vector<float>& x,
  const std::vector<float>& y
);

void verifyEqFloat(
  const std::vector<float>& expectedVals,
  const std::vector<float>& values,
  const std::vector<float>& x
);

void verifyEqFloat(
  const std::vector<float>& expectedVals,
  const std::vector<float>& values
);

