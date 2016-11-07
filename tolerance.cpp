#include "tolerance.h"

// google test
#include "gtest/gtest.h"
// test helpers
#include "test_tools.h"

void verifyTol(
  const std::vector<double>& expectedVals,
  const std::vector<double>& values,
  const std::vector<double>& x,
  const std::vector<double>& y,
  const double tol
) {
	double err;

  if (expectedVals.size() != values.size()) {
    FAIL() << "The vectors have different lengths.";
  }

  for (auto i = 0; i < values.size(); i++) {
  	if (tol == 0.0) {
    	err = absoluteError(values[i], expectedVals[i]);
    } else {
    	err = relativeError(values[i], expectedVals[i]);
    }
    if (isnan(expectedVals[i]) &&  isnan(values[i])) {
      SUCCEED();
    } else if (isinf(expectedVals[i]) && isinf(values[i])) {
      ASSERT_EQ(expectedVals[i], values[i]);
    } else {
      if (!x.empty() && !y.empty()) {
        ASSERT_LE(err, tol) << "with x, y: " << x[i] << ", " << y[i]
                            << " and exp, val: "
                            << expectedVals[i] << ", " << values[i];
      } else if (!x.empty()) {
        ASSERT_LE(err, tol) << "with x: " << x[i]
                            << " and exp, val: "
                            << expectedVals[i] << ", " << values[i];
      } else {
        ASSERT_LE(err, tol) << "with: " << expectedVals[i] << ", " << values[i];
      }
    }
  }
}

//VerifyTol function with Float

void verifyTolFloat(
  const std::vector<float>& expectedVals,
  const std::vector<float>& values,
  const std::vector<float>& x,
  const std::vector<float>& y,
  const float tol
) {
	float err;

  if (expectedVals.size() != values.size()) {
    FAIL() << "The vectors have different lengths.";
  }

  for (auto i = 0; i < values.size(); i++) {
  	if (tol == 0.0) {
    	err = absoluteError(values[i], expectedVals[i]);
    } else {
    	err = relativeError(values[i], expectedVals[i]);
    }
    if (isnan(expectedVals[i]) == isnan(values[i])) {
      SUCCEED();
    } else if (isinf(expectedVals[i]) && isinf(values[i])) {
      ASSERT_EQ(expectedVals[i], values[i]);
    } else {
      if (!x.empty() && !y.empty()) {
        ASSERT_LE(err, tol) << "with x, y: " << x[i] << ", " << y[i]
                            << " and exp, val: "
                            << expectedVals[i] << ", " << values[i];
      } else if (!x.empty()) {
        ASSERT_LE(err, tol) << "with x: " << x[i]
                            << " and exp, val: "
                            << expectedVals[i] << ", " << values[i];
      } else {
        ASSERT_LE(err, tol) << "with: " << expectedVals[i] << ", " << values[i];
      }
    }
  }
}

void verifyTol(
  const std::vector<double>& expectedVals,
  const std::vector<double>& values,
  const std::vector<double>& x,
  const double tol
) {
  std::vector<double> emptiness;
  verifyTol(expectedVals, values, x, emptiness, tol);
}

void verifyTol(
  const std::vector<double>& expectedVals,
  const std::vector<double>& values,
  const double tol
) {
  std::vector<double> emptiness;
  verifyTol(expectedVals, values, emptiness, emptiness, tol);
}

void verifyEq(
  const std::vector<double>& expectedVals,
  const std::vector<double>& values,
  const std::vector<double>& x,
  const std::vector<double>& y
) {
  verifyTol(expectedVals, values, x, y, 0.0);
}

void verifyEq(
  const std::vector<double>& expectedVals,
  const std::vector<double>& values,
  const std::vector<double>& x
) {
  std::vector<double> emptiness;
  verifyEq(expectedVals, values, x, emptiness);
}
void verifyEq(
  const std::vector<double>& expectedVals,
  const std::vector<double>& values
) {
  std::vector<double> emptiness;
  verifyEq(expectedVals, values, emptiness, emptiness);
}

//VerifyTol and verifyEq FLOATING POINTS.

void verifyTolFloat(
  const std::vector<float>& expectedVals,
  const std::vector<float>& values,
  const std::vector<float>& x,
  const float tol
) {
  std::vector<float> emptiness;
  verifyTolFloat(expectedVals, values, x, emptiness, tol);
}

void verifyTolFloat(
  const std::vector<float>& expectedVals,
  const std::vector<float>& values,
  const float tol
) {
  std::vector<float> emptiness;
  verifyTolFloat(expectedVals, values, emptiness, emptiness, tol);
}

void verifyEqFloat(
  const std::vector<float>& expectedVals,
  const std::vector<float>& values,
  const std::vector<float>& x,
  const std::vector<float>& y
) {
  verifyTolFloat(expectedVals, values, x, y, 0.0);
}

void verifyEqFloat(
  const std::vector<float>& expectedVals,
  const std::vector<float>& values,
  const std::vector<float>& x
) {
  std::vector<float> emptiness;
  verifyEqFloat(expectedVals, values, x, emptiness);
}

void verifyEqFloat(
  const std::vector<float>& expectedVals,
  const std::vector<float>& values
) {
  std::vector<float> emptiness;
  verifyEqFloat(expectedVals, values, emptiness, emptiness);
}



