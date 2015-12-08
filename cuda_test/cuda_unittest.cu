// google test
#include "gtest/gtest.h"
// printf, scanf, puts, NULL
#include <stdio.h>
// srand, rand
#include <stdlib.h>
// test helpers
#include "test_tools.h"
// ptr
#include <memory>
// gpulog, cpulog, liblog
#include "log_func.h"
// to use the functors on vectors
#include "functors.h"

const int N = 10;
const double TOL = 10E-10;

// =============================================================================
// VERIFICATION HELPERS
// =============================================================================

void verifyTol(
  const std::vector<double>& expectedValues,
  const std::vector<double>& values,
  const double tol
) {
  if (expectedValues.size() != values.size()) {
    FAIL() << "The vectors have different lengths.";
  }

  for (auto i = 0; i < values.size(); i++) {
    double err = relativeError(values[i], expectedValues[i]);
    if (isnan(expectedValues[i]) && isnan(values[i])) {
      SUCCEED();
    } else if (isinf(expectedValues[i]) && isinf(values[i])) {
      if (expectedValues[i] < 0 && values[i] < 0) {
        SUCCEED();
      } else {
        FAIL() << "One value is inf and the other is -inf.";
      }
    } else {
      ASSERT_LE(err, tol);
    }
  }
}

void verifyEq(
  const std::vector<double>& expectedValues,
  const std::vector<double>& values
) {
  if (expectedValues.size() != values.size()) {
    FAIL() << "The vectors have different lengths.";
  }

  for (int i = 0; i < values.size(); i++) {
    if (isnan(expectedValues[i]) && isnan(values[i])) {
      SUCCEED();
    } else {
      ASSERT_EQ(expectedValues[i], values[i]);
    }
  }
}

// =============================================================================
// TESTING THE LOGARITHM FUNCTIONS
// =============================================================================

TEST(LogCPUTest, PositiveValues) {
  std::unique_ptr<std::vector<double>> x = randomFill(0, 1, N);
  std::unique_ptr<std::vector<double>> expectedValues = applyCpuOp1(*x.get(), CpuLog());
  std::unique_ptr<std::vector<double>> crClimValues = applyCpuOp1(*x.get(), LibLog());
  verifyTol(*expectedValues.get(), *crClimValues.get(), TOL);
}

TEST(LogGPUTest, PositiveValues) {
  std::unique_ptr<std::vector<double>> x = randomFill(0, 1, N);
  std::vector<double> xx = *x.get();
  //std::unique_ptr<std::vector<double>> expectedValues = applyGpuOp1(xx, GpuLog());
  std::vector<double> expectedValues = applyGpuOp1(xx, GpuLog());
  std::unique_ptr<std::vector<double>> crClimValues = applyCpuOp1(xx, LibLog());
  verifyTol(expectedValues, *crClimValues.get(), TOL);
}

// =============================================================================
// ENTRY POINT
// =============================================================================
int main(int argc, char **argv) {
  //srand(time(NULL));
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
