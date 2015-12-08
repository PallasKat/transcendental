// vector ? not in std ??
#include <vector>
// printf, scanf, puts, NULL
#include <stdio.h>
// srand, rand
#include <stdlib.h>
// test helpers
#include "test_tools.h"
// ptr
#include <memory>

int main()
{
  //auto p = std::unique_ptr<vector<int>>(new std::std::vector<int>(10));
  srand(time(NULL));
  auto x = randomFill(0, 1, 10);
  for(auto it = (*x).begin(); it != (*x).end(); ++it) {
    printf("%f\n", *it);
  }
}
