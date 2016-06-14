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

std::vector<double> seqTo(int n) {
  std::vector<double> u(n);
  for (int i = 0; i < n; i++) {
    u[i] = 1.0*i;
  }

  printf("====================\n");
  for (auto it = u.begin(); it != u.end(); ++it) {
    printf("%f\n", *it);
  }
  printf("====================\n");
  return u;
}

int main()
{
  //auto p = std::unique_ptr<vector<int>>(new std::std::vector<int>(10));
/*  srand(time(NULL));
  auto x = randomFill(0, 1, 10);
  for (auto it = (*x).begin(); it != (*x).end(); ++it) {
    printf("%f\n", *it);
  }
*/
  auto v = seqTo(4);
  printf("====================\n");
  for (auto it = v.begin(); it != v.end(); ++it) {
    printf("%f\n", *it);
  }
  printf("====================\n");
}
