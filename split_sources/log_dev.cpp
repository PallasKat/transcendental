#include <cmath>
#include <stdio.h>

union SI {
  unsigned int ii;
  unsigned long int li;
  double dd;
};

void printDd(double d) {
  SI num;
  num.dd = d;
  printf("[CPU] %f %lu\n", num.dd, num.li);
}

double log_dev(double x) {     
  double ln2 = 6.93147180559945290e-001;
  // x = m * 2^e
  int tmpExp;
  double m = std::frexp(x, &tmpExp);  
  double e = static_cast<double>(tmpExp);
  
//  printDd(m);
//  printDd(e);
  
  // as m < 1 => 0.5 * log((1+m)/(1-m)) = atanh(m) <=>
  // and with with substitution y = (1+x)/(1-x)
  // log(m) = 2 * atanh((m-1)/(m+1))
  double f = m - 1.0;
  double g = m + 1.0;
  double u = f/g;
  
//  printDd(f);
//  printDd(g);
//  printDd(u);
  
  // compute atanh(u) = atanh((m-1)/(m+1)) with taylor:
  // x + x^3/3 + x^5/5 + ... (sum of x^i/i, i=1,3,5,9,...)
  const double uu = u*u;
  double y = u;
  for (int k = 3; k < 10; k = k + 2) {
    double d = static_cast<double>(k);
    u = uu*u;
    y = y + (u/d);
        
//    printf("[CPU] Loop %i\n", k);
//    printDd(u);
//    printDd(y);
  }
  
  // log(a) = log(m) + ln2 * e
  double w = ln2*e;
  double z = y + w;
//  printDd(w);
//  printDd(z);
  return z;  
}

void test_dev(double* a, double* b, int n) {
  for (int i = 0; i < n; i++) {
    b[i] = log_dev(a[i]);
  }
}

