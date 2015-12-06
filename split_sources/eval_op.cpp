#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <fstream> 
#include <iomanip>
#include <limits>
#include <bitset>
//#include <cmath>

//#include "mch_math.h"
#include "hell_log.h"


// Fill the array ary with random values in [a,b).
void randomFill(double* ary, int a, int b, int n) {    
  for (int i = 0; i < n; i++) {
    //ary[i] = a + (std::rand() % (b-a+1));
    ary[i] = (b - a) * ((double) rand() / (double) RAND_MAX ) + a;
  }  
}

// Fill the array ary with random values in [0,1).
void randomFill(double* ary, int n) {
  return randomFill(ary, 0, 1, n);
}

void zeroFill(double* ary, int n) {
  for (int i = 0; i < n; i++) {
    ary[i] = 0.0;
  } 
}

int main(void) {
  const unsigned N = 1000000;
//  const unsigned N = 1;    
  double X[N];
  double Y[N];
  double Z[N];
  //randomFill(X, 0, 1, N);
  //zeroFill(Y, N);

  randomFill(X, 0, 10, N);
  zeroFill(Y, N);
  zeroFill(Z, N);

  std::ofstream f;
  f.open("log.txt");
  
  /*
  double d = 123.456;
  print_double(d);
  int hi = double2hiint(d);
  int lo = double2loint(d);
  print_int(hi);
  print_int(lo);
  */
  
  for (int j = 0; j < N; j++) {
    Y[j] = cpu_log(X[j]);
    Z[j] = log(X[j]);  
              
    // f << X[j] << " " << Y[j] << std::endl;

    f << 
        std::setprecision(std::numeric_limits<long double>::digits10 + 1) 
      << X[j] 
      << " " <<
        std::setprecision(std::numeric_limits<long double>::digits10 + 1)
      << Y[j] 
      << " " <<
        std::setprecision(std::numeric_limits<long double>::digits10 + 1)  
      << Z[j]
      << std::endl;
  }

  f.close();
}

