#include "hell_log.h"
#include <stdio.h>
#include <limits>
#include <bitset>
#include <iostream>

union uinteger {
  int i;
  unsigned long long u;
};

union ufloat {
  float f;
  unsigned int li;
  unsigned int u;
};

double cpu_log(double x) {
  return uber_log(x);
}

// FLOAT: 32 bits
// INT: 32 bits
// UINT: 32 bits

short float2hiint(float f) {
  ufloat uf;
  uf.f = f;
  /*
  printf("[D2H] d %llu\n", ud.li);
  print_ull(ud.li);
  */
  unsigned int high = (ud.u >> 16);
  /*
  printf("[D2H] h %llu\n", high);
  print_ull(high);
  printf("\n");
  */
  return (short) high;
}

// TESTED AND WORKING
short float2loint(float f) {
  ufloat uf;
  uf.f = f;
  unsigned int mask = 0x0000FFFF;
  unsigned int low = ud.f & mask;
  
  /*
  printf("[D2L] d %llu\n", ud.li);
  print_ull(ud.li);
  
  printf("[D2L] mask %llu\n", mask);
  print_ull(mask);
  
  printf("[D2L] low %llu\n", low);
  print_ull(low);
  printf("\n");
  */
  
  return (short) low;
}


void print_i(int a) {
  uinteger ui;
  ui.i = a;
  
  std::bitset<32> bs(a);
  std::cout << bs << std::endl;
}

float hiloint2float(short high, short low) {

  // MY HACK
  unsigned int h = high;
  unsigned int l = low;
  
  unsigned int uber = h;
  uber <<= 16;
  uber |= l;
  
  ufloat f;
  u.u = uber;  
  return u.f;  
}

void getExpoMant(double a, double* EM) {
  const double two_to_54 = 18014398509481984.0;
  
  short ihi = float2hiint(a);
  short ilo = float2loint(a);
  int e = -127;
  
  /*
  printf("a %f\n", a);
  printf("ihi1 %i\n", ihi);
  printf("ilo1 %i\n", ilo);
  */
  
  /* normalize denormals */
  if ((unsigned) ihi < (unsigned) 0x00100000) {
    a = a*two_to_54;
    e -= 54;
    ihi = double2hiint(a);
    ilo = double2loint(a);
  }
  
  /*
  printf("ihi2 %i\n", ihi);
  printf("ilo2 %i\n", ilo);
  */
  
  /* 
   * a = m * 2^e
   * m <= sqrt(2): log2(a) = log2(m) + e
   * m > sqrt(2): log2(a) = log2(m/2) + (e+1)
   */
  e += (ihi >> 20);
  ihi = (ihi & 0x800fffff) | 0x3ff00000;
  
  /*
  printf("ihi2b %i\n", ihi);
  printf("ilo2b %i\n", ilo);
  */
  
  double m = hiloint2double(ihi, ilo);
  //printf("m1 %f\n", m);
  
  if ((unsigned) ihi > (unsigned) 0x3ff6a09e) {
    m = 0.5*m;
    e = e + 1;
  }
  
  /*
  printf("e %i\n", e);
  printf("m2 %f\n", m);
  printf("ihi3 %i\n", ihi);
  printf("ilo3 %i\n", ilo);
  */
  
  EM[0] = (double) e;
  EM[1] = m;
}

double uber_log(double a) {
  const float ln2_hi = 6.9314718055994529e-1;
  const float ln2_lo = 2.3190468138462996e-17;
  const float infinity = std::numeric_limits<float>::infinity();
  const float notanumber = std::numeric_limits<float>::quiet_NaN();
  
  double q;
  
  double EM[2];
  getExpoMant(a, EM);
  double e = EM[0];
  double m = EM[1];
  
  if ((a > 0.0) && (a < infinity)) {    
    /* 
     * log((1+m)/(1-m)) = 2*atanh(m). 
     * log(m) = 2*atanh((m-1)/(m+1)) 
     */
    double f = m - 1.0;
    double g = m + 1.0;
    g = 1.0/g;
    double u = f * g;
    u = u + u;  
    
    /*
    printf("f %f\n", f);
    printf("m %f\n", m);
    printf("g %f\n", g);
    printf("u %f\n", u);
    */
    
    /* 
     * u = 2.0 * (m - 1.0) / (m + 1.0) 
     */
    double v = u*u;
    q = 6.7261411553826339E-2/65536.0;
    q = q*v + 6.6133829643643394E-2/16384.0;
    q = q*v + 7.6940931149150890E-2/4096.0;
    q = q*v + 9.0908745692137444E-2/1024.0;
    q = q*v + 1.1111111499059706E-1/256.0;
    q = q*v + 1.4285714283305975E-1/64.0;
    q = q*v + 2.0000000000007223E-1/16.0;
    q = q*v + 3.3333333333333326E-1/4.0;
    
    double tmp = 2.0*(f - u);
    tmp = -u*f + tmp; // tmp = remainder of division
    double ulo = g*tmp; // less significant quotient bits
    /* 
     * u + ulo = 2.0 * (m - 1.0) / (m + 1.0) 
     * to more than double precision 
     */
    q = q * v;
    /* 
     * log_hi + log_lo = log(m) 
     * to more than double precision 
     */ 
     
    double log_hi = u;
    double log_lo = q*u + ulo;
    
    /*
    printf("log_hi %f\n", log_hi);
    printf("log_lo %f\n", log_lo);
    */
    
    /* 
     * log_hi + log_lo = log(m) + e*log(2) = log(a) 
     * to more than double precision
     */
    q = e*ln2_hi + log_hi;
    //printf("q %f\n", q);
    
    tmp = -e*ln2_hi + q;
    tmp = tmp - log_hi;
    log_hi = q;
    log_lo = log_lo - tmp;
    log_lo = e*ln2_lo + log_lo;
        
    
    
    q = log_hi + log_lo;
  } else if (a != a) {
    //printf("case b\n");
    q = a + a;
  } 
  /* log(0) = -INF */
  else if (a == 0) {
    //printf("case c\n");
    q = -infinity;
  } 
  /* log(INF) = INF */
  else if (a == infinity) { 
    //printf("case d\n");
    q = a;
  } 
  /* 
   * log(x) is undefined for x < 0.0, 
   * return INDEFINITE 
   */
  else {
    //printf("case e\n");
    q = notanumber;
  }
  /*
  printf("returning %f\n", q);
  printf("===============================\n");
  */
  return q;
}

