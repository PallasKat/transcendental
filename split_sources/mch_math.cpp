#include <math.h> 

#include "mch_math.h"

const bool VERBOSE = false;

void printDd(double d) {
  udouble num;
  num.d = d;
  printf("[CPU] %f %lu\n", num.d, num.li);
}

void printIi(int i) {
  printf("[CPU] %i\n", i);
}

void print_int(int a) {
  uinteger ui;
  ui.i = a;

  std::bitset<32> bs(ui.u);
  std::cout << bs << std::endl;
}

void print_double(double a) {
  udouble ud;
  ud.d = a;  
  print_ull(ud.u);
}

void print_ull(unsigned long long a) {
  std::bitset<64> bs(a);
  std::cout << bs << std::endl;
}

double hiloint2double(int high, int low) {
  unsigned long long uberhigh = ((unsigned long long) high) << 32;
  unsigned long long uberlow = low;
  unsigned long long uber = uberhigh | uberlow;
  
  udouble u;
  u.u = uber;  
  return u.d;
}

int double2hiint(double d) {
  udouble ud;
  ud.d = d;
  unsigned long long high = (ud.u >> 32);

  return (int) high;
}

int double2loint(double d) {
  udouble ud;
  ud.d = d;
  unsigned long long mask = 0x00000000 | 0xFFFFFFFF;
  unsigned long long low = ud.u & mask;
  return (int) low;
}

int mch_isnand(double a) {
  return a != a;
//  return !(mch_fabs(a) <= MCH_INF);
}

double mch_rint(double a) {
  if (a > 0) {
    return (int) (a+0.5);
  }

  if (a < 0) {
    return (int) (a-0.5);
  }
  
  return 0;
}

int sabs(int i) {
  if (MCH_INT_MIN == i) {
    return MCH_INT_MAX;
  } else {
    return i < 0 ? -i : i;
  }
}

int mch_fabs(double i) {
  return i < 0 ? -i : i;
}

double mch_exp(double a) {
  double t;
  int i = double2hiint(a);
  // We only check if we are in a specific range [-a,b] to compute 
  // the exp
  if (((unsigned) i < (unsigned) 0x40862e43) || ((int) i < (int) 0xC0874911)) {
    t = mch_exp_kernel(a, 0);
  } 
  // Otherwise the result is a very small value, then returning 0 or
  // a very large value then returning inf
  else {
    t = (i < 0) ? MCH_ZERO : MCH_INF;
    if (mch_isnand(a)) {
      t = a + a;
    }
  }
  return t;
}

double mch_exp_kernel(double a, int scale) { 
  int i;
  double t; 
  double z;  
  
  // FMA(x,y,z) = x*y+z
  t = mch_rint(a*MCH_L2E);
  i = (int) t;
  z = t*(-MCH_LN2_HI)+a;
  z = t*(-MCH_LN2_LO)+z;
  t = mch_exp_poly(z);
  z = mch_exp_scale(t, i + scale);
  return z;
}

double mch_exp_poly(double a) {
  double t;
  
  t = 2.5052097064908941E-008;
  t = t*a + 2.7626262793835868E-007;
  t = t*a + 2.7557414788000726E-006;
  t = t*a + 2.4801504602132958E-005;
  t = t*a + 1.9841269707468915E-004;
  t = t*a + 1.3888888932258898E-003;
  t = t*a + 8.3333333333978320E-003;
  t = t*a + 4.1666666666573905E-002;
  t = t*a + 1.6666666666666563E-001;
  t = t*a + 5.0000000000000056E-001;
  t = t*a + 1.0000000000000000E+000;
  t = t*a + 1.0000000000000000E+000;
  
  return t;
}

double mch_exp_scale(double a, int i) {
  unsigned int k;
  unsigned int j;
  
  if (sabs(i) < 1023) {
    k = (i << 20) + (1023 << 20);
  } else {
    k = i + 2*1023;  
    j = k/2;
    j = j << 20;
    k = (k << 20) - j;
    a = a*hiloint2double(j, 0);
  }
  a = a*hiloint2double(k, 0);

  return a;
}

double mch_trunc(double x) {
  return (x > 0) ? floor(x) : ceil(x);
}

int mch_isinfd(double a) {
  if (VERBOSE) {
    double xxx = a+1.;
    printf("Computation of a+1:\n");
    printDd(xxx);
  }
  return (a == a) && (a-a != a-a);
//  return mch_fabs(a) == MCH_INF;
}

doubleXY mch_ddadd_xgty(doubleXY x, doubleXY y) {
  doubleXY z;
  double r; 
  double s; 
  double e;
  r = x.y + y.y;
  e = x.y - r;
  s = ((e + y.y) + y.x) + x.x;
  z.y = e = r + s;
  z.x = (r - e) + s;
  return z;
}

doubleXY mch_ddmul(doubleXY x, doubleXY y) {
  double e;
  doubleXY t; 
  doubleXY z;
  t.y = x.y*y.y;
  t.x = x.y*y.y - t.y;
  t.x = x.y*y.x + t.x;
  t.x = x.x*y.y + t.x;
  z.y = e = t.y + t.x;
  z.x = (t.y - e) + t.x;
  return z;
}


double mch_pow(double a, double b) {
  double t;
  int ahi;
  int bhi; 
  int thi; 
  int bIsOddInteger;
  
  ahi = double2hiint(a);
  bhi = double2hiint(b);
  
  if (VERBOSE) {
    printf("================================\n");
    printf("Pow of %f %f\n", a, b);
    printf("================================\n");
    printDd(a);
    printDd(b);
    printIi(ahi);
    printIi(bhi);
  }
  
  bIsOddInteger = mch_fabs(b - (2.0*mch_trunc (0.5*b))) == 1.0;
  if ((a == 1.0) || (b == 0.0)) {
    t = 1.0;
  } else if (mch_isnand(a) || mch_isnand(b)) {
    t = a + b;
  } else if (mch_isinfd(b)) {
    thi = 0;
    if (mch_fabs(a) > 1.0) {
      thi = 0x7ff00000;
    }
    if (bhi < 0) { 
      thi = thi ^ 0x7ff00000;
    }
    if (a == -1.0) {
      thi = 0x3ff00000;
    }
    t = hiloint2double (thi, 0);
  } else if (mch_isinfd(a)) {
    thi = 0;
    if (bhi >= 0) {
      thi = 0x7ff00000;
    }
    if ((ahi < 0) && bIsOddInteger) {
      thi = thi ^ 0x80000000;
    }
    t = hiloint2double(thi, 0);
  } else if (a == 0.0) {
    thi = 0;
    if (bIsOddInteger) {
      thi = ahi;
    }
    if (bhi < 0) {
      thi = thi | 0x7ff00000;
    }
    t = hiloint2double (thi, 0);
  } else if ((ahi < 0) && (b != mch_trunc(b))) {
    t = MCH_NAN;
  } else {
    t = mch_accurate_pow(mch_fabs(a), b);    
    if (VERBOSE) {
      printf("[CPU] ACC POW\n");
      printDd(a);
      printDd(mch_fabs(a));
      printDd(t);
    }        
    if ((ahi < 0) && bIsOddInteger) {
      t = mch_neg(t); 
    }
  }
  if (VERBOSE) {  
    printDd(t);
    printf("================================\n");
  }
  return t;
}

double mch_neg(double a) {
  int hi = double2hiint(a);
  int lo = double2loint(a);
  return hiloint2double(hi ^ 0x80000000, lo);
}

double mch_accurate_pow(double a, double b) {
  doubleXY loga;
  doubleXY prod;
  double t_hi;
  double t_lo;
  double tmp;
  double e;
  
  /* compute log(a) in double-double format*/
  loga = mch_log_ext_prec(a);
  
  /* prevent overflow during extended precision multiply */
  if (mch_fabs(b) > 1e304) b *= 1.220703125e-4;
  /* compute b * log(a) in double-double format */
  
  t_hi = loga.y*b;
  t_lo = loga.y*b+(-t_hi);
  
  if (VERBOSE) {
    printf("[CPU] LOGA DOUBLE-DOUBLE:\n");
    printDd(a);
    printf("[CPU] LOGA:\n");
    printDd(loga.x);
    printDd(loga.y);
    printf("[CPU] T_HI/LOW:\n");  
    printDd(t_hi);  
    printDd(t_lo);
  }
  
  t_lo = loga.x*b+t_lo;
  
  prod.y = e = t_hi + t_lo;
  prod.x = (t_hi - e) + t_lo;

  /* compute pow(a,b) = exp(b*log(a)) */
  tmp = mch_exp(prod.y);
  
  if (VERBOSE) {  
    printDd(t_lo);
    printf("[CPU] PROD.X/Y:\n");  
    printDd(prod.y);  
    printDd(prod.x);  
    printDd(tmp);
  }
  /* prevent -INF + INF = NaN */
  if (! mch_isinfd(tmp)) {
    /* if prod.x is much smaller than prod.y, then exp(prod.y + prod.x) ~= 
     * exp(prod.y) + prod.x * exp(prod.y) 
     */
    tmp = tmp*prod.x+tmp;
    if (VERBOSE) {  
      printf("[CPU] -INF+INF=NAN:\n");
      printDd(prod.x);
      printDd(tmp);
    }
  }
  return tmp;
}

double mch__half(double a)
{
  unsigned int ihi, ilo;
  ilo = double2loint(a);
  ihi = double2hiint(a);
  return hiloint2double(ihi - 0x00100000, ilo);
}

doubleXY mch_log_ext_prec(double a) {
  doubleXY res;
  doubleXY qq; 
  doubleXY cc; 
  doubleXY uu; 
  doubleXY tt;
  
  double f; 
  double g; 
  double u; 
  double v; 
  double q; 
  double ulo; 
  double tmp; 
  double m;

  int ilo; 
  int ihi; 
  int expo;

  ihi = double2hiint(a);
  ilo = double2loint(a);
  
  expo = (ihi >> 20) & 0x7ff;
  
  if (VERBOSE) {
    printf("[CPU] LOG_EXT_PREC\n");
    printDd(a);
    printIi(ihi);
    printIi(ilo);
    printIi(expo);
  }
  
  /* convert denormals to normals for computation of log(a) */
  if (expo == 0) {
    a *= MCH_TWO_TO_54;
    ihi = double2hiint(a);
    ilo = double2loint(a);
    expo = (ihi >> 20) & 0x7ff;
    expo -= 54;
  }  
  
  if (VERBOSE) {
    printIi(expo);
  }
  
  expo -= 1023;
  
  if (VERBOSE) {
    printIi(expo);
  }
  
  /*
     log(a) = log(m*2^expo) = 
     log(m) + log(2)*expo, if m < sqrt(2), 
     log(m*0.5) + log(2)*(expo+1), if m >= sqrt(2)
  */
  ihi = (ihi & 0x800fffff) | 0x3ff00000;
  m = hiloint2double (ihi, ilo);
  
  if (VERBOSE) {
    printIi(ihi);  
    printDd(m);
  }
  if ((unsigned) ihi > (unsigned) 0x3ff6a09e) {
    //m = m*0.5; //__internal_half(m);
    m = mch__half(m);
    expo = expo + 1;
    if (VERBOSE) {
      printDd(m);    
      printIi(expo);
    }
  }
  /* compute log(m) with extended precision using an algorithm derived from 
   * P.T.P. Tang, "Table Driven Implementation of the Logarithm Function", 
   * TOMS, Vol. 16., No. 4, December 1990, pp. 378-400. A modified polynomial 
   * approximation to atanh(x) on the interval [-0.1716, 0.1716] is utilized.
   */
  f = m - 1.0;
  g = m + 1.0;
  g = 1/g;
  u = f*g;
  u = u + u;  
  /* u = 2.0 * (m - 1.0) / (m + 1.0) */
  v = u*u;
  q = 6.6253631649203309E-2/65536.0;
  q = q*v + 6.6250935587260612E-2/16384.0;
  q = q*v + 7.6935437806732829E-2/4096.0;
  q = q*v + 9.0908878711093280E-2/1024.0;
  q = q*v + 1.1111111322892790E-1/256.0;
  q = q*v + 1.4285714284546502E-1/64.0;
  q = q*v + 2.0000000000003113E-1/16.0;
  q = q*v;
  /* u + ulo = 2.0 * (m - 1.0) / (m + 1.0) to more than double precision */
  tmp = 2.0*(f - u);
  tmp = -u*f + tmp; // tmp = remainder of division
  ulo = g*tmp;               // less significand quotient bits
  /* switch to double-double at this point */
  qq.y = q;
  qq.x = 0.0;
  uu.y = u;
  uu.x = ulo;
  cc.y = 3.3333333333333331E-1/4.0;
  cc.x = -9.8201492846582465E-18/4.0;
  qq = mch_ddadd_xgty(cc, qq);
  /* compute log(m) in double-double format */
  qq = mch_ddmul(qq, uu);
  qq = mch_ddmul(qq, uu);
  qq = mch_ddmul(qq, uu);
  uu = mch_ddadd_xgty(uu, qq);
  u = uu.y;
  ulo = uu.x;
  /* log(2)*expo in double-double format */
  tt.y = expo*6.9314718055966296e-01;
  tt.x = expo*2.8235290563031577e-13;
  /* log(a) = log(m) + log(2)*expo;  if expo != 0, |log(2)*expo| > |log(m)| */
  res = mch_ddadd_xgty(tt, uu);
  return res;
}


