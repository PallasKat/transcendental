/**
  GPU version of the mathematical functions. This listing should be exactly the 
  same as the one implemented for the CPU. It has been split for test purpose
  in the context of bit reproducibility.
  
  See cpu_mch_math.cpp for the CPU twin file.
*/

// -------------------------------------------------------------------------
// HELPER FUNCTION TO MANIPULATE DOUBLES
// -------------------------------------------------------------------------

union udouble {
  double d;
  unsigned long int li;
  unsigned long long u;
};

__device__ int mch_double2hiint(double d) {
  udouble ud;
  ud.d = d;

  unsigned long long high = (ud.u >> 32);
  return (int) high;
}

__device__ int mch_double2loint(double d) {
  udouble ud;
  ud.d = d;
  unsigned long long mask = 0x00000000 | 0xFFFFFFFF;
  unsigned long long low = ud.u & mask;

  return (int) low;
}

__device__ double mch_hiloint2double(int high, int low) {
  unsigned h = high;
  unsigned l = low;

  unsigned long long uber = h;
  uber <<= 32;
  uber |= l;

  udouble u;
  u.u = uber;
  return u.d;
}

__device__ void getExpoMant(double a, double* EM) {
  const double two_to_54 = 18014398509481984.0;

  int ihi = mch_double2hiint(a);
  int ilo = mch_double2loint(a);
  int e = -1023;

  /* normalize denormals */
  if ((unsigned) ihi < (unsigned) 0x00100000) {
    a = a*two_to_54;
    e -= 54;
    ihi = mch_double2hiint(a);
    ilo = mch_double2loint(a);
  }
  
  /*
   * a = m * 2^e.
   * m <= sqrt(2): log2(a) = log2(m) + e.
   * m > sqrt(2): log2(a) = log2(m/2) + (e+1)
   */
  e += (ihi >> 20);
  ihi = (ihi & 0x800fffff) | 0x3ff00000;

  double m = mch_hiloint2double(ihi, ilo);
  if ((unsigned) ihi > (unsigned) 0x3ff6a09e) {
    m = 0.5*m;
    e = e + 1;
  }

  EM[0] = (double) e;
  EM[1] = m;
}

// -------------------------------------------------------------------------
// HELPER MATHEMATICAL FUNCTIONS
// -------------------------------------------------------------------------

__device__ double mch_rint(double a) {
  if (a > 0) {
    return (int) (a+0.5);
  }

  if (a < 0) {
    return (int) (a-0.5);
  }

  return 0;
}

__device__ int i_abs(int i) {
  const int i_max = 2147483647;
  const int i_min =  -i_max - 1;

  if (i_min == i) {
    return i_max;
  } else {
    return i < 0 ? -i : i;
  }
}

__device__ double f_abs(double i) {
  return i < 0 ? -i : i;
}

__device__ bool __is_nan(double x) {
  return x != x;
}

__device__ bool __is_inf(double x) {
  unsigned long long ull_inf = 0x7ff00000;
  ull_inf <<= 32;
  const double infinity = *reinterpret_cast<double*>(&ull_inf);
  return x == infinity;
}

__device__ bool __is_abs_inf(double y) {
  double x = f_abs(y);
  return __is_inf(x);
}

// -------------------------------------------------------------------------
// END OF HELPERS
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// LOG BASED FUNCTIONS
// -------------------------------------------------------------------------

/**
* Function computing the natural logarithm
*/
__device__ double friendly_log(const double x) {
  double a = (double) x;
  const double ln2_hi = 6.9314718055994529e-1;
  const double ln2_lo = 2.3190468138462996e-17;
  
  unsigned long long ull_inf = 0x7ff00000;
  ull_inf <<= 32;
  const double infinity = *reinterpret_cast<double*>(&ull_inf);
  
  unsigned long long ull_nan = 0xfff80000;
  ull_nan <<= 32;
  const double notanumber = *reinterpret_cast<double*>(&ull_nan);
  
  double EM[2];
  getExpoMant(a, EM);
  double e = EM[0];
  double m = EM[1];
  
  double q;

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
     * log_hi + log_lo = log(m) + e*log(2) = log(a)
     * to more than double precision
     */
    q = e*ln2_hi + log_hi;

    tmp = -e*ln2_hi + q;
    tmp = tmp - log_hi;
    log_hi = q;
    log_lo = log_lo - tmp;
    log_lo = e*ln2_lo + log_lo;

    q = log_hi + log_lo;
  } else if (a != a) {
    q = a + a;
  }
  /* log(0) = -INF */
  else if (a == 0) {
    q = -infinity;
  }
  /* log(INF) = INF */
  else if (a == infinity) {
    q = a;
  }
  /*
   * log(x) is undefined for x < 0.0,
   * return INDEFINITE
   */
  else {
    q = notanumber;
  }

  //printf("Returned q: %f\n", q);
  //double qq = log(x);
  //printf("Should be: %f\n", qq);
  return q;
}

/*
__device__ double friendly_pow(const double x, const double y) {
  return friendly_log(x)*y;
}
*/
//--------------------------------------------------------------------------
// EXPONENTIAL BASED FUNCTIONS
//--------------------------------------------------------------------------

__device__ double __exp_poly(double a) {
  double t = 2.5052097064908941E-008;
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

__device__ double __exp_scale(double a, int i) {
  unsigned int k;
  unsigned int j;

  if (i_abs(i) < 1023) {
    k = (i << 20) + (1023 << 20);
  } else {
    k = i + 2*1023;
    j = k/2;
    j = j << 20;
    k = (k << 20) - j;
    a = a*mch_hiloint2double(j, 0);
  }
  a = a*mch_hiloint2double(k, 0);

  return a;
}

__device__ double __exp_kernel(double a, int scale) {
  const double l2e = 1.4426950408889634e+0;
  const double ln2_hi = 6.9314718055994529e-1;
  const double ln2_lo = 2.3190468138462996e-17;
  
  double t = mch_rint(a*l2e);
  int i = (int) t;
  double z = t*(-ln2_hi)+a;
  z = t*(-ln2_lo)+z;
  t = __exp_poly(z);
  z = __exp_scale(t, i + scale);
  return z;
}

__device__ double friendly_exp(double x) {
  unsigned long long ull_inf = 0x7ff00000;
  ull_inf <<= 32;
  const double infinity = *reinterpret_cast<double*>(&ull_inf);
  
  double a = (double) x;
  double t;
  int i = mch_double2hiint(a);
  // We only check if we are in a specific range [-a,b] to compute
  // the exp
  if (((unsigned) i < (unsigned) 0x40862e43) || ((int) i < (int) 0xC0874911)) {
    t = __exp_kernel(a, 0);
  }
  // Otherwise the result is a very small value, then returning 0 or
  // a very large value then returning inf
  else {
    t = (i < 0) ? 0.0 : infinity;
    // is a == NaN ?
    if (a != a) {
      t = a + a;
    }
  }
  return t;
}

//--------------------------------------------------------------------------
// POWER BASED FUNCTIONS
//--------------------------------------------------------------------------

__device__ double __internal_pow(const double x, const double y) {
  double lnx = friendly_log(x);
  double ylnx = y*lnx;
  return friendly_exp(ylnx);
}

__device__ double friendly_pow(const double x, const double y) {
  unsigned long long ull_inf = 0x7ff00000;
  ull_inf <<= 32;
  const double infinity = *reinterpret_cast<double*>(&ull_inf);
  
  unsigned long long ull_nan = 0xfff80000;
  ull_nan <<= 32;
  const double notanumber = *reinterpret_cast<double*>(&ull_nan);
  
  int yFloorOdd = f_abs(y - (2.0*trunc(0.5*y))) == 1.0;
  
  if ((x == 1.0) || (y == 0.0)) {
    return 1.0;
  } 
  
  else if (__is_nan(x) || __is_nan(y)) {    
    return x + y;
  }
  
  else if (__is_abs_inf(y)) {
    double ax = f_abs(x);
    if (ax > 1.0) {      
      if (y < 0.0) { // y is -infinity
        return infinity;
      }       
      else { // y is infinity
        return 0.0;
      }
    } else {
      if (x == -1.0) {
        return 1.0;
      } 
      else {        
       if (y < 0.0) { // y is -infinity
          return 0.0;
        }         
        else { // y is infinity
          return infinity;
        } 
      }
    }        
  }
  
  else if (__is_abs_inf(x)) {
    if (y >= 0) {
      if (x < 0.0 && yFloorOdd) {
        return -1.0*infinity;
      } else {
        return infinity;
      }
    }    
    else {
      if (x < 0.0 && yFloorOdd) {
        return -0.0;
      } else {
        return 0.0;
      }
    }
  }
  
  else if (x == 0.0) {
    if (y > 0.0) {
      return 0.0;
    } else {
      return infinity;
    }
  }
  
  else if ((x < 0.0) && (y != trunc(y))) {
    return notanumber;
  }
  
  else { //pow(a,b) = exp(b*log(a))
    double ax = f_abs(x);
    double z = __internal_pow(ax, y);
    if (x < 0.0 && yFloorOdd) {
      return -z;
    }
    else {
      return z;
    }
  }
}

