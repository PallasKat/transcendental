//#include <cmath>

/**
  GPU version of the mathematical functions. This listing should be exactly the
  same as the one implemented for the CPU. It has been split for test purpose
  in the context of bit reproducibility.

  See cpu_mch_math.cpp for the CPU twin file.
*/

#ifdef __CUDACC__
  #pragma message using nvcc
  #define __ACC__ __device__
  namespace gm {
#else
  #pragma message probably using gcc
  #include <cmath>
  #define __ACC__
  namespace cm {
#endif
  // -------------------------------------------------------------------------
  // HELPER FUNCTION TO MANIPULATE DOUBLES
  // -------------------------------------------------------------------------
  union udouble {
    double d;
    unsigned long int li;
    unsigned long long u;
  };

  __ACC__ 
  int _double2hiint(double d) {
    udouble ud;
    ud.d = d;

    unsigned long long high = (ud.u >> 32);
    return (int) high;
  }

  __ACC__ 
  int _double2loint(double d) {
    udouble ud;
    ud.d = d;
    unsigned long long mask = 0x00000000 | 0xFFFFFFFF;
    unsigned long long low = ud.u & mask;

    return (int) low;
  }

  __ACC__ 
  double _hiloint2double(int high, int low) {
    unsigned h = high;
    unsigned l = low;

    unsigned long long uber = h;
    uber <<= 32;
    uber |= l;

    udouble u;
    u.u = uber;
    return u.d;
  }

  __ACC__ 
  void getExpoMant(double a, double* EM) {
    const double two_to_54 = 18014398509481984.0;

    int ihi = _double2hiint(a);
    int ilo = _double2loint(a);
    int e = -1023;

    /* normalize denormals */
    if ((unsigned) ihi < (unsigned) 0x00100000) {
      a = a*two_to_54;
      e -= 54;
      ihi = _double2hiint(a);
      ilo = _double2loint(a);
    }

    /*
     * a = m * 2^e.
     * m <= sqrt(2): log2(a) = log2(m) + e.
     * m > sqrt(2): log2(a) = log2(m/2) + (e+1)
     */
    e += (ihi >> 20);
    ihi = (ihi & 0x800fffff) | 0x3ff00000;

    double m = _hiloint2double(ihi, ilo);
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

  __ACC__ 
  double _rint(double a) {
    if (a > 0) {
      return (int) (a+0.5);
    }

    if (a < 0) {
      return (int) (a-0.5);
    }

    return 0;
  }

  __ACC__ 
  int i_abs(int i) {
    const int i_max = 2147483647;
    const int i_min =  -i_max - 1;

    if (i_min == i) {
      return i_max;
    } else {
      return i < 0 ? -i : i;
    }
  }

  __ACC__ 
  double f_abs(double i) {
    return i < 0 ? -i : i;
  }

  __ACC__ 
  bool _is_nan(double x) {
    return x != x;
  }

  __ACC__ 
  bool _is_inf(double x) {
    unsigned long long ull_inf = 0x7ff00000;
    ull_inf <<= 32;
    const double infinity = *reinterpret_cast<double*>(&ull_inf);
    return x == infinity;
  }

  __ACC__ bool _is_abs_inf(double y) {
    double x = f_abs(y);
    return _is_inf(x);
  }

  // -------------------------------------------------------------------------
  // LOG BASED FUNCTIONS
  // -------------------------------------------------------------------------

  /**
  * Function computing the natural logarithm
  */
  __ACC__ 
  double _log(const double x) {
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
      // log((1+m)/(1-m)) = 2*atanh(m).
      // log(m) = 2*atanh((m-1)/(m+1))      
      double f = m - 1.0;
      double g = m + 1.0;
      g = 1.0/g;
      double u = f * g;
      u = u + u;

      // u = 2.0 * (m - 1.0) / (m + 1.0)
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

      // u + ulo = 2.0 * (m - 1.0) / (m + 1.0)
      // to more than double precision
      q = q * v;
      
      // log_hi + log_lo = log(m)
      // to more than double precision
      double log_hi = u;
      double log_lo = q*u + ulo;

      // log_hi + log_lo = log(m) + e*log(2) = log(a)
      // to more than double precision
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
    // log(0) = -INF 
    else if (a == 0) {
      q = -infinity;
    }
    // log(INF) = INF
    else if (a == infinity) {
      q = a;
    }

    // log(x) is undefined for x < 0.0,
    // return INDEFINITE
    else {
      q = notanumber;
    }

    return q;
  }

  __ACC__
  double log(const double x) {
    return _log(x);
  }

  //--------------------------------------------------------------------------
  // EXPONENTIAL BASED FUNCTIONS
  //--------------------------------------------------------------------------

  __ACC__ 
  double _exp_poly(double a) {
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

  __ACC__ 
  double _exp_scale(double a, int i) {
    unsigned int k;
    unsigned int j;

    if (i_abs(i) < 1023) {
      k = (i << 20) + (1023 << 20);
    } else {
      k = i + 2*1023;
      j = k/2;
      j = j << 20;
      k = (k << 20) - j;
      a = a*_hiloint2double(j, 0);
    }
    a = a*_hiloint2double(k, 0);

    return a;
  }

  __ACC__ 
  double _exp_kernel(double a, int scale) {
    const double l2e = 1.4426950408889634e+0;
    const double ln2_hi = 6.9314718055994529e-1;
    const double ln2_lo = 2.3190468138462996e-17;

    double t = _rint(a*l2e);
    int i = (int) t;
    double z = t*(-ln2_hi)+a;
    z = t*(-ln2_lo)+z;
    t = _exp_poly(z);
    z = _exp_scale(t, i + scale);
    return z;
  }

  __ACC__ 
  double _exp(double x) {
    //printf("x -> %f", x);
    unsigned long long ull_inf = 0x7ff00000;
    ull_inf <<= 32;
    const double infinity = *reinterpret_cast<double*>(&ull_inf);

    double a = (double) x;
    double t;
    int i = _double2hiint(a);
    // We only check if we are in a specific range [-a,b] to compute
    // the exp
    if (((unsigned) i < (unsigned) 0x40862e43) || ((int) i < (int) 0xC0874911)) {
      t = _exp_kernel(a, 0);
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

  __ACC__
  double exp(const double x) {
    return _exp(x);
  }

  //--------------------------------------------------------------------------
  // POWER BASED FUNCTIONS
  //--------------------------------------------------------------------------

  __ACC__ 
  double _internal_pow(const double x, const double y) {
    return _exp(y*_log(x));
  }

  __ACC__ 
  double _pow(const double x, const double y) {
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

    else if (_is_nan(x) || _is_nan(y)) {
      return x + y;
    }

    else if (_is_abs_inf(y)) {
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

    else if (_is_abs_inf(x)) {
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

    else { // pow(a,b) = exp(b*log(a))
      double ax = f_abs(x);
      double z = _internal_pow(ax, y);
      if (x < 0.0 && yFloorOdd) {
        return -z;
      }
      else {
        return z;
      }
    }
  }

  __ACC__
  double pow(const double x, const double y) {
    return _pow(x, y);
  }
}
