#include "portable_math.h"

#include <stdint.h>

# define __BITREPFMA(a,b,c) (a*b+c)

#ifdef __GPUACC__
  #pragma message "using openacc"
  #include <cmath>
#else
  #pragma message "probably using gcc"
  #include <cmath>
#endif

//namespace bitrep {
// -------------------------------------------------------------------------
// HELPER FUNCTION TO MANIPULATE -  DOUBLES
// -------------------------------------------------------------------------
  union udouble {
   double d;
   unsigned long int li;
   unsigned long long u;
  };

  #pragma acc routine seq 
  int _double2hiint(double d) {
    udouble ud;
    ud.d = d;

    unsigned long long high = (ud.u >> 32);
    return (int) high;
  }

  #pragma acc routine seq 
  int _double2loint(double d) {
    udouble ud;
    ud.d = d;
    unsigned long long mask = 0x00000000 | 0xFFFFFFFF;
    unsigned long long low = ud.u & mask;

    return (int) low;
  }

  #pragma acc routine seq 
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

  // -------------------------------------------------------------------------
  // HELPER MATHEMATICAL FUNCTIONS - DOUBLES
  // -------------------------------------------------------------------------

  #pragma acc routine seq 
  double _rint(double a) {
    if (a > 0) {
      return (int) (a+0.5);
    }

    if (a < 0) {
      return (int) (a-0.5);
    }

    return 0;
  }

  #pragma acc routine seq 
  int i_abs(int i) {
    const int i_max = 2147483647;
    const int i_min = -i_max - 1;

    if (i_min == i) {
      return i_max;
    } else {
      return i < 0 ? -i : i;
    }
  }

  #pragma acc routine seq 
  double f_abs(double i) {
    return i < 0 ? -i : i;
  }

  #pragma acc routine seq 
  bool _is_nan(double x) {
    return x != x;
  }

  #pragma acc routine seq 
  bool _is_inf(double x) {
    unsigned long long ull_inf = 0x7ff00000;
    ull_inf <<= 32;
    const double infinity = *reinterpret_cast<double*>(&ull_inf);
    return x == infinity;
  }

  #pragma acc routine seq 
  bool _is_abs_inf(double y) {
    double x = f_abs(y);
    return _is_inf(x);
  }

  // -------------------------------------------------------------------------
  // LOG BASED FUNCTIONS - DOUBLES
  // -------------------------------------------------------------------------

  /**
  * Function computing the natural logarithm
  */
  #pragma acc routine seq 
  double _log(const double x) {
    double a = (double) x;
    const double ln2_hi = 6.9314718055994529e-1;
    const double ln2_lo = 2.3190468138462996e-17;
    const double two_to_54 = 18014398509481984.0;

    unsigned long long ull_inf = 0x7ff00000;
    ull_inf <<= 32;
    const double infinity = *reinterpret_cast<double*>(&ull_inf);

    unsigned long long ull_nan = 0xfff80000;
    ull_nan <<= 32;
    const double notanumber = *reinterpret_cast<double*>(&ull_nan);

      // From a double (64 bits) represented as the IEEE standard, get
  // the exponent and the mantissa
  int ihi = _double2hiint(a);
  int ilo = _double2loint(a);
  // exponent bias
  int ei = -1023;

  /* normalize denormals */
  if ((unsigned) ihi < (unsigned) 0x00100000) {
    a = a*two_to_54;
    ei -= 54;
    ihi = _double2hiint(a);
    ilo = _double2loint(a);
  }

  /*
   * a = m * 2^e.
   * m <= sqrt(2): log2(a) = log2(m) + e.
   * m > sqrt(2): log2(a) = log2(m/2) + (e+1)
   */
  ei += (ihi >> 20);
  ihi = (ihi & 0x800fffff) | 0x3ff00000;
  // the mantissa
  double m = _hiloint2double(ihi, ilo);
  if ((unsigned) ihi > (unsigned) 0x3ff6a09e) {
    m = 0.5*m;
    ei = ei + 1;
  }
  // the exponent
  double e = (double) ei;

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

  #pragma acc routine seq
  double br_log(const double x) {
    return _log(x);
  }

  //--------------------------------------------------------------------------
  // EXPONENTIAL BASED FUNCTIONS - DOUBLES
  //--------------------------------------------------------------------------

  #pragma acc routine seq 
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

  #pragma acc routine seq 
  double _exp_scale(double b, int i) {
    double a = b;
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

  #pragma acc routine seq 
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

  #pragma acc routine seq 
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

  #pragma acc routine seq
  double br_exp(const double x) {
    return _exp(x);
  }

  //--------------------------------------------------------------------------
  // POWER BASED FUNCTIONS - DOUBLES
  //--------------------------------------------------------------------------

  #pragma acc routine seq 
  double _internal_pow(const double x, const double y) {
    return _exp(y*_log(x));
  }

  #pragma acc routine seq 
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
        if (y < 0.0) { // y is less than 0 and to -infinity
          return 0.0;
        }
        else { // y is infinity
          return infinity;
        }
      } else {
        if (x == -1.0) {
          return 1.0;
        }
        else {
          if (y < 0.0 && y != -infinity) { // y is less than 0 but not -infinity
            return 0.0;
          } else if (y == infinity){ // y is infinity
                return 0.0;
           } else if (y == -infinity){ // y is -infinity
                return infinity; 
	         } else { // y is greater than 0 but not infinity
            	return infinity;
          }
        }
      }
   } 

    else if (_is_abs_inf(x)){
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

  #pragma acc routine seq
  double br_pow(const double x, const double y) {
	  return _pow(x, y);
  }

// -------------------------------------------------------------------------
//    TRIGNOMETRY FUNCTIONS
// -------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------
//    HELPER FUNCTION TO MANIPULATE DOUBLES OF TRIGNOMETRY FUNCTIONS
// ------------------------------------------------------------------------------------------------
  #pragma acc routine seq
  double __internal_sin_cos_kerneld(double x_in, int q)
  {
      double x = x_in;
      double __sin_cos_coefficient[16];

      __sin_cos_coefficient[0]  = 1.590307857061102704e-10;
      __sin_cos_coefficient[1]  = -2.505091138364548653e-08;
      __sin_cos_coefficient[2]  =  2.755731498463002875e-06;
      __sin_cos_coefficient[3]  = -1.984126983447703004e-04;
      __sin_cos_coefficient[4]  =  8.333333333329348558e-03;
      __sin_cos_coefficient[5]  = -1.666666666666666297e-01;
      __sin_cos_coefficient[6]  =  0.00000000000000000;
      __sin_cos_coefficient[7]  =  0.00000000000000000;
      __sin_cos_coefficient[8]  = -1.136781730462628422e-11;
      __sin_cos_coefficient[9]  =  2.087588337859780049e-09;
      __sin_cos_coefficient[10] = -2.755731554299955694e-07;
      __sin_cos_coefficient[11] =  2.480158729361868326e-05;
      __sin_cos_coefficient[12] = -1.388888888888066683e-03;
      __sin_cos_coefficient[13] =  4.166666666666663660e-02;
      __sin_cos_coefficient[14] = -5.000000000000000000e-01;
      __sin_cos_coefficient[15] =  1.000000000000000000e+0;

      unsigned int idx = 8*(q&1);
      double x2 = x*x;

      double z = (q & 1) ? -1.136781730462628422e-11 : 1.590307857061102704e-10;

      z = __BITREPFMA(z, x2, __sin_cos_coefficient[idx+1]);
      z = __BITREPFMA(z, x2, __sin_cos_coefficient[idx+2]);
      z = __BITREPFMA(z, x2, __sin_cos_coefficient[idx+3]);
      z = __BITREPFMA(z, x2, __sin_cos_coefficient[idx+4]);
      z = __BITREPFMA(z, x2, __sin_cos_coefficient[idx+5]);
      z = __BITREPFMA(z, x2, __sin_cos_coefficient[idx+6]);

      x = __BITREPFMA(z, x, x);

      if (q & 1) x = __BITREPFMA(z, x2, 1.);
      if (q & 2) x = __BITREPFMA(x, -1., 0.);

      return x;
  }

  #pragma acc routine seq
  double __internal_trig_reduction_kerneld(double x ,  int *q_)
  {
      double __const_2_over_pi = 6.3661977236758138e-1;
      double j, t;
      int& q = *q_;

      q = static_cast<int>(floor(x* __const_2_over_pi + .5));
      j = q;

      t = (-j) * 1.5707963267948966e+000 + x; 
      t = (-j) * 6.1232339957367574e-017 + t;
      t = (-j) * 8.4784276603688985e-032 + t;

      return t;
  }

  #pragma acc routine seq
  double __internal_tan_kernel(double x, int i)
  {
    double x2, z, q;
    x2 = x*x;
    z = 9.8006287203286300E-006;

    z = __BITREPFMA(z, x2, -2.4279526494179897E-005);
    z = __BITREPFMA(z, x2,  4.8644173130937162E-005);
    z = __BITREPFMA(z, x2, -2.5640012693782273E-005);
    z = __BITREPFMA(z, x2,  6.7223984330880073E-005);
    z = __BITREPFMA(z, x2,  8.3559287318211639E-005);
    z = __BITREPFMA(z, x2,  2.4375039850848564E-004);
    z = __BITREPFMA(z, x2,  5.8886487754856672E-004);
    z = __BITREPFMA(z, x2,  1.4560454844672040E-003);
    z = __BITREPFMA(z, x2,  3.5921008885857180E-003);
    z = __BITREPFMA(z, x2,  8.8632379218613715E-003);
    z = __BITREPFMA(z, x2,  2.1869488399337889E-002);
    z = __BITREPFMA(z, x2,  5.3968253972902704E-002);
    z = __BITREPFMA(z, x2,  1.3333333333325342E-001);
    z = __BITREPFMA(z, x2,  3.3333333333333381E-001);
    z = z * x2;
    q = __BITREPFMA(z, x, x);

    if (i) {
        double s = q - x;
        double w = __BITREPFMA(z, x, -s); // tail of q
        z = - (1. / q);
        s = __BITREPFMA(q, z, 1.0);
        q = __BITREPFMA(__BITREPFMA(z,w,s), z, z);
    }

    return q;
  }

// =============================================================================
//  SINE BASED FUNCTIONS
//  =============================================================================

  #pragma acc routine seq
  double _sin(double x_in)
  {
    double x = x_in;
    double z;
    int q;

    unsigned long long ull_inf = 0x7ff00000;
    ull_inf <<= 32;
    const double infinity = *reinterpret_cast<double*>(&ull_inf);
   
    if (x == infinity || x == -infinity) {
      x = x * 0.; // Gives NaN
    }
   
    z = __internal_trig_reduction_kerneld(x, &q);
    z = __internal_sin_cos_kerneld(z, q);

    return z;
  }

  #pragma acc routine seq
  double br_sin(const double x) {
    return _sin(x); 
  }

// =============================================================================
// COSINE BASED FUNCTIONS
// =============================================================================

  #pragma acc routine seq
  double _cos(double x_in)
  {
    double x = x_in;
    double z;
    int q;

    unsigned long long ull_inf = 0x7ff00000;
    ull_inf <<= 32;
    const double infinity = *reinterpret_cast<double*>(&ull_inf);

    if (x == infinity || x == -infinity) {
        x = x * 0.; // Gives NaN
     } 

    z = __internal_trig_reduction_kerneld(x, &q);
    ++q;
    z = __internal_sin_cos_kerneld(z, q);

    return z;
  }

  #pragma acc routine seq
  double br_cos(double x){
      return _cos(x);

  }

// =============================================================================
//  TANGENT BASED FUNCTIONS
// =============================================================================

  #pragma acc routine seq
  double _tan(double x_in)
  {
      double x = x_in;
      unsigned long long ull_inf = 0x7ff00000;
      ull_inf <<= 32;
      const double infinity = *reinterpret_cast<double*>(&ull_inf);

      double z;
      int i;

      if (x == infinity || x == -infinity) {
          x = x * 0.; // Gives NaN
      }
      z = __internal_trig_reduction_kerneld(x, &i);
      z = __internal_tan_kernel(z, i & 1);
      return z;
  }

  #pragma acc routine seq
  double br_tan(double x){
      return _tan(x);

  }

/*   ***********************************
 *  * INVERSE TRIGONOMETRIC FUNCTIONS *
 *   ***********************************/

// ------------------------------------------------------------------------------------------------
//  HELPER FUNCTION TO MANIPULATE DOUBLES OF INVERSE TRIGNOMETRY FUNCTIONS
// ------------------------------------------------------------------------------------------------

  #pragma acc routine seq
  double __internal_copysign_pos(double a, double b)
  {
      union {
          int32_t i[2];
          double d;
      } aa, bb;
      aa.d = a;
      bb.d = b;
      aa.i[1] = (bb.i[1] & 0x80000000) | aa.i[1];
      return aa.d;
  }

  #pragma acc routine seq
  double __internal_asin_kernel(double x)
  {
    double r;
    r = 6.259798167646803E-002;
    r = __BITREPFMA (r, x, -7.620591484676952E-002);
    r = __BITREPFMA (r, x,  6.686894879337643E-002);
    r = __BITREPFMA (r, x, -1.787828218369301E-002); 
    r = __BITREPFMA (r, x,  1.745227928732326E-002);
    r = __BITREPFMA (r, x,  1.000422754245580E-002);
    r = __BITREPFMA (r, x,  1.418108777515123E-002);
    r = __BITREPFMA (r, x,  1.733194598980628E-002);
    r = __BITREPFMA (r, x,  2.237350511593569E-002);
    r = __BITREPFMA (r, x,  3.038188875134962E-002);
    r = __BITREPFMA (r, x,  4.464285849810986E-002);
    r = __BITREPFMA (r, x,  7.499999998342270E-002);
    r = __BITREPFMA (r, x,  1.666666666667375E-001);
    r = r * x;
    return r;
  }

  #pragma acc routine seq
  double __internal_atan_kernel(double x)
  {
    double t, x2;
    x2 = x * x;
    t = -2.0258553044438358E-005 ;
    t = __BITREPFMA (t, x2,  2.2302240345758510E-004);
    t = __BITREPFMA (t, x2, -1.1640717779930576E-003);
    t = __BITREPFMA (t, x2,  3.8559749383629918E-003);
    t = __BITREPFMA (t, x2, -9.1845592187165485E-003);
    t = __BITREPFMA (t, x2,  1.6978035834597331E-002);
    t = __BITREPFMA (t, x2, -2.5826796814495994E-002);
    t = __BITREPFMA (t, x2,  3.4067811082715123E-002);
    t = __BITREPFMA (t, x2, -4.0926382420509971E-002);
    t = __BITREPFMA (t, x2,  4.6739496199157994E-002);
    t = __BITREPFMA (t, x2, -5.2392330054601317E-002);
    t = __BITREPFMA (t, x2,  5.8773077721790849E-002);
    t = __BITREPFMA (t, x2, -6.6658603633512573E-002);
    t = __BITREPFMA (t, x2,  7.6922129305867837E-002);
    t = __BITREPFMA (t, x2, -9.0909012354005225E-002);
    t = __BITREPFMA (t, x2,  1.1111110678749424E-001);
    t = __BITREPFMA (t, x2, -1.4285714271334815E-001);
    t = __BITREPFMA (t, x2,  1.9999999999755019E-001);
    t = __BITREPFMA (t, x2, -3.3333333333331860E-001);
    t = t * x2;
    t = __BITREPFMA (t, x, x);
    return t;
  }

// =============================================================================
//   INVERSE OF SINE BASED FUNCTIONS
// =============================================================================

  #pragma acc routine seq
  double _asin(double x)
  {
    double fx, t0, t1;
    double xhi, ihi;

    union {
        int32_t i[2];
        double d;
    } xx, fxx;

    fx = f_abs(x);
    xx.d = x;
    xhi = xx.i[1];
    fxx.d = fx;
    ihi = fxx.i[1];

    if (ihi < 0x3fe26666) {
      t1 = fx * fx;
      t1 = __internal_asin_kernel (t1);
      t1 = __BITREPFMA (t1, fx, fx);
      t1 = __internal_copysign_pos(t1, x);
    } else {
      t1 = __BITREPFMA (-0.5, fx, 0.5);
      t0 = std::sqrt (t1);
      t1 = __internal_asin_kernel (t1);
      t0 = -2.0 * t0;
      t1 = __BITREPFMA (t0, t1, 6.1232339957367660e-17);
      t0 = t0 + 7.8539816339744828e-1;
      t1 = t0 + t1;
      t1 = t1 + 7.8539816339744828e-1;
      if (xhi < 0x3ff00000) {
        t1 = __internal_copysign_pos(t1, x); 	
      }		
    }
    return t1;
  }

  #pragma acc routine seq
  double br_asin(double x){
      return _asin(x);
  }

// =============================================================================
//    INVERSE OF COSINE BASED FUNCTIONS
// =============================================================================


  #pragma acc routine seq
  double _acos(double x)
  {
    double t0, t1;

    union {
      int32_t i[2];
      double d;
    } xx, fxx;
    xx.d = x;
    fxx.d = (t0 = f_abs(x));

    const int32_t& xhi = xx.i[1];
    const int32_t& ihi = fxx.i[1];

    unsigned long long ull_inf = 0x7ff00000;
    ull_inf <<= 32;
    const double infinity = *reinterpret_cast<double*>(&ull_inf);

    if (ihi < 0x3fe26666) {  
        t1 = t0 * t0;
        t1 = __internal_asin_kernel (t1);
        t0 = __BITREPFMA (t1, t0, t0);
        if (xhi < 0) {
          t0 = t0 + 6.1232339957367660e-17;
          t0 = 1.5707963267948966e+0 + t0;
        } else {
          t0 = t0 - 6.1232339957367660e-17;
          t0 = 1.5707963267948966e+0 - t0;
        }
    } else {
      // acos(x) = [y + y^2 * p(y)] * rsqrt(y/2), y = 1 - x 
      double p, r, y;
      y = 1.0 - t0;
      r = 1. / std::sqrt(y / 2.);
      p = 2.7519189493111718E-006;
      p = __BITREPFMA (p, y, -1.5951212865388395E-006);
      p = __BITREPFMA (p, y,  6.1185294127269731E-006);
      p = __BITREPFMA (p, y,  6.9283438595562408E-006);
      p = __BITREPFMA (p, y,  1.9480663162164715E-005);
      p = __BITREPFMA (p, y,  4.5031965455307141E-005);
      p = __BITREPFMA (p, y,  1.0911426300865435E-004);
      p = __BITREPFMA (p, y,  2.7113554445344455E-004);
      p = __BITREPFMA (p, y,  6.9913006155254860E-004);
      p = __BITREPFMA (p, y,  1.8988715243469585E-003);
      p = __BITREPFMA (p, y,  5.5803571429249681E-003);
      p = __BITREPFMA (p, y,  1.8749999999999475E-002);
      p = __BITREPFMA (p, y,  8.3333333333333329E-002);
      p = p * y * y * r;
      fxx.d = y;
      if (ihi <= 0) {
          t0 = t0 * 0.;
      } else {
          t0 = __BITREPFMA (r, y, p);
      }
        if (ihi < 0) {
          t0 = t0 * infinity;// EDIT THIS FUNCTION
      }
      if (xhi < 0) {    
          t0 = t0 - 1.2246467991473532e-16;
          t0 = 3.1415926535897931e+0 - t0;
      }
    } 
    return t0;
  }

  #pragma acc routine seq
  double br_acos(double x){
      return _acos(x);
  }

// =============================================================================
//    INVERSE OF TANGENT BASED FUNCTIONS
// =============================================================================

  #pragma acc routine seq
  double _atan(double x)
  {
      double t0, t1;
      // reduce argument to first octant
      t0 = f_abs(x);
      t1 = t0;
      if (t0 > 1.0) {
        t1 = 1.0/t1;
        if (_is_inf(t0)) t1 = 0.0;
         //if (t0 == double INFINITY) t1 = 0.0;// EDIT THIS FUNCTION
      }

      // approximate atan(r) in first octant
      t1 = __internal_atan_kernel(t1);

      // map result according to octant. 
      if (t0 > 1.0) {
          t1 = 1.5707963267948966e+0 - t1;
      }
      return __internal_copysign_pos(t1, x);
  }

  #pragma acc routine seq
  double br_atan(double x){
      return _atan(x);
  }

/************************
 * HYPERBOLIC FUNCTIONS *
 ************************/

// ------------------------------------------------------------------------------------------------
//   HELPER FUNCTION TO MANIPULATE DOUBLES OF HYPERBOLIC FUNCTIONS
// ------------------------------------------------------------------------------------------------

  #pragma acc routine seq
  double __internal_expm1_kernel(double x)
  {
    double t;
    t = 2.0900320002536536E-009;
    t = __BITREPFMA (t, x, 2.5118162590908232E-008);
    t = __BITREPFMA (t, x, 2.7557338697780046E-007);
    t = __BITREPFMA (t, x, 2.7557224226875048E-006);
    t = __BITREPFMA (t, x, 2.4801587233770713E-005);
    t = __BITREPFMA (t, x, 1.9841269897009385E-004);
    t = __BITREPFMA (t, x, 1.3888888888929842E-003);
    t = __BITREPFMA (t, x, 8.3333333333218910E-003);
    t = __BITREPFMA (t, x, 4.1666666666666609E-002);
    t = __BITREPFMA (t, x, 1.6666666666666671E-001);
    t = __BITREPFMA (t, x, 5.0000000000000000E-001);
    t = t * x;
    t = __BITREPFMA (t, x, x);
    return t;
  }

  #pragma acc routine seq
  double __internal_exp2i_kernel(int32_t b)
  {
      union {
          int32_t i[2];
          double d;
      } xx;

      xx.i[0] = 0;
      xx.i[1] = (b + 1023) << 20;

      return xx.d;
  }

  #pragma acc routine seq
  double __internal_expm1_scaled(double x_in, int scale)
  { 
    double x = x_in;
    double t, z, u;
    int i, j;

    union {
        uint32_t i[2];
        double d;
    } xx;
    xx.d = x;
    uint32_t& k = xx.i[1];

    t = std::floor (__BITREPFMA(x, 1.4426950408889634e+0, 4.99999999999999945e-1));
    i = (int)t + scale;
    z = __BITREPFMA (t, -6.9314718055994529e-1, x);
    z = __BITREPFMA (t, -2.3190468138462996e-17, z);
    k = k + k;
    if ((unsigned)k < (unsigned)0x7fb3e647) {
      z = x;
      i = 0;
    }
    t = __internal_expm1_kernel(z);
    j = i;
    if (i == 1024) j--;
    u = __internal_exp2i_kernel(j);

    xx.i[0] = 0;
    xx.i[1] = 0x3ff00000 + (scale << 20);
    x = xx.d;

    x = u - x;
    t = __BITREPFMA (t, u, x);
    if (i == 1024) t = t + t;
    if (k == 0) t = z;              /* preserve -0 */
    return t;
  } 

  #pragma acc routine seq
  double __internal_exp_poly(double x)
  {
    double t;

    t = 2.5052097064908941E-008;
    t = __BITREPFMA (t, x, 2.7626262793835868E-007);
    t = __BITREPFMA (t, x, 2.7557414788000726E-006);
    t = __BITREPFMA (t, x, 2.4801504602132958E-005);
    t = __BITREPFMA (t, x, 1.9841269707468915E-004);
    t = __BITREPFMA (t, x, 1.3888888932258898E-003);
    t = __BITREPFMA (t, x, 8.3333333333978320E-003);
    t = __BITREPFMA (t, x, 4.1666666666573905E-002);
    t = __BITREPFMA (t, x, 1.6666666666666563E-001);
    t = __BITREPFMA (t, x, 5.0000000000000056E-001);
    t = __BITREPFMA (t, x, 1.0000000000000000E+000);
    t = __BITREPFMA (t, x, 1.0000000000000000E+000);
    return t;
  }

  #pragma acc routine seq
  double __internal_exp_scale(double x_in, int i)
  {
      double x = x_in;
      unsigned int j, k;

      union {
          int32_t i[2];
          double d;
      } xx;

      if (f_abs(i) < 1023) {
          k = (i << 20) + (1023 << 20);
      } else {
          k = i + 2*1023;
          j = k / 2;
          j = j << 20;
          k = (k << 20) - j;
          xx.i[0] = 0;
          xx.i[1] = j;
          x = x * xx.d;
      }

      xx.i[0] = 0;
      xx.i[1] = k;
      x = x * xx.d;

      return x;
  }

  #pragma acc routine seq
  double __internal_exp_kernel(double x, int scale)
  { 
  	double t, z;
    int i;

    t = std::floor (x*1.4426950408889634e+0 + 4.99999999999999945e-1);
    i = (int)t;
    z = __BITREPFMA(t, -6.9314718055994529e-1, x);
    z = __BITREPFMA(t, -2.3190468138462996e-17, z);
    t = __internal_exp_poly (z);
    z = __internal_exp_scale (t, i + scale); 
    return z;
  }  

  #pragma acc routine seq
  double __internal_old_exp_kernel(double x, int scale)
  { 
      double t, z;
      int i, j, k;

      union {
          int32_t i[2];
          double d;
      } zz;

      t = std::floor (__BITREPFMA(x, 1.4426950408889634e+0, 4.99999999999999945e-1));
      i = (int)t;
      z = __BITREPFMA (t, -6.9314718055994529e-1, x);
      z = __BITREPFMA (t, -2.3190468138462996e-17, z);
      t = __internal_expm1_kernel (z);
      k = ((i + scale) << 20) + (1023 << 20);

      if (f_abs(i) < 1021) {
          zz.i[0] = 0; zz.i[1] = k;
          z = zz.d;
          z = __BITREPFMA (t, z, z);
      } else {
          j = 0x40000000;
          if (i < 0) {
              k += (55 << 20);
              j -= (55 << 20);
          }
          k = k - (1 << 20);

          zz.i[0] = 0; zz.i[1] = j; /* 2^-54 if a is denormal, 2.0 otherwise */
          z = zz.d;
          t = __BITREPFMA (t, z, z);
          
          zz.i[0] = 0; zz.i[1] = k; /* 2^-54 if a is denormal, 2.0 otherwise */
          z = zz.d;
          z = t * z;
      }
      return z;
  }   

// =============================================================================
//     HYPERBOLIC SINE BASED FUNCTIONS
// =============================================================================

  #pragma acc routine seq
  double _sinh(double x)
  {
      double z;

      unsigned long long ull_inf = 0x7ff00000;
      ull_inf <<= 32;
      const double infinity = *reinterpret_cast<double*>(&ull_inf);

      union {
          int32_t i[2];
          double d;
      } xx;
      xx.d = x;
      xx.i[1] = xx.i[1] & 0x7fffffff;

      int32_t& thi = xx.i[1];
      //int32_t& tlo = xx.i[0];
      double& t = xx.d;

      if (thi < 0x3ff00000) {
          double t2 = t*t;
          z = 7.7587488021505296E-013;
          z = __BITREPFMA (z, t2, 1.6057259768605444E-010);
          z = __BITREPFMA (z, t2, 2.5052123136725876E-008);
          z = __BITREPFMA (z, t2, 2.7557319157071848E-006);
          z = __BITREPFMA (z, t2, 1.9841269841431873E-004);
          z = __BITREPFMA (z, t2, 8.3333333333331476E-003);
          z = __BITREPFMA (z, t2, 1.6666666666666669E-001);
          z = z * t2;
          z = __BITREPFMA (z, t, t);
      } else {
          z = __internal_expm1_scaled (t, -1);
          z = z + z / (__BITREPFMA (2.0, z, 1.0));
          if (t >= 7.1047586007394398e+2) {
              z = infinity;
        }
      }

      z = __internal_copysign_pos(z, x);
      return z;
  }

  #pragma acc routine seq
  double br_sinh(double x){
      return _sinh(x);
  }

// =============================================================================
//      HYPERBOLIC COSINE BASED FUNCTIONS
// =============================================================================

  #pragma acc routine seq
  double _cosh(double x_in)
  {
      double x = x_in;
      double t, z;
      z = f_abs(x);

      unsigned long long ull_inf = 0x7ff00000;
      ull_inf <<= 32;
      const double infinity = *reinterpret_cast<double*>(&ull_inf);

      union {
          int32_t i[2];
          double d;
      } xx;
      xx.d = z;

      int32_t& i = xx.i[1];

      if ((unsigned)i < (unsigned)0x408633cf) {
          z = __internal_exp_kernel(z, -2);
          t = 1. / z;
          z = __BITREPFMA(2.0, z, 0.125 * t);
      } else {
          if (z > 0.0) x = infinity;
          z = x + x;
      }

      return z;
  }  

  #pragma acc routine seq
  double br_cosh(double x){
      return _cosh(x);
  }

// =============================================================================
//       HYPERBOLIC TANGENT BASED FUNCTIONS
// =============================================================================

  #pragma acc routine seq
  double _tanh(double x_in)
  {
    double x = x_in;
    double t;
    t = f_abs(x);
    if (t >= 0.55) {
      double s;
      s = 1. / (__internal_old_exp_kernel (2.0 * t, 0) + 1.0);
      s = __BITREPFMA (2.0, -s, 1.0);
      if (t > 350.0) {
        s = 1.0;       /* overflow -> 1.0 */
      }
      x = __internal_copysign_pos(s, x);
    } else {
      double x2;
      x2 = x * x;
      t = 5.102147717274194E-005;
      t = __BITREPFMA (t, x2, -2.103023983278533E-004);
      t = __BITREPFMA (t, x2,  5.791370145050539E-004);
      t = __BITREPFMA (t, x2, -1.453216755611004E-003);
      t = __BITREPFMA (t, x2,  3.591719696944118E-003);
      t = __BITREPFMA (t, x2, -8.863194503940334E-003);
      t = __BITREPFMA (t, x2,  2.186948597477980E-002);
      t = __BITREPFMA (t, x2, -5.396825387607743E-002);
      t = __BITREPFMA (t, x2,  1.333333333316870E-001);
      t = __BITREPFMA (t, x2, -3.333333333333232E-001);
      t = t * x2;
      t = __BITREPFMA (t, x, x);
      x = __internal_copysign_pos(t, x);
    }
    return x;
  }

  #pragma acc routine seq
  double br_tanh(double x){
      return _tanh(x);
  }

/********************************
 * INVERSE HIPERBOLIC FUNCTIONS *
 *******************************/

// =============================================================================
//     INVESRE HYPERBOLIC SINE BASED FUNCTIONS
// =============================================================================

  #pragma acc routine seq
  double __internal_atanh_kernel (double a_1, double a_2)
  {
      double a, a2, t;

      a = a_1 + a_2;
      a2 = a * a;
      t = 7.597322383488143E-002/65536.0;
      t = __BITREPFMA (t, a2, 6.457518383364042E-002/16384.0);          
      t = __BITREPFMA (t, a2, 7.705685707267146E-002/4096.0);
      t = __BITREPFMA (t, a2, 9.090417561104036E-002/1024.0);
      t = __BITREPFMA (t, a2, 1.111112158368149E-001/256.0);
      t = __BITREPFMA (t, a2, 1.428571416261528E-001/64.0);
      t = __BITREPFMA (t, a2, 2.000000000069858E-001/16.0);
      t = __BITREPFMA (t, a2, 3.333333333333198E-001/4.0);
      t = t * a2;
      t = __BITREPFMA (t, a, a_2);
      t = t + a_1;
      return t;
  }

  #pragma acc routine seq
  double br_log1p(double x)
  {
      double t;
      union {
          int32_t i[2];
          double d;
      } xx;
      xx.d = x;
      
      int i = xx.i[1];
      if (((unsigned)i < (unsigned)0x3fe55555) || ((int)i < (int)0xbfd99999)) {
          /* Compute log2(x+1) = 2*atanh(x/(x+2)) */
          t = x + 2.0;
          t = x / t;
          t = -x * t;
          t = __internal_atanh_kernel(x, t);
      } else {
          t = _log(x + 1.);
      }
      return t;
  }


  #pragma acc routine seq
  double _asinh(double x)
  {
    double fx, t;
    fx = f_abs(x);

    union {
        int32_t i[2];
        double d;
    } fxx;
    fxx.d = fx;

    if (fxx.i[1] >= 0x5ff00000) { /* prevent intermediate underflow */
      t = 6.9314718055994529e-1 + _log(fx);
    } else {
      t = fx * fx;
      //t = 1.0;
      t = br_log1p(fx + t / (1.0 + std::sqrt(1.0 + t)));
    }
    return __internal_copysign_pos(t, x);  
  }

  #pragma acc routine seq
  double br_asinh(double x){
      return _asinh(x);
  }

// =============================================================================
//     INVERSE HYPERBOLIC COSINE BASED FUNCTIONS
// =============================================================================

  #pragma acc routine seq
  double _acosh(double x)
  {
    double t;
    t = x - 1.0;
    if (f_abs(t) > 4503599627370496.0) {
      // for large a, acosh = log(2*a) 
      t = 6.9314718055994529e-1 + _log(x);
    } else {
      t = t + std::sqrt(__BITREPFMA(x, t, t));
      t = br_log1p(t);
    }
    return t;
  }

  #pragma acc routine seq
  double br_acosh(double x){
      return _acosh(x);
  }

// =============================================================================
//     INVERSE HYPERBOLIC TANGENT BASED FUNCTIONS
// =============================================================================

  #pragma acc routine seq
  double _atanh(double x)
  {
    double fx, t;
    fx = f_abs(x);

    union {
        int32_t i[2];
        double d;
    } xx;
    xx.d = x;

    t = (2.0 * fx) / (1.0 - fx);
    t = 0.5 * br_log1p(t);
    if (xx.i[1] < 0) {
      t = -t;
    }
    return t;
  }

  #pragma acc routine seq
  double br_atanh(double x){
    return _atanh(x);
  }
//}

// The C interface for ISO binding C

//double br_pow(double x, double y) { return bitrep::pow(x, y); }

//double br_log(double x) { return bitrep::log(x); }
//double br_exp(double x) { return bitrep::exp(x); }

//double br_sinh(double x) { return bitrep::sinh(x); }
//double br_tanh(double x) { return bitrep::tanh(x); }

//double br_acos(double x) { return bitrep::acos(x); }
//double br_asin(double x) { return bitrep::asin(x); }
//double br_atan(double x) { return bitrep::atan(x); }

//double br_cos(double x) { return bitrep::cos(x); }
//double br_sin(double x) { return bitrep::sin(x); }
//double br_tan(double x) { return bitrep::tan(x); }
