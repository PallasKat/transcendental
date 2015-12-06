namespace gpumath {

  union uinteger {
    int i;
    unsigned long long u;
  };

  union udouble {
    double d;
    unsigned ii[2];
    unsigned long int li;
    unsigned long long u;
  };

  // =============================================================================
  // PROTOTYPE
  // =============================================================================

  __device__ double cuda_log(double x);
  __device__ int double2hiint(double d);__device__ int double2loint(double d);
  __device__ double hiloint2double(int high, int low);
  __device__ void getExpoMant(double a, double* EM);
  __device__ double uber_log(double a);

  // =============================================================================
  // IMPLEMENTATION
  // =============================================================================
  
  __device__ double cuda_log(double x) {
    return uber_log(x);
  }
  
  __device__ double cuda_pow(double x, double y) {
    return uber_log(x)*y;
  }

  // TESTED AND WORKING
  __device__ int double2hiint(double d) {
    udouble ud;
    ud.d = d;

    unsigned long long high = (ud.u >> 32);
    return (int) high;
  }

  // TESTED AND WORKING
  __device__ int double2loint(double d) {
    udouble ud;
    ud.d = d;
    unsigned long long mask = 0x00000000 | 0xFFFFFFFF;
    unsigned long long low = ud.u & mask;
    
    return (int) low;
  }

  // TESTED AND WORKING
  __device__ double hiloint2double(int high, int low) {
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
    
    int ihi = double2hiint(a);
    int ilo = double2loint(a);
    int e = -1023;
    
    /* normalize denormals */
    // 1 0000 0000 0000 0000 0000
    // The exponent is smaller than 1
    if ((unsigned) ihi < (unsigned) 0x00100000) {
      a = a*two_to_54;
      e -= 54;
      ihi = double2hiint(a);
      ilo = double2loint(a);
    }
    
    /* 
     * a = m * 2^e. 
     * m <= sqrt(2): log2(a) = log2(m) + e.
     * m > sqrt(2): log2(a) = log2(m/2) + (e+1)
     */
    e += (ihi >> 20);
    ihi = (ihi & 0x800fffff) | 0x3ff00000;
    
    double m = hiloint2double(ihi, ilo);
    if ((unsigned) ihi > (unsigned) 0x3ff6a09e) {
      m = 0.5*m;
      e = e + 1;
    }
    
    EM[0] = (double) e;
    EM[1] = m;
  }

  __device__ double uber_log(double a) {
    const double ln2_hi = 6.9314718055994529e-1;
    const double ln2_lo = 2.3190468138462996e-17;
    
    //const double infinity = std::numeric_limits<double>::infinity();
    unsigned long long ull_inf = 0x7ff00000;
    ull_inf <<= 32;
    const double infinity = *reinterpret_cast<double*>(&ull_inf);
    
    //const double notanumber = std::numeric_limits<double>::quiet_NaN();
    unsigned long long ull_nan = 0x7ff00000;
    ull_nan <<= 32;
    const double notanumber = *reinterpret_cast<double*>(&ull_nan);
    
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
    
    return q;
  }
}
