namespace mf {

// ----------------------------------------------------------
// HELPER FUNCTION TO MANIPULATE DOUBLES
// ----------------------------------------------------------

    int __double2hiint(double d);

    int __double2loint(double d);

    double __hiloint2double(int high, int low);

    void getExpoMant(double a, double* EM);

// ----------------------------------------------------------
// END OF HELPERS
// ----------------------------------------------------------

    /**
    * Function computing the natural logarithm
    */
    template<typename T>
    T friendly_log(const T x)
    {
      double a = (double) x;
      const double ln2_hi = 6.9314718055994529e-1;
      const double ln2_lo = 2.3190468138462996e-17;

#ifdef __CUDA_BACKEND__
      unsigned long long ull_inf = 0x7ff00000;
      ull_inf <<= 32;
      const double infinity = *reinterpret_cast<double*>(&ull_inf);

      unsigned long long ull_nan = 0x7ff00000;
      ull_nan <<= 32;
      const double notanumber = *reinterpret_cast<double*>(&ull_nan);
#else
      const double infinity = std::numeric_limits<double>::infinity();
      const double notanumber = std::numeric_limits<double>::quiet_NaN();
#endif

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

      return (T) q;
    }

    /**
    * pow (see math.h)
    *
    * x to the power of y
    */
    template<typename T>
    T pow(const T x, const T y)
    {
      T z = friendly_log(x)*y;
      return z;
    }

    /**
    * Function computing the natural logarithm
    */
    template<typename T>
    T log(const T a)
    {
      return friendly_log(a);
    }

    /**
    * Function computing the exponential
    */
    template<typename T>
    T exp(const T x)
    {
      T y = x;
      int n = 10;
      T e = 1 + y;
      T f = 1;
      for (int i = 2; i < n; i++)
      {
        y = y * y;
        f = f * i;
        e = e * (y/f);
      }

      T z = e;

      return z;
    }
}

