#include "transc_fct.h"

namespace mf
{
  union udouble {
    double d;
    unsigned ii[2];
    unsigned long int li;
    unsigned long long u;
  };
  
  int __double2hiint(double d) {
    udouble ud;
    ud.d = d;

    unsigned long long high = (ud.u >> 32);
    return (int) high;
  }

  int __double2loint(double d) {
    udouble ud;
    ud.d = d;
    unsigned long long mask = 0x00000000 | 0xFFFFFFFF;
    unsigned long long low = ud.u & mask;

    return (int) low;
  }

  double __hiloint2double(int high, int low) {
    unsigned h = high;
    unsigned l = low;

    unsigned long long uber = h;
    uber <<= 32;
    uber |= l;

    udouble u;
    u.u = uber;
    return u.d;
  }

  void getExpoMant(double a, double* EM) {
    const double two_to_54 = 18014398509481984.0;

    int ihi = __double2hiint(a);
    int ilo = __double2loint(a);
    int e = -1023;

    /* normalize denormals */
    if ((unsigned) ihi < (unsigned) 0x00100000) {
      a = a*two_to_54;
      e -= 54;
      ihi = __double2hiint(a);
      ilo = __double2loint(a);
    }

    /*
     * a = m * 2^e.
     * m <= sqrt(2): log2(a) = log2(m) + e.
     * m > sqrt(2): log2(a) = log2(m/2) + (e+1)
     */
    e += (ihi >> 20);
    ihi = (ihi & 0x800fffff) | 0x3ff00000;

    double m = __hiloint2double(ihi, ilo);
    if ((unsigned) ihi > (unsigned) 0x3ff6a09e) {
      m = 0.5*m;
      e = e + 1;
    }

    EM[0] = (double) e;
    EM[1] = m;
  }
}

