#include "mch_cuda_math.h"
#include <math_constants.h>
#include <math.h>

/*
union double2 {
  double x;
  double y;
}
*/

__device__ double __mch_cuda_pow(double a, double b) {
  double t;
  int ahi, bhi, thi, bIsOddInteger;
  
  ahi = __double2hiint (a);
  bhi = __double2hiint (b);
  bIsOddInteger = fabs (b - (2.0 * trunc (0.5 * b))) == 1.0;
  if ((a == 1.0) || (b == 0.0)) {
    t = 1.0;
  } else if (isnan (a) || isnan (b)) {
    t = a + b;
  } else if (isinf (b)) {
    thi = 0;
    if (fabs(a) > 1.0) thi = 0x7ff00000;
    if (bhi < 0) thi = thi ^ 0x7ff00000;
    if (a == -1.0) thi = 0x3ff00000;
    t = __hiloint2double (thi, 0);
  } else if (isinf (a)) {
    thi = 0;
    if (bhi >= 0) thi = 0x7ff00000;
    if ((ahi < 0) && bIsOddInteger) thi = thi ^ 0x80000000;
    t = __hiloint2double (thi, 0);
  } else if (a == 0.0) {
    thi = 0;
    if (bIsOddInteger) thi = ahi;
    if (bhi < 0) thi = thi | 0x7ff00000;
    t = __hiloint2double (thi, 0);
  } else if ((ahi < 0) && (b != trunc (b))) {
    t = CUDART_NAN;
  } else {
    t = __mch__accurate_pow (fabs (a), b);
    if ((ahi < 0) && bIsOddInteger) {
      t = __mch__neg (t); 
    }
  }
  return t;
}

__device__ double __mch__fma_rn(double a, double b, double c) {
  return a*b+c;
}

__device__ double __mch__accurate_pow(double a, double b)
{
  double2 loga;
  double2 prod;
  double t_hi, t_lo;
  double tmp;
  double e;

  /* compute log(a) in double-double format*/
  loga = __mch__log_ext_prec(a);

  /* prevent overflow during extended precision multiply */
  if (fabs(b) > 1e304) b *= 1.220703125e-4;
  /* compute b * log(a) in double-double format */
  t_hi = __dmul_rn (loga.y, b);   /* prevent FMA-merging */
  t_lo = __mch__fma_rn (loga.y, b, -t_hi);
  t_lo = __mch__fma_rn (loga.x, b, t_lo);
  prod.y = e = t_hi + t_lo;
  prod.x = (t_hi - e) + t_lo;

  /* compute pow(a,b) = exp(b*log(a)) */
  tmp = exp(prod.y);
  /* prevent -INF + INF = NaN */
  if (!isinf(tmp)) {
    /* if prod.x is much smaller than prod.y, then exp(prod.y + prod.x) ~= 
     * exp(prod.y) + prod.x * exp(prod.y) 
     */
    tmp = __mch__fma_rn (tmp, prod.x, tmp);
  }
  return tmp;
}

__device__ double2 __mch__log_ext_prec(double a)
{
  double2 res;
  double2 qq, cc, uu, tt;
  double f, g, u, v, q, ulo, tmp, m;
  int ilo, ihi, expo;

  ihi = __double2hiint(a);
  ilo = __double2loint(a);
  expo = (ihi >> 20) & 0x7ff;
  /* convert denormals to normals for computation of log(a) */
  if (expo == 0) {
    a *= CUDART_TWO_TO_54;
    ihi = __double2hiint(a);
    ilo = __double2loint(a);
    expo = (ihi >> 20) & 0x7ff;
    expo -= 54;
  }  
  expo -= 1023;
  /* log(a) = log(m*2^expo) = 
     log(m) + log(2)*expo, if m < sqrt(2), 
     log(m*0.5) + log(2)*(expo+1), if m >= sqrt(2)
  */
  ihi = (ihi & 0x800fffff) | 0x3ff00000;
  m = __hiloint2double (ihi, ilo);
  if ((unsigned)ihi > (unsigned)0x3ff6a09e) {
    m = __mch__half(m);
    expo = expo + 1;
  }
  /* compute log(m) with extended precision using an algorithm derived from 
   * P.T.P. Tang, "Table Driven Implementation of the Logarithm Function", 
   * TOMS, Vol. 16., No. 4, December 1990, pp. 378-400. A modified polynomial 
   * approximation to atanh(x) on the interval [-0.1716, 0.1716] is utilized.
   */
  f = m - 1.0;
  g = m + 1.0;
  g = __mch__fast_rcp(g);
  u = f * g;
  u = u + u;  
  /* u = 2.0 * (m - 1.0) / (m + 1.0) */
  v = u * u;
  q =                 6.6253631649203309E-2/65536.0;
  q = __mch__fma_rn (q, v, 6.6250935587260612E-2/16384.0);
  q = __mch__fma_rn (q, v, 7.6935437806732829E-2/4096.0);
  q = __mch__fma_rn (q, v, 9.0908878711093280E-2/1024.0);
  q = __mch__fma_rn (q, v, 1.1111111322892790E-1/256.0);
  q = __mch__fma_rn (q, v, 1.4285714284546502E-1/64.0);
  q = __mch__fma_rn (q, v, 2.0000000000003113E-1/16.0);
  q = q * v;
  /* u + ulo = 2.0 * (m - 1.0) / (m + 1.0) to more than double precision */
  tmp = 2.0 * (f - u);
  tmp = __mch__fma_rn (-u, f, tmp); // tmp = remainder of division
  ulo = g * tmp;               // less significand quotient bits
  /* switch to double-double at this point */
  qq.y = q;
  qq.x = 0.0;
  uu.y = u;
  uu.x = ulo;
  cc.y =  3.3333333333333331E-1/4.0;
  cc.x = -9.8201492846582465E-18/4.0;
  qq = __mch__ddadd_xgty (cc, qq);
  /* compute log(m) in double-double format */
  qq = __mch__ddmul(qq, uu);
  qq = __mch__ddmul(qq, uu);
  qq = __mch__ddmul(qq, uu);
  uu = __mch__ddadd_xgty (uu, qq);
  u   = uu.y;
  ulo = uu.x;
  /* log(2)*expo in double-double format */
  tt.y = __dmul_rn(expo, 6.9314718055966296e-01); /* multiplication is exact */
  tt.x = __dmul_rn(expo, 2.8235290563031577e-13);
  /* log(a) = log(m) + log(2)*expo;  if expo != 0, |log(2)*expo| > |log(m)| */
  res = __mch__ddadd_xgty (tt, uu);
  return res;
}

__device__ double2 __mch__ddmul (double2 x, double2 y)
{
  double e;
  double2 t, z;
  t.y = __dmul_rn (x.y, y.y);       /* prevent FMA-merging */
  t.x = __mch__fma_rn (x.y, y.y, -t.y);
  t.x = __mch__fma_rn (x.y, y.x, t.x);
  t.x = __mch__fma_rn (x.x, y.y, t.x);
  z.y = e = t.y + t.x;
  z.x = (t.y - e) + t.x;
  return z;
}

__device__ double2 __mch__ddadd_xgty (double2 x, double2 y)
{
  double2 z;
  double r, s, e;
  r = x.y + y.y;
  e = x.y - r;
  s = ((e + y.y) + y.x) + x.x;
  z.y = e = r + s;
  z.x = (r - e) + s;
  return z;
}


__device__ double __mch__half(double a)
{
  unsigned int ihi, ilo;
  ilo = __double2loint(a);
  ihi = __double2hiint(a);
  return __hiloint2double(ihi - 0x00100000, ilo);
}

__device__ double __mch__fast_rcp(double a) 
{ 
  double e, y; 
  float x; 
  asm ("cvt.rn.f32.f64     %0,%1;" : "=f"(x) : "d"(a)); 
  asm ("rcp.approx.ftz.f32 %0,%1;" : "=f"(x) : "f"(x)); 
  asm ("cvt.f64.f32        %0,%1;" : "=d"(y) : "f"(x)); 
  e = __mch__fma_rn (-a, y, 1.0); 
  e = __mch__fma_rn ( e, e,   e); 
  y = __mch__fma_rn ( e, y,   y); 
  return y; 
} 

__device__ double __mch__neg(double a)
{
  int hi = __double2hiint(a);
  int lo = __double2loint(a);
  return __hiloint2double(hi ^ 0x80000000, lo);
}

