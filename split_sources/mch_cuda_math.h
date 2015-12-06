#include <math_constants.h>

const bool VERBOSE = false;

/*
union double2 {
  double x;
  double y;
}
*/

__device__ double __mch_cuda_pow(double a, double b);

__device__ double __mch__accurate_pow(double a, double b);

__device__ double2 __mch__log_ext_prec(double a);

__device__ double2 __mch__ddmul (double2 x, double2 y);

__device__ double2 __mch__ddadd_xgty (double2 x, double2 y);

__device__ double __mch__half(double a);

__device__ double __mch__fast_rcp(double a);

__device__ double __mch__neg(double a);

__device__ double __mch__fma_rn(double a, double b, double c);

__device__ int __mch_isnand(double a);

__device__ double __mch_rint(double a);

__device__ double __mch_exp_scale(double a, int i);

__device__ double __mch_exp_poly(double a);

__device__ double __mch_exp_kernel(double a, int scale);

__device__ double __mch_exp(double a);

__device__ int __mch_fabs(double i);

union SI {
  unsigned int ii;
  unsigned long int li;
  double dd;
};

__device__ void printGpuDd(double d) {
  SI num;
  num.dd = d;
  printf("[GPU] %f %lu\n", num.dd, num.li);
}

__device__ void printGpuIi(int i) {
  printf("[GPU] %i\n", i);
}

__device__ int __mch__sabs(int i) {
  if (MCH_INT_MIN == i) {
    return MCH_INT_MAX;
  } else {
    return i < 0 ? -i : i;
  }
}

__device__ double __mch_cuda_pow(double a, double b) {
  
  double t;
  int ahi, bhi, thi, bIsOddInteger;    
  
  ahi = __double2hiint (a);
  bhi = __double2hiint (b);
  
  if (VERBOSE) {
    printf("================================\n");
    printf("Pow of %f %f\n", a, b);
    printf("================================\n");
    printGpuDd(a);
    printGpuDd(b);
    printGpuIi(ahi);
    printGpuIi(bhi);
  }
  
  bIsOddInteger = __mch_fabs (b - (2.0 * trunc (0.5 * b))) == 1.0;
  if ((a == 1.0) || (b == 0.0)) {
    t = 1.0;
  } else if (isnan (a) || isnan (b)) {
    t = a + b;
  } else if (isinf (b)) {
    thi = 0;
    if (__mch_fabs(a) > 1.0) thi = 0x7ff00000;
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
    t = __mch__accurate_pow (__mch_fabs(a), b);
    if (VERBOSE) {
      printf("[GPU] ACC POW\n");
      printGpuDd(a);
      printGpuDd(__mch_fabs(a));    
      printGpuDd(t);
    }
    
    if ((ahi < 0) && bIsOddInteger) {
      t = __mch__neg (t); 
    }
  }
  
  if (VERBOSE) { 
    printGpuDd(t);
    printf("================================\n");
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
  if (__mch_fabs(b) > 1e304) b *= 1.220703125e-4;
  /* compute b * log(a) in double-double format */
  
  t_hi = __dmul_rn (loga.y, b);   /* prevent FMA-merging */
  t_lo = __mch__fma_rn (loga.y, b, -t_hi);

  if (VERBOSE) {  
    printf("[GPU] LOGA DOUBLE-DOUBLE:\n");
    printGpuDd(a);
    printf("[GPU] LOGA:\n");
    printGpuDd(loga.x);
    printGpuDd(loga.y);
    printf("[GPU] T_HI/LOW:\n");  
    printGpuDd(t_hi);  
    printGpuDd(t_lo);
  }
  
  t_lo = __mch__fma_rn (loga.x, b, t_lo);  
  
  prod.y = e = t_hi + t_lo;  
  prod.x = (t_hi - e) + t_lo;
  
  /* compute pow(a,b) = exp(b*log(a)) */
  tmp = __mch_exp(prod.y);
  if (VERBOSE) {  
    printGpuDd(t_lo);
    printf("[GPU] PROD.X/Y:\n");
    printGpuDd(prod.y);
    printGpuDd(prod.x);
    printGpuDd(tmp);
  }
  
  /* prevent -INF + INF = NaN */
  if (!isinf(tmp)) {
    /* if prod.x is much smaller than prod.y, then exp(prod.y + prod.x) ~= 
     * exp(prod.y) + prod.x * exp(prod.y) 
     */
    tmp = __mch__fma_rn (tmp, prod.x, tmp);
    if (VERBOSE) {  
      printf("[GPU] -INF+INF=NAN:\n");
      printGpuDd(prod.x);
      printGpuDd(tmp);
    }
  }
  return tmp;
}


__device__ double __mch_exp(double a) {
  double t;
  int i = __double2hiint(a);
  // We only check if we are in a specific range [-a,b] to compute 
  // the exp
  if (((unsigned) i < (unsigned) 0x40862e43) || ((int) i < (int) 0xC0874911)) {
    t = __mch_exp_kernel(a, 0);
  } 
  // Otherwise the result is a very small value, then returning 0 or
  // a very large value then returning inf
  else {
    t = (i < 0) ? CUDART_ZERO : CUDART_INF;
    if (__mch_isnand(a)) {
      t = a + a;
    }
  }
  return t;
}

__device__ double __mch_exp_kernel(double a, int scale) { 
  int i;
  double t; 
  double z;  
  
  // FMA(x,y,z) = x*y+z
  t = __mch_rint(a*MCH_L2E);
  i = (int) t;
  z = t*(-MCH_LN2_HI)+a;
  z = t*(-MCH_LN2_LO)+z;
  t = __mch_exp_poly(z);
  z = __mch_exp_scale(t, i + scale);
  return z;
}

__device__ double __mch_exp_poly(double a) {
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

__device__ double __mch_exp_scale(double a, int i) {
  unsigned int k;
  unsigned int j;
  
  if (__mch__sabs(i) < 1023) {
    k = (i << 20) + (1023 << 20);
  } else {
    k = i + 2*1023;  
    j = k/2;
    j = j << 20;
    k = (k << 20) - j;
    a = a*__hiloint2double(j, 0);
  }
  a = a*__hiloint2double(k, 0);

  return a;
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
  
  if (VERBOSE) {  
    printf("[CPU] LOG_EXT_PREC\n");
    printGpuDd(a);
    printGpuIi(ihi);
    printGpuIi(ilo);
    printGpuIi(expo);
  }  
  
  /* convert denormals to normals for computation of log(a) */
  if (expo == 0) {
    a *= CUDART_TWO_TO_54;
    ihi = __double2hiint(a);
    ilo = __double2loint(a);
    expo = (ihi >> 20) & 0x7ff;
    expo -= 54;
  }  
  if (VERBOSE) {
    printGpuIi(expo);
  }
  
  expo -= 1023;
  
  if (VERBOSE) {
    printGpuIi(expo);
  }
  
  /* log(a) = log(m*2^expo) = 
     log(m) + log(2)*expo, if m < sqrt(2), 
     log(m*0.5) + log(2)*(expo+1), if m >= sqrt(2)
  */
  ihi = (ihi & 0x800fffff) | 0x3ff00000;
  m = __hiloint2double (ihi, ilo);
  
  if (VERBOSE) {
    printGpuIi(ihi);  
    printGpuDd(m);
  }
  
  if ((unsigned)ihi > (unsigned)0x3ff6a09e) {
    m = __mch__half(m);    
    expo = expo + 1;
    if (VERBOSE) {
      printGpuDd(m);
      printGpuIi(expo);
    }
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
  t.y = x.y*y.y;       /* prevent FMA-merging */
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

__device__ int __mch_fabs(double i) {
  return i < 0 ? -i : i;
}

__device__ int __mch_isnand(double a) {
  return !(__mch_fabs(a) <= CUDART_INF);
}


__device__ double __mch_rint(double a) {
  if (a > 0) {
    return (int) (a+0.5);
  }

  if (a < 0) {
    return (int) (a-0.5);
  }
  
  return 0;
}

