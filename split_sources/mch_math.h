#pragma once

#include <iostream>
#include <bitset>
#include <stdio.h>

#define MCH_INF hiloint2double(0x7ff00000, 0x00000000)
#define MCH_NAN hiloint2double(0xfff80000, 0x00000000)

#define MCH_ZERO 0.0
#define MCH_L2E 1.4426950408889634e+0
#define MCH_LN2_HI 6.9314718055994529e-1
#define MCH_LN2_LO 2.3190468138462996e-17
#define MCH_LN2 6.9314718055994529e-1
#define MCH_TWO_TO_54 18014398509481984.0

#define MCH_INT_MAX 2147483647
#define MCH_INT_MIN (-MCH_INT_MAX - 1)

union udouble {
  double d;
  unsigned i[2];
  unsigned long int li;
  unsigned long long u;
};

// hi -> i[0]
// lo -> i[1]

union uinteger {
  int i;
  unsigned long long u;
};

struct doubleXY {
    double x;
    double y;
};


void printDd(double d);

void printIi(int i);

void print_int(int a);

void print_double(double a);

void print_ull(unsigned long long a);

double hiloint2double(int hi, int lo);

double hiloint2double(int low, int high);

int mch_isnand(double a);

int double2hiint(double combined);

int double2loint(double d);

double mch_rint(double a);

int sabs(int i);

int mch_fabs(double i);

double mch_exp(double a);

double mch_exp_kernel(double a, int scale);

double mch_exp_poly(double a);

double mch_exp_scale(double a, int i);

double mch_trunc(double x);

int mch_isinfd(double a);

doubleXY mch_ddadd_xgty(doubleXY x, doubleXY y);

doubleXY mch_ddmul(doubleXY x, doubleXY y);

double mch_pow(double a, double b);

doubleXY mch_log_ext_prec(double a);

double mch_accurate_pow(double a, double b);

double mch_neg(double a);

