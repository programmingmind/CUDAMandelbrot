#ifndef CPU_H
#define CPU_H

#include <inttypes.h>

#ifdef QUAD
#include <quadmath.h>
typedef __float128 data_t;
#define DEPTH 40
#else
   #ifdef LONGDOUB
   typedef long double data_t;
   #define DEPTH 25
   #else
   typedef double data_t;
   #define DEPTH 15
   #endif
#endif

void Mandelbrot(data_t x, data_t y, data_t resolution, uint32_t *iters);

#endif