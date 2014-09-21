#ifndef CPU_H
#define CPU_H

#include <inttypes.h>


#ifdef QUAD
#include <quadmath.h>
typedef __float128 data_t;
#else
   #ifdef LONGDOUB
   typedef long double data_t;
   #else
   typedef double data_t;
   #endif
#endif

void Mandelbrot(data_t x, data_t y, data_t resolution, uint32_t *iters);

#endif