#ifndef CPU_H
#define CPU_H

#include <inttypes.h>

typedef double data_t;

void Mandelbrot(data_t x, data_t y, data_t resolution, uint32_t *iters);

#endif