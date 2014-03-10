#ifndef CUDA_H
#define CUDA_H

#include "datatypes.h"

typedef Decimal data_t;

#define BLOCK_LEN 32

void Mandelbrot(data_t x, data_t y, data_t resolution, uint32_t *iters);

#endif
