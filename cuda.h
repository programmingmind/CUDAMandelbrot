#ifndef CUDA_H
#define CUDA_H

#include "common.h"

#define BLOCK_LEN 32

typedef double data_t;

void Mandelbrot(data_t x, data_t y, data_t resolution, uint32_t *iters);

#endif