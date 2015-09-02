#ifndef CUDA_H
#define CUDA_H

#include "BigFloat.h"

typedef BigFloat data_t;

#define BLOCK_LEN 8
#define DEPTH 15

void Mandelbrot(data_t x, data_t y, data_t resolution, uint32_t *iters);

#endif
