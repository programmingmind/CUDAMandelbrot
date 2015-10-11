#ifndef CUDA_H
#define CUDA_H

#include "BigFloat.h"

typedef BigFloat data_t;

#define BLOCK_LEN 8
#define DBL_LIMIT 15
#define DEPTH 100

void Mandelbrot(data_t x, data_t y, data_t resolution, uint32_t *iters);
void DoubleMandelbrot(double startX, double startY, double resolution, uint32_t *iters);

#endif
