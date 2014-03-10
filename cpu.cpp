#include "cpu.h"
#include "common.h"

uint32_t numIter(data_t x, data_t y) {
   uint32_t it=0;	
   data_t x0 = x;
   data_t y0 = y;
   data_t xSqr, ySqr;
	
	while (((xSqr = x*x) + (ySqr = y*y) <= 4) && (it < MAX)) {
	   y = 2*x*y + y0;
		x = xSqr - ySqr + x0;
		it++;
	}
	
	return it;
}

void Mandelbrot(data_t x, data_t y, data_t resolution, uint32_t *iters) {
   // inline numIters and remove the loops for CUDA implementation
   for (int i = 0; i < HEIGHT; i++)
      for (int j = 0; j < WIDTH; j++)
         iters[i * WIDTH + j] = numIter(x + (j * resolution / WIDTH), y + (i * resolution / HEIGHT));
}
