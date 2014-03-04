#include "common.h"

#ifdef CUDA
#include "cuda.h"
#else
#include "cpu.h"
#endif

int main(int argc, char* argv[]) {
   int run = findCurrentRun();
   int len = ceil(log10(double (DEPTH)));

   data_t startX = -1.50;
	data_t startY = -1.00;
	data_t resolution = INITIAL_RESOLUTION;
	
	uint32_t iters[WIDTH * HEIGHT];

   for (int i = 0; i < DEPTH; i++) {
	   Mandelbrot(startX, startY, resolution, iters);
		saveImage(run, len, i, iters);
		findPath(iters, &startX, &startY, &resolution);
	}
	
   return 0;
}
