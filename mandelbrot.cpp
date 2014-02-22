#include "common.h"

#ifdef CUDA
#include "cuda.h"
#else
#include "cpu.h"
#endif

int main(int argc, char* argv[]) {
   data_t startX = -1.50;
	data_t startY = -1.00;
	data_t resolution = INITIAL_RESOLUTION;
	
	uint32_t iters[WIDTH * HEIGHT];

	char file[] = "a.bmp";
   for (int i = 0; i < DEPTH; i++, file[0]++) {
	   Mandelbrot(startX, startY, resolution, iters);
		saveImage(file, iters);
		findPath(iters, &startX, &startY, &resolution);
	}
	
   return 0;
}
