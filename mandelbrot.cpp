#include "common.h"

#ifdef CUDA
#include "cuda.h"
#else
#include "cpu.h"
#endif

void updateScreen(int len, int gen, int saved) {
   printf("\rgenerated: %*d / %d\tsaved: %*d / %d", len, gen, DEPTH, len, saved, DEPTH);
   fflush(stdout);
}

int main(int argc, char* argv[]) {
   int run = findCurrentRun();
   int len = ceil(log10(double (DEPTH)));

   data_t startX = -1.50;
	data_t startY = -1.00;
	data_t resolution = INITIAL_RESOLUTION;
	
	uint32_t iters[WIDTH * HEIGHT];

   printf("\n");
   updateScreen(len, 0, 0);
   for (int i = 0; i < DEPTH; i++) {
	   Mandelbrot(startX, startY, resolution, iters);
      updateScreen(len, i + 1, i);

		saveImage(run, len, i, iters);
      updateScreen(len, i + 1, i + 1);

		findPath(iters, &startX, &startY, &resolution);
	}

   printf("\n");
	
   return 0;
}
