#include "common.h"

void updateScreen(int len, int xLen, int yLen, int gen, int xNdx, int yNdx) {
   printf("\rgenerated: %*d / %d\tzoom: ( %*d , %*d )", len, gen, DEPTH, xLen, xNdx, yLen, yNdx);
   fflush(stdout);
}

int main(int argc, char* argv[]) {
   if (argc > 1)
      setNumThreads(atoi(argv[1]));
   
   int run = findCurrentRun();
   int len = ceil(log10(double (DEPTH)));
   int wLen = ceil(log10(double (WIDTH)));
   int hLen = ceil(log10(double (WIDTH)));

   int xNdx = WIDTH / 2, yNdx = HEIGHT / 2;

   data_t startX = -1.50;
	data_t startY = -1.00;
	data_t resolution = INITIAL_RESOLUTION;
	
	uint32_t iters[WIDTH * HEIGHT];

   printf("\n");
   updateScreen(len, wLen, hLen, 0, xNdx, yNdx);
   for (int i = 0; i < DEPTH; i++) {
	   Mandelbrot(startX, startY, resolution, iters);
		saveImage(run, len, i, iters);
		findPath(iters, &startX, &startY, &resolution, &xNdx, &yNdx);

      updateScreen(len, wLen, hLen, i + 1, xNdx, yNdx);
	}

   printf("\n");
	
   return 0;
}
