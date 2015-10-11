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

   int64_t xPos = 0, yPos = 0;
	
	uint32_t iters[WIDTH * HEIGHT];

	data_t initX;
	data_t initY;

	initX = INITIAL_X;
	initY = INITIAL_Y;

   printf("\n");
   updateScreen(len, wLen, hLen, 0, xNdx, yNdx);

   int i = 0;
#ifdef CUDA
   for (; i < DBL_LIMIT; i++) {
      double resolution = INITIAL_RESOLUTION;
      resolution /= (1 << i);

      double startX = INITIAL_X + (resolution * xPos) / WIDTH;
      double startY = INITIAL_Y + (resolution * yPos) / HEIGHT;

      cout << xPos << "\t" << yPos << "\t" << startX << "\t" << startY << endl;

	   DoubleMandelbrot(startX, startY, resolution, iters);
		saveImage(run, len, i, iters);
		findPath(iters, &xNdx, &yNdx);

		xPos += xNdx;
		yPos += yNdx;

		xPos *= 2;
		yPos *= 2;

		xPos -= WIDTH/2;
		yPos -= HEIGHT/2;

      updateScreen(len, wLen, hLen, i + 1, xNdx, yNdx);
	}
#endif

   for (; i < DEPTH; i++) {
      data_t resolution;
      resolution = INITIAL_RESOLUTION;

#ifdef CUDA
      resolution >>= i;
#else
      resolution /= ((uint64_t)1) << i;
#endif

#ifdef CUDA
      data_t startX = (resolution * xPos) >> DIM_POWER;
      data_t startY = (resolution * yPos) >> DIM_POWER;
#else
      data_t startX = (resolution * xPos) / WIDTH;
      data_t startY = (resolution * yPos) / HEIGHT;
#endif

      startX += initX;
      startY += initY;

	   Mandelbrot(startX, startY, resolution, iters);
		saveImage(run, len, i, iters);
		findPath(iters, &xNdx, &yNdx);

		xPos += xNdx;
		yPos += yNdx;

		xPos *= 2;
		yPos *= 2;

		xPos -= WIDTH/2;
		yPos -= HEIGHT/2;

      updateScreen(len, wLen, hLen, i + 1, xNdx, yNdx);
	}
   printf("\n");
	
   return 0;
}
