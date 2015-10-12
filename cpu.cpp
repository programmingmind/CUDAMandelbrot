#include "cpu.h"
#include "common.h"

#include <pthread.h>

typedef struct {
   data_t x;
   data_t y;
   data_t resolution;
   uint32_t *iters;
} posInfo;

static int col;
static pthread_mutex_t mut;

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

void *calcRow(void *arg) {
   data_t x = ((posInfo *) arg)->x;
   data_t y = ((posInfo *) arg)->y;
   data_t resolution = ((posInfo *) arg)->resolution;
   uint32_t *iters = ((posInfo *) arg)->iters;

   while (1) {
      pthread_mutex_lock(&mut);
      int i = col++;
      pthread_mutex_unlock(&mut);
      if (i >= HEIGHT)
         return NULL;

      for (int j = 0; j < WIDTH; j++)
         iters[i * WIDTH + j] = numIter(x + (j * resolution / WIDTH), y + (i * resolution / HEIGHT));
   }

   return NULL;
}

void Mandelbrot(data_t x, data_t y, data_t resolution, uint32_t *iters) {
   posInfo info = {x, y, resolution, iters};
   col = 0;
   pthread_mutex_init(&mut, NULL);

   int numThreads = getNumThreads();
   pthread_t *threads = (pthread_t*) malloc(numThreads * sizeof(pthread_t));

   for (int i = 0; i < numThreads; i++) {
      pthread_create(threads + i, NULL, &calcRow, &info);
   }

   for (int i = 0; i < numThreads; i++) {
      pthread_join(threads[i], NULL);
   }

   pthread_mutex_destroy(&mut);
   free(threads);
}
