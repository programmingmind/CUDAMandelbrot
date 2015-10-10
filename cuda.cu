#include "cuda.h"
#include "common.h"

#define cudaSafe(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort)
         exit(code);
   }
}

__global__ void init(BigFloat startX, BigFloat startY, BigFloat resolution, BigFloat *vals) {
   unsigned int ndx = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int yNdx, xNdx;

   BigFloat *x0;
   BigFloat *y0;
   BigFloat x;
   BigFloat y;
   BigFloat temp;
   BigFloat multTemp;

   while (ndx < WIDTH*HEIGHT) {
      xNdx = ndx & ((1<<DIM_POWER)-1);
      yNdx = ndx >> DIM_POWER;

      x0 = vals + (ndx*2+0);
      y0 = vals + (ndx*2+1);
      
      add(&startX, shiftR(mult(&resolution, init(&temp, xNdx), &x, &multTemp), DIM_POWER), x0);
      add(&startY, shiftR(mult(&resolution, init(&temp, yNdx), &y, &multTemp), DIM_POWER), y0);

      ndx += gridDim.x * blockDim.x;
   }
}

__global__ void iterate(BigFloat *originals, BigFloat *vals, uint32_t *iters, uint32_t lastLimit, uint32_t limit) {
   unsigned int ndx = blockIdx.x * blockDim.x + threadIdx.x;

   uint32_t it;
   BigFloat x0;
   BigFloat y0;
   BigFloat x;
   BigFloat y;
   BigFloat xSqr;
   BigFloat ySqr;
   BigFloat temp;
   BigFloat multTemp;

   while (ndx < WIDTH*HEIGHT) {
      x0 = originals[ndx*2+0];
      y0 = originals[ndx*2+1];
      x  =      vals[ndx*2+0];
      y  =      vals[ndx*2+1];

      if (lastLimit) {
         it = iters[ndx];
         if (it < lastLimit) {
            continue;
         }
      } else {
         it = 0;
         x = x0;
         y = y0;
      }

      while (it < limit && base2Cmp(add(mult(&x, &x, &xSqr, &multTemp), mult(&y, &y, &ySqr, &multTemp), &temp), 2) != GT) {
         (void)add(&y0, shiftL(mult(&x, &y, &temp, &multTemp), 1), &y);
         (void)add(sub(&xSqr, &ySqr, &temp), &x0, &x);
         it++;
      }

      iters[ndx] =  it;
      vals[ndx*2+0] = x;
      vals[ndx*2+1] = y;

      ndx += gridDim.x * blockDim.x;
   }
}

void Mandelbrot(data_t x, data_t y, data_t resolution, uint32_t *iters) {
   const int numBlocks  = 2048;
   const int numThreads = 64;
   BigFloat *originals, *vals;
   uint32_t *cuda;
   const int size = WIDTH * HEIGHT * sizeof(uint32_t);

   cudaSafe(cudaMalloc(&cuda, size));
   cudaSafe(cudaMalloc(&originals, WIDTH*HEIGHT*2*sizeof(BigFloat)));
   cudaSafe(cudaMalloc(&vals,      WIDTH*HEIGHT*2*sizeof(BigFloat)));

   cudaDeviceSetLimit(cudaLimitStackSize, 8*sizeof(BigFloat) + 1024);
   init<<<numBlocks, numThreads>>>(x, y, resolution, originals);
   cudaSafe(cudaPeekAtLastError());
   cudaSafe(cudaDeviceSynchronize());

   int lastLimit = 0;
   for (int limit = ITER_STEP; limit <= MAX; limit += ITER_STEP) {
      std::cout<<limit<<std::endl;
      iterate<<<numBlocks, numThreads>>>(originals, vals, cuda, lastLimit, limit);
      cudaSafe(cudaPeekAtLastError());
      cudaSafe(cudaDeviceSynchronize());
   }

   cudaSafe(cudaMemcpy(iters, cuda, size, cudaMemcpyDeviceToHost));
   cudaSafe(cudaFree(cuda));
   cudaSafe(cudaFree(originals));
   cudaSafe(cudaFree(vals));
}

