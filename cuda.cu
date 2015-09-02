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

__global__ void iterate(BigFloat startX, BigFloat startY, BigFloat resolution, uint32_t *iters) {
   unsigned int yNdx = blockIdx.x >> 1;
   unsigned int xNdx = threadIdx.x | ((blockIdx.x & 1) << 7);

   uint32_t it=0;
   BigFloat x0;
   BigFloat y0;
   BigFloat x;
   BigFloat y;
   BigFloat xSqr;
   BigFloat ySqr;
   BigFloat temp;
   BigFloat multTemp;

   if (xNdx < WIDTH && yNdx < HEIGHT) {
      add(&startX, shiftR(mult(&resolution, init(&temp, xNdx), &x, &multTemp), DIM_POWER), &x0);
      add(&startY, shiftR(mult(&resolution, init(&temp, yNdx), &y, &multTemp), DIM_POWER), &y0);
      assign(&x, &x0);
      assign(&y, &y0);

      while (it < MAX && base2Cmp(add(mult(&x, &x, &xSqr, &multTemp), mult(&y, &y, &ySqr, &multTemp), &temp), 2) != GT) {
         (void)add(&y0, shiftL(mult(&x, &y, &temp, &multTemp), 1), &y);
         (void)add(sub(&xSqr, &ySqr, &temp), &x0, &x);
         it++;
      }

      iters[yNdx * WIDTH + xNdx] =  it;
   }
}

void Mandelbrot(data_t x, data_t y, data_t resolution, uint32_t *iters) {
   uint32_t *cuda;
   int size = WIDTH * HEIGHT * sizeof(uint32_t);
   cudaSafe(cudaMalloc(&cuda, size));

   cudaDeviceSetLimit(cudaLimitStackSize, 2048 * 16);
   iterate<<<1024, 256>>>(x, y, resolution, cuda);
   cudaSafe(cudaPeekAtLastError());
   cudaSafe(cudaDeviceSynchronize());

   cudaSafe(cudaMemcpy(iters, cuda, size, cudaMemcpyDeviceToHost));
   cudaSafe(cudaFree(cuda));
}
