#include "cuda.h"
#include "common.h"

#define cudaSafe(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort)
         exit(code);
   }
}

__global__ void iterate(data_t startX, data_t startY, data_t resolution, uint32_t *iters) {
   unsigned int yNdx = blockIdx.y * blockDim.y + threadIdx.y;
   unsigned int xNdx = blockIdx.x * blockDim.x + threadIdx.x;

   startX.clearDevice();
   startY.clearDevice();
   resolution.clearDevice();

   if (xNdx < WIDTH && yNdx < HEIGHT) {
      uint32_t it=0;	
      data_t x0 = startX + ((resolution * xNdx) / WIDTH);
      data_t y0 = startY + ((resolution * yNdx) / HEIGHT);
      data_t x = x0, y = y0;
      data_t xSqr((unsigned int) 0), ySqr((unsigned int) 0);

      while (((xSqr = x*x) + (ySqr = y*y) <= 4) && (it < MAX)) {
         y = x*y*((unsigned int) 2) + y0;
         x = xSqr - ySqr + x0;
         it++;
      }

      iters[yNdx * WIDTH + xNdx] =  it;
   }
}

void Mandelbrot(data_t x, data_t y, data_t resolution, uint32_t *iters, bool first) {
   if (first) {
      initializeHeap();
   }

   uint32_t *cuda;
   int size = WIDTH * HEIGHT * sizeof(uint32_t);
   cudaSafe(cudaMalloc(&cuda, size));

   dim3 dimGrid(1 + (WIDTH - 1)/BLOCK_LEN, 1 + (HEIGHT - 1)/BLOCK_LEN);
   dim3 dimBlock(BLOCK_LEN, BLOCK_LEN);

   data_t dX = x.toDevice();
   data_t dY = y.toDevice();
   data_t dR = resolution.toDevice();

   cudaDeviceSetLimit(cudaLimitStackSize, 1024*2);
   cudaDeviceSetLimit(cudaLimitMallocHeapSize, 536870912);
   iterate<<<dimGrid, dimBlock>>>(dX, dY, dR, cuda);
   cudaSafe(cudaPeekAtLastError());
   cudaSafe(cudaDeviceSynchronize());

   dX.deviceFree();
   dY.deviceFree();
   dR.deviceFree();

   cudaSafe(cudaMemcpy(iters, cuda, size, cudaMemcpyDeviceToHost));
   cudaSafe(cudaFree(cuda));
}
