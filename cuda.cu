#include "cuda.h"

__global__ void iterate(data_t startX, data_t startY, data_t resolution, uint32_t *iters) {
   int yNdx = blockIdx.y * blockDim.y + threadIdx.y;
	int xNdx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (xNdx < WIDTH && yNdx < HEIGHT) {
      uint32_t it=0;	
      data_t x0 = startX + (xNdx * resolution / WIDTH);
      data_t y0 = startY + (yNdx * resolution / HEIGHT);
		data_t x = x0, y = y0;
      data_t xSqr, ySqr;
	   
	   while (((xSqr = x*x) + (ySqr = y*y) <= 4) && (it < MAX)) {
         y = 2*x*y + y0;
         x = xSqr - ySqr + x0;
         it++;
	   }
	   
      iters[yNdx * WIDTH + xNdx] =  it;
	}
}

void Mandelbrot(data_t x, data_t y, data_t resolution, uint32_t *iters) {
   uint32_t *cuda;
	int size = WIDTH * HEIGHT * sizeof(uint32_t);
	cudaMalloc(&cuda, size);
	
	dim3 dimGrid(1 + (WIDTH - 1)/BLOCK_LEN, 1 + (HEIGHT - 1)/BLOCK_LEN);
	dim3 dimBlock(BLOCK_LEN, BLOCK_LEN);
	
	iterate<<<dimGrid, dimBlock>>>(x, y, resolution, cuda);
	
	cudaMemcpy(iters, cuda, size, cudaMemcpyDeviceToHost);
}
