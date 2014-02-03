NVFLAGS=-O3 -g -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 
CCFLAGS=-O3

cpu: MandelbrotCPU.cpp
	g++ $(CCFLAGS) -o MandelbrotCPU $^

cuda: MandelbrotGPU.cu
	nvcc $(NVFLAGS) -o MandelbrotGPU $^