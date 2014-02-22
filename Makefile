NVFLAGS=-O3 -g -c -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 
CCFLAGS=-O3
LDFLAGS=-L/usr/local/cuda/lib64 -lcuda -lcudart

COMMON=mandelbrot.cpp common.cpp common.h

all: cpu cuda tests

cpu: $(COMMON) cpu.cpp
	g++ $(CCFLAGS) -o MandelbrotCPU $^

cuda: $(COMMON) cuda.cu
	nvcc $(NVFLAGS) -o cuda.o cuda.cu
	g++ $(CFLAGS) -DCUDA -o MandelbrotGPU cuda.o $(COMMON) $(LDFLAGS)

tests: tests.cpp datatypes.h
	g++ $(CFLAGS) -o tests $^ -lm

clean:
	rm -f cuda.o MandelbrotCPU MandelbrotGPU tests