NVARCH=-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
NVFLAGS=-O3 -G -g -c $(NVARCH)
CCFLAGS=-O3 -g
LDFLAGS=-L/usr/local/cuda/lib64 -lcuda -lcudart

COMMON=mandelbrot.cpp common.cpp common.h

all: cpu cuda tests div

cpu: $(COMMON) cpu.cpp
	g++ $(CCFLAGS) -o MandelbrotCPU $^

cuda: $(COMMON) cuda.cu datatypes.cu
	nvcc $(NVFLAGS) -DCUDA -dc cuda.cu datatypes.cu -lcudadevrt --relocatable-device-code true
	nvcc $(NVARCH) -dlink cuda.o datatypes.o -o dlink.o
	g++ $(CCFLAGS) -DCUDA -o MandelbrotGPU cuda.o datatypes.o dlink.o $(COMMON) $(LDFLAGS) -lcudadevrt

tests: tests.cpp datatypes.cu
	nvcc $(NVFLAGS) -o datatypes.o datatypes.cu
	g++ $(CCFLAGS) -o tests datatypes.o tests.cpp $(LDFLAGS)

div: fourierDivision.cpp
	g++ $(CCFLAGS) -o fd $^

clean:
	rm -f cuda.o datatypes.o dlink.o MandelbrotCPU MandelbrotGPU tests fd
