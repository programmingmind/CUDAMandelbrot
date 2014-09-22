CC=g++-4.9
NVARCH=-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
NVFLAGS=-O3 -G -g -c $(NVARCH)
CCFLAGS=-O3 -g
LDFLAGS=-L/usr/local/cuda/lib64 -lcuda -lcudart

COMMON=mandelbrot.cpp common.cpp common.h

all: cpu cuda tests div

cpu: double longdoub quad
	# all cpu versions made

cuda: $(COMMON) cuda.cu datatypes.cu
	nvcc $(NVFLAGS) -DCUDA -dc cuda.cu datatypes.cu -lcudadevrt --relocatable-device-code true
	nvcc $(NVARCH) -dlink cuda.o datatypes.o -o dlink.o
	$(CC) $(CCFLAGS) -DCUDA -o cuda cuda.o datatypes.o dlink.o $(COMMON) $(LDFLAGS) -lcudadevrt

double: $(COMMON) cpu.cpp
	$(CC) $(CCFLAGS) -o double $^ -pthread

longdoub: $(COMMON) cpu.cpp
	$(CC) $(CCFLAGS) -DLONGDOUB -o longdoub $^ -pthread

quad: $(COMMON) cpu.cpp
	$(CC) $(CCFLAGS) -DQUAD -o quad $^ -lquadmath -pthread

tests: tests.cpp datatypes.cu
	nvcc $(NVFLAGS) -o datatypes.o datatypes.cu
	$(CC) $(CCFLAGS) -o tests datatypes.o tests.cpp $(LDFLAGS)

div: fourierDivision.cpp
	$(CC) $(CCFLAGS) -o fd $^

clean:
	rm -f cuda.o datatypes.o dlink.o cuda double longdoub quad tests fd
