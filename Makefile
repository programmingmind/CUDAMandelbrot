CC=g++
NVARCH=-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50
NVFLAGS=-O3 -lineinfo -g -c $(NVARCH)
CCFLAGS=-O3 -g
LDFLAGS=-L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcuda -lcudart -Wl,-rpath,/usr/local/cuda/lib

COMMON=mandelbrot.cpp common.cpp

all: cpu cuda tests bftests div

cpu: double longdoub quad
	# all cpu versions made

cuda: $(COMMON) cuda.cu BigFloat.cu common.h
	nvcc $(NVFLAGS) -DCUDA -dc cuda.cu BigFloat.cu -lcudadevrt --relocatable-device-code true
	nvcc $(NVARCH) -dlink cuda.o BigFloat.o -o dlink.o
	$(CC) $(CCFLAGS) -DCUDA -o cuda cuda.o BigFloat.o dlink.o $(COMMON) $(LDFLAGS) -lcudadevrt

double: $(COMMON) cpu.cpp common.h
	$(CC) $(CCFLAGS) -o double $^ -pthread

longdoub: $(COMMON) cpu.cpp common.h
	$(CC) $(CCFLAGS) -DLONGDOUB -o longdoub $^ -pthread

quad: $(COMMON) cpu.cpp common.h
	$(CC) $(CCFLAGS) -DQUAD -o quad $^ -lquadmath -pthread

tests: tests.cpp BigFloat.cu
	nvcc $(NVFLAGS) -o BigFloat.o BigFloat.cu
	$(CC) $(CCFLAGS) -o tests BigFloat.o tests.cpp $(LDFLAGS)

bftests: BigFloatTests.cpp BigFloat.cu BigFloat.h
	nvcc $(NVFLAGS) -o BigFloat.o BigFloat.cu
	$(CC) $(CCFLAGS) -o bftests BigFloat.o BigFloatTests.cpp $(LDFLAGS)

div: fourierDivision.cpp
	$(CC) $(CCFLAGS) -o fd $^

clean:
	rm -f BigFloat.o cuda.o dlink.o bftests cuda double longdoub quad tests fd
