default: MandelbrotCPU.cpp
	g++ -o MandelbrotCPU MandelbrotCPU.cpp -O3

cpu: MandelbrotCPU.cpp
	g++ -o MandelbrotCPU MandelbrotCPU.cpp -O3
