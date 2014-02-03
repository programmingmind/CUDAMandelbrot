#include <inttypes.h>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>

using namespace std;

#define WIDTH 480
#define HEIGHT WIDTH
#define INITIAL_RESOLUTION 2.0

#define MAX 65536
#define DEPTH 15

#define STD_DEV_RADIUS 5
#define RANDOM_POOL_SIZE (1 << 2)

typedef struct {
   double variance;
	double mean;
   int xNdx;
	int yNdx;
} StdDevInfo_t;

inline uint32_t getColor(uint32_t it) {
   unsigned char R = (it>>2)&1 | (it>>4)&2 | (it>>6)&4 | (it>>8)&8 | (it>>10)&16;
   unsigned char G = (it>>1)&1 | (it>>3)&2 | (it>>5)&4 | (it>>7)&8 | (it>>9)&16;
   unsigned char B = it&1 | (it>>2)&2 | (it>>4)&4 | (it>>6)&8 | (it>>8)&16 | (it>>10)&32;
   return B<<2 | G<<11 | R<<19;
}

uint32_t numIter(double x, double y) {
   uint32_t it=0;	
   double x0 = x;
   double y0 = y;
   double xSqr, ySqr;
	
	while (((xSqr = x*x) + (ySqr = y*y) <= 4) && (it < MAX)) {
	   y = 2*x*y + y0;
		x = xSqr - ySqr + x0;
		it++;
	}
	
	return it;
}

void saveImage(char *name, uint32_t *iters) {
   char bfType1=0x42;
   char bfType2=0x4D;
   char bfSize[4]={0x00,0x00,0x00,0x00}; 
   char bfReserved1[2]={0x00,0x00};
   char bfReserved2[2]={0x00,0x00};
   char bfOffset[4]={0x36,0x00,0x00,0x00};
   char biSize[4]={0x28,0x00,0x00,0x00};
	
   char biWidth[4] = {char (WIDTH),
	                   char (WIDTH >> 8),
							 char (WIDTH >> 16),
							 char (WIDTH >> 24)};
   char biHeight[4] = {char (HEIGHT),
	                    char (HEIGHT >> 8),
							  char (HEIGHT >> 16),
							  char (HEIGHT >> 24)};
							 
   char biPlanes[2]={0x01,0x00};
   char biBitCount[2]={0x18,0x00};
   char biCompression[4]={0x00,0x00,0x00,0x00};
   char biSizeImage[4]={0x00,0x00,0x00,0x00};
   char biXPelsPerMeter[4]={0x00,0x00,0x00,0x00};
   char biYPelsPerMeter[4]={0x00,0x00,0x00,0x00};
   char biClrUsed[4]={0x00,0x00,0x00,0x00};
   char biClrImportant[4]={0x00,0x00,0x00,0x00};

   fstream image;
   image.open (name, fstream::binary | fstream::out);
	
   image.put(bfType1);
   image.put(bfType2);
   for(int i=0;i<4;i++){image.put(bfSize[i]);}
   for(int i=0;i<2;i++){image.put(bfReserved1[i]);}
   for(int i=0;i<2;i++){image.put(bfReserved2[i]);}
   for(int i=0;i<4;i++){image.put(bfOffset[i]);}
   for(int i=0;i<4;i++){image.put(biSize[i]);}
   for(int i=0;i<4;i++){image.put(biWidth[i]);}
   for(int i=0;i<4;i++){image.put(biHeight[i]);}
   for(int i=0;i<2;i++){image.put(biPlanes[i]);}
   for(int i=0;i<2;i++){image.put(biBitCount[i]);}
   for(int i=0;i<4;i++){image.put(biCompression[i]);}
   for(int i=0;i<4;i++){image.put(biSizeImage[i]);}
   for(int i=0;i<4;i++){image.put(biXPelsPerMeter[i]);}
   for(int i=0;i<4;i++){image.put(biYPelsPerMeter[i]);}
   for(int i=0;i<4;i++){image.put(biClrUsed[i]);}
   for(int i=0;i<4;i++){image.put(biClrImportant[i]);}

	int offset = 4 - ((WIDTH * 3) % 4);
	if (offset == 4)
	   offset = 0;
	
   for (int i = 0; i < HEIGHT; i++) {
      for (int j = 0; j < WIDTH; j++) {
		   uint32_t color = getColor(iters[i * WIDTH + j]);
         image.put(char(color >> 16));
         image.put(char(color >> 8));
         image.put(char(color));
      }
		
      for (int j = 0; j < offset; j++)
         image.put(0);
   }
   
	image.close();
}

inline bool BetterZoom(double oMean, double oVar, double nMean, double nVar) {
   return nVar > oVar;
}

double Variance(uint32_t iters[], double mean, uint32_t count) {
   if (count == 0)
      return 0.0;
	
   double sqrSum = 0.0;
   for (int i = 0; i < count; i++)
      sqrSum = pow(mean - (double)iters[i], 2);
	
   return (sqrSum/(count-1));
}

void insertSorted(StdDevInfo_t stdDevs[], int *varCount, uint32_t iters[], int count, int xNdx, int yNdx) {
   if (count == 0)
	   return;
	
	uint32_t sum = 0;
	int ndx = *varCount;
	double mean, variance;
	
	for (int i = 0; i < count; i++)
	   sum += iters[i];
	
	mean = (double) sum / (double) count;
	variance = Variance(iters, sum, count);
	
	while (ndx > 0 && BetterZoom(stdDevs[ndx - 1].mean, stdDevs[ndx - 1].variance, mean, variance)) {
	   if (ndx < RANDOM_POOL_SIZE)
	      stdDevs[ndx] = stdDevs[ndx - 1];
	   ndx--;
	}
	
	if (ndx < RANDOM_POOL_SIZE) {
	   if (*varCount < RANDOM_POOL_SIZE)
		   ++*varCount;
		
		stdDevs[ndx].variance = variance;
		stdDevs[ndx].mean = mean;
		stdDevs[ndx].xNdx = xNdx;
		stdDevs[ndx].yNdx = yNdx;
	}
}

void Mandelbrot(double x, double y, double resolution, uint32_t *iters) {
   // inline numIters and remove the loops for CUDA implementation
	for (int i = 0; i < HEIGHT; i++)
	   for (int j = 0; j < WIDTH; j++)
		   iters[i * WIDTH + j] = numIter(x + (j * resolution / WIDTH), y + (i * resolution / HEIGHT));
}

void findPath(uint32_t *iters, double *startX, double *startY, double *resolution) {
   int count;
	uint32_t subIter[(2*STD_DEV_RADIUS+1)*(2*STD_DEV_RADIUS+1)];
	
	StdDevInfo_t stdDevs[RANDOM_POOL_SIZE];
	int varCount = 0;
	
	for (int i = 0; i < HEIGHT; i++) {
	   for (int j = 0; j < WIDTH; j++) {
		   count = 0;
			
			for (int k = max(0, i - STD_DEV_RADIUS); k <= min(HEIGHT - 1, i + STD_DEV_RADIUS); k++)
			   for (int l = max(0, j - STD_DEV_RADIUS); l <= min(WIDTH - 1, j + STD_DEV_RADIUS); l++)
				   subIter[count++] = getColor(iters[k * WIDTH + l]);
			
			insertSorted(stdDevs, &varCount, subIter, count, j, i);
		}
	}
	
	int path = clock() & (RANDOM_POOL_SIZE - 1);
	
	*startX += stdDevs[path].xNdx * *resolution / WIDTH;
	*startY += stdDevs[path].yNdx * *resolution / HEIGHT;
	
	(*resolution) /= 2.0;
}

int main(int argc, char* argv[]) {
   double startX = -1.50;
	double startY = -1.00;
	double resolution = INITIAL_RESOLUTION;
	
	uint32_t iters[WIDTH * HEIGHT];

	char file[] = "a.bmp";
   for (int i = 0; i < DEPTH; i++, file[0]++) {
	   Mandelbrot(startX, startY, resolution, iters);
		saveImage(file, iters);
		findPath(iters, &startX, &startY, &resolution);
	}
	
   return 0;
}
