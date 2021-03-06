#ifndef COMMON_H
#define COMMON_H

#ifdef CUDA
#include "cuda.h"
#else
#include "cpu.h"
#endif

#include <inttypes.h>
#include <string.h>

#include <sys/stat.h>
#include <sys/types.h>

#if defined _MSC_VER
#include <direct.h>
#include "dirent.h"
#elif defined __GNUC__
#include <dirent.h>
#endif

#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>

using namespace std;

#define DIM_POWER 9
#define WIDTH (1 << DIM_POWER)
#define HEIGHT WIDTH

#define INITIAL_RESOLUTION 2.0
#define INITIAL_X -1.50
#define INITIAL_Y -1.00

#define ITER_STEP 256
#define MAX (ITER_STEP*4)
//#define MAX 65536

#define STD_DEV_RADIUS 3
#define RANDOM_POOL_SIZE (1 << 2)

typedef struct {
   double variance;
   double mean;
   unsigned int xNdx;
   unsigned int yNdx;
   unsigned int numNonMax;
} StdDevInfo_t;

int getNumThreads();

void setNumThreads(int t);

uint32_t getColor(uint32_t it);

int findCurrentRun();

void saveImage(int run, int len, int num, uint32_t *iters);

bool BetterZoom(double oMean, double oVar, double nMean, double nVar);

double Variance(uint32_t iters[], double mean, uint32_t count);

void insertSorted(StdDevInfo_t stdDevs[], int *varCount, uint32_t iters[], int count, int xNdx, int yNdx);

void findPath(uint32_t *iters, int *xNdx, int *yNdx);

#endif
