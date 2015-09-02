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
#include <dirent.h>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>

using namespace std;

#define DIM_POWER 9
#define WIDTH (1 << DIM_POWER)
#define HEIGHT WIDTH
#define INITIAL_RESOLUTION 2.0

#define MAX 65536

#define STD_DEV_RADIUS 5
#define RANDOM_POOL_SIZE (1 << 2)

typedef struct {
   double variance;
   double mean;
   unsigned int xNdx;
   unsigned int yNdx;
} StdDevInfo_t;

int getNumThreads();

void setNumThreads(int t);

uint32_t getColor(uint32_t it);

int findCurrentRun();

void saveImage(int run, int len, int num, uint32_t *iters);

bool BetterZoom(double oMean, double oVar, double nMean, double nVar);

double Variance(uint32_t iters[], double mean, uint32_t count);

void insertSorted(StdDevInfo_t stdDevs[], int *varCount, uint32_t iters[], int count, int xNdx, int yNdx);

void findPath(uint32_t *iters, data_t *startX, data_t *startY, data_t *resolution, int *xNdx, int *yNdx);

#endif
