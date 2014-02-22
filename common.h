#ifndef COMMON_H
#define COMMON_H

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

uint32_t getColor(uint32_t it);

void saveImage(char *name, uint32_t *iters);

bool BetterZoom(double oMean, double oVar, double nMean, double nVar);

double Variance(uint32_t iters[], double mean, uint32_t count);

void insertSorted(StdDevInfo_t stdDevs[], int *varCount, uint32_t iters[], int count, int xNdx, int yNdx);

void findPath(uint32_t *iters, double *startX, double *startY, double *resolution);

#endif