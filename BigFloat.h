#ifndef BIGFLOAT_H
#define BIGFLOAT_H

#ifdef __CUDACC__
#ifdef _WIN32
#define HOST
#define DEVICE
#else
#define HOST __host__
#define DEVICE __device__
#endif
#else
#define HOST
#define DEVICE
#endif

#include <inttypes.h>

#include <iostream>
#include <iomanip>

#define BF_SIZE 16
//#define BF_SIZE 32

#define EQ 0
#define GT 1
#define LT -1

typedef struct BigFloat {
   uint64_t data[BF_SIZE]; // only use 32 bits, but keep as 64 bit so no overflow
   int64_t exponent;
   uint8_t negative;

   HOST
   friend std::ostream& operator<<(std::ostream& os, const struct BigFloat& bf);

   HOST
   BigFloat& operator=(const double d);

   HOST
   BigFloat& operator+=(BigFloat bf);

   HOST
   BigFloat operator*(const unsigned int i);

   HOST
   BigFloat& operator>>=(const int i);

   HOST
   BigFloat operator>>(const int i);
} BigFloat;

HOST DEVICE
BigFloat* init(BigFloat *val, uint32_t number);

HOST
BigFloat* initDouble(BigFloat *val, double number);

HOST DEVICE
void assign(BigFloat *dest, BigFloat *src);

HOST DEVICE
int isZero(BigFloat *val);

HOST DEVICE
int cmp(BigFloat *one, BigFloat *two);

HOST DEVICE
int base2Cmp(BigFloat *val, int32_t power);

HOST DEVICE
BigFloat* shiftL(BigFloat *val, uint64_t amount);

HOST DEVICE
BigFloat* shiftR(BigFloat *val, uint64_t amount);

HOST DEVICE
BigFloat* add(BigFloat *one, BigFloat *two, BigFloat *result);

HOST DEVICE
BigFloat* sub(BigFloat *one, BigFloat *two, BigFloat *result);

HOST DEVICE
BigFloat* mult(BigFloat *one, BigFloat *two, BigFloat *result, BigFloat *tmp);

#endif
