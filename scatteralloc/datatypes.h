#ifndef DATATYPES_H
#define DATATYPES_H

#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <utility>
#include <vector>

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#define SCATTERALLOC_OVERWRITE_MALLOC 1
#define SCATTERALLOC_HEAPARGS 4096, 8, 16, 2, true, false
#else
#define HOST
#define DEVICE
#endif

#define MIN_BYTES 4

#define HIGH32 0x80000000
#define HIGH8 0x80
#define LOWBIT 0x01

#define BASE 65536
#define BASE_SQR 4294967296ULL

void initializeHeap();

typedef struct {
   uint32_t *data;
   int len;
   int extra;
} splitInfo_t;

class Number {
private:
   void *data;
   int numBytes;
   bool onDevice;

   HOST DEVICE
   bool compare(const Number& a, bool lt);

   inline HOST DEVICE
   Number& copyIn(Number a);

   HOST DEVICE
   int topBytesEmpty() const;

   HOST DEVICE
   bool nonZero() const;

   HOST DEVICE
   splitInfo_t split() const;

   HOST DEVICE
   uint32_t getLSU32() const;

   HOST DEVICE
   uint32_t getLSU16() const;

public:
   HOST DEVICE
   Number();

   HOST DEVICE
   Number(int bytes);

   HOST DEVICE
   Number(const void *bytes, int len);

   HOST DEVICE
   Number(const Number& num);

   HOST DEVICE
   ~Number();

   HOST DEVICE
   void resize(int bytes);

   // returns exponent of first high bit
   HOST DEVICE
   int binlog() const;

   HOST DEVICE
   bool isBase2() const;

   HOST DEVICE
   Number& operator=(const Number& a);

   HOST DEVICE
   Number& operator=(unsigned int a);

   HOST DEVICE
   Number& operator=(uint64_t a);

   HOST DEVICE
   Number operator+(const Number& a);

   HOST DEVICE
   Number operator-(const Number& a);

   HOST DEVICE
   Number operator*(const Number& a);

   HOST DEVICE
   Number operator/(const Number& aN);

   HOST DEVICE
   Number operator<<(const int a) const;

   HOST DEVICE
   Number operator>>(const int a) const;

   HOST DEVICE
   Number operator&(const Number& a);

   HOST DEVICE
   Number operator|(const Number& a);

   HOST DEVICE
   Number operator^(const Number& a);

   HOST DEVICE
   Number& operator+=(const Number& a);

   HOST DEVICE
   Number& operator-=(const Number& a);

   HOST DEVICE
   Number& operator*=(const Number& a);

   HOST DEVICE
   Number& operator/=(const Number& a);

   HOST DEVICE
   Number& operator<<=(const int a);

   HOST DEVICE
   Number& operator>>=(const int a);

   HOST DEVICE
   Number& operator&=(const Number& a);

   HOST DEVICE
   Number& operator|=(const Number& a);

   HOST DEVICE
   Number& operator^=(const Number& a);

   HOST DEVICE
   Number& operator&=(const uint32_t a);

   HOST DEVICE
   Number& operator|=(const uint32_t a);

   HOST DEVICE
   Number& operator^=(const uint32_t a);

   HOST DEVICE
   bool operator==(const Number& a);

   HOST DEVICE
   bool operator!=(const Number& a);

   HOST DEVICE
   bool operator>(const Number& a);

   HOST DEVICE
   bool operator<(const Number& a);

   HOST DEVICE
   bool operator>=(const Number& a);

   HOST DEVICE
   bool operator<=(const Number& a);

   HOST DEVICE
   Number operator%(const uint32_t a);

   HOST DEVICE
   Number operator+(const uint32_t a);

   HOST DEVICE
   Number operator-(const uint32_t a);

   HOST DEVICE
   Number operator*(const uint32_t a);

   HOST DEVICE
   Number operator*(const uint64_t a);

   HOST DEVICE
   Number operator/(const uint32_t a);

   HOST DEVICE
   Number operator+=(const uint32_t a);

   HOST DEVICE
   Number operator-=(const uint32_t a);

   HOST DEVICE
   Number operator*=(const uint32_t a);

   HOST DEVICE
   Number operator/=(const uint32_t a);

   HOST DEVICE
   Number operator&(const uint32_t a);

   HOST DEVICE
   Number operator|(const uint32_t a);

   HOST DEVICE
   Number operator^(const uint32_t a);

   HOST DEVICE
   bool operator==(const uint32_t a);

   HOST DEVICE
   bool operator!=(const uint32_t a);

   HOST DEVICE
   bool operator>(const uint32_t a);

   HOST DEVICE
   bool operator<(const uint32_t a);

   HOST DEVICE
   bool operator>=(const uint32_t a);

   HOST DEVICE
   bool operator<=(const uint32_t a);

   HOST DEVICE
   void trim();

   HOST DEVICE
   void* getData();

   HOST DEVICE
   int getSize() const;

   HOST
   friend std::ostream& operator<<(std::ostream& os, const Number& n);

   HOST DEVICE
   Number absVal();

   HOST DEVICE
   Number toDevice() const;

   HOST
   void deviceFree();

   HOST DEVICE
   bool isDevice() const;

   DEVICE
   void clearDevice();
};

class Decimal {
private:
   bool negative, onDevice;
   int32_t exponent;
   Number mantissa;

   HOST DEVICE
   bool compare(const Decimal& a, bool lt);

   inline HOST DEVICE
   Decimal& copyIn(Decimal d);

public:
   HOST DEVICE
   Decimal(unsigned int i);

   HOST DEVICE
   Decimal(float f);

   HOST DEVICE
   Decimal(double d);

   HOST DEVICE
   Decimal(Number &n);

   HOST DEVICE
   Decimal(const Decimal& d);

   HOST DEVICE
   Number getMantissa();

   HOST DEVICE
   Decimal& operator=(const Decimal& a);

   HOST DEVICE
   Decimal operator+(const Decimal& a);

   HOST DEVICE
   Decimal operator-(const Decimal& a);

   HOST DEVICE
   Decimal operator*(const Decimal& a);

   HOST DEVICE
   Decimal operator/(const Decimal& a);

   HOST DEVICE
   bool operator>(const Decimal& a);

   HOST DEVICE
   bool operator<(const Decimal& a);

   HOST DEVICE
   bool operator>=(const Decimal& a);

   HOST DEVICE
   bool operator<=(const Decimal& a);

   HOST DEVICE
   bool operator==(const Decimal& a);

   HOST DEVICE
   bool operator>(const uint32_t a);

   HOST DEVICE
   bool operator<(const uint32_t a);

   HOST DEVICE
   bool operator<(const double a);

   HOST DEVICE
   bool operator>=(const uint32_t a);

   HOST DEVICE
   bool operator<=(const uint32_t a);

   HOST DEVICE
   Decimal& operator+=(const Decimal& d);

   HOST DEVICE
   Decimal& operator/=(const Decimal& d);

   HOST
   friend std::ostream& operator<<(std::ostream& os, const Decimal& d);

   HOST DEVICE
   Decimal absVal();

   HOST
   Decimal toDevice() const;

   HOST
   void deviceFree();

   DEVICE
   void clearDevice();
};

#endif
