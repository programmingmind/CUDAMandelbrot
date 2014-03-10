#ifndef DATATYPES_H
#define DATATYPES_H

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <utility>
#include <vector>

#define MIN_BYTES 4

#define HIGH32 0x80000000
#define HIGH8 0x80
#define LOWBIT 0x01

#define BASE 65536
#define BASE_SQR 4294967296ULL

typedef struct {
   uint32_t *data;
   int len;
   int extra;
} splitInfo_t;

class Number {
private:
   void *data;
   int numBytes;

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
   static Number comb(int64_t *l, int len);

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
   Number operator>>(const int a);

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
   int getSize();

   HOST
   friend std::ostream& operator<<(std::ostream& os, const Number& n);
};

class Decimal {
private:
   bool negative;
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
   bool operator>=(const uint32_t a);

   HOST DEVICE
   bool operator<=(const uint32_t a);

   HOST DEVICE
   Decimal& operator+=(const Decimal& d);

   HOST DEVICE
   Decimal& operator/=(const Decimal& d);

   HOST
   friend std::ostream& operator<<(std::ostream& os, const Decimal& d);
};

#endif
