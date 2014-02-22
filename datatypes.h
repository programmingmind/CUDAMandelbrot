#ifndef DATATYPES_H
#define DATATYPES_H

#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <iostream>

#define MIN_EXP 2

uint32_t nextBase2(uint32_t n) {
   uint32_t num;
   for (int i = 0; i < 32; i++)
      if ((num = (1 << i)) > n)
         return num;
   return num;
}

class Number {
private:
   void *data;
   int numBytes;

public:
   Number(int exp) {
      numBytes = 1 << std::max(exp, MIN_EXP);
      data = malloc(numBytes);
   }

   Number(const Number& num) {
      numBytes = num.numBytes;
      data = malloc(numBytes);

      memset(data, 0, numBytes);
      memcpy(data, num.data, numBytes);
   }

   ~Number() {
      free(data);
   }

   Number& operator=(const Number& a) {
      if (this == &a)
         return *this;

      memset(data, 0, numBytes);
      memcpy(data, a.data, std::min(numBytes, a.numBytes));
      return *this;
   }

   Number& operator=(unsigned int a) {
      memset(data, 0, numBytes);
      ((unsigned int *)data)[0] = a;
      return *this;
   }

   Number operator+(const Number& a) {
      int resultBytes = std::max(numBytes, a.numBytes);
      Number n(log2(resultBytes));

      int lSize = numBytes >> 2, rSize = a.numBytes >> 2, len = resultBytes >> 2;
      uint32_t *num1 = ((uint32_t *)data);
      uint32_t *num2 = ((uint32_t *)a.data);

      uint32_t l, r, s;
      char carry = 0;
      for (int i = 0; i*4 < len; i++) {
         l = i < lSize ? num1[i] : 0;
         r = i < rSize ? num2[i] : 0;

         s = l + r + carry;
         carry = ((s < l || s < r) || (carry > 0 && (s == l || s == r))) ? 1 : 0;

         ((uint32_t *)n.data)[i] = s;
      }

      if (carry > 0) {
         Number t(log2(resultBytes) + 1);
         t = n;
         ((uint32_t *)t.data)[resultBytes/4] = carry;

         return t;
      }

      return n;
   }

   // I know there is a bug here with the subtraction carry, need a way to propagate it furthur without modifying this.data
   Number operator-(const Number& a) {
      int resultBytes = std::max(numBytes, a.numBytes);
      Number n(log2(resultBytes));

      uint32_t *num1 = ((uint32_t *)data);
      uint32_t *num2 = ((uint32_t *)a.data);

      uint32_t l, r, s;
      char carry = 0;

      for (int i = 0; i*4 < resultBytes; i++) {
         l = num1[i];
         r = num2[i];

         if (r + carry <= l) {
            // normal subtraction
            s = l -r;
            carry = 0;
         } else {
            // l - r == -1 * (r - l)
            s = 1 + ~(r + carry - l);
            carry = 1;
         }

         ((uint32_t *)n.data)[i] = s;
      }

      return n;
   }

   Number& operator*(const Number& a) {
      return *this;
   }

   Number& operator/(const Number& a) {
      return *this;
   }

   void trim() {
      char* ptr = (char *) data;
      int used;

      for (int i = numBytes - 1; i >= 0; i--) {
         if (ptr[i]) {
            used = i + 1;
            break;
         }
      }

      int newBytes = std::max(nextBase2(used), (uint32_t) MIN_EXP);
      if (newBytes < numBytes) {
         void *smaller = malloc(newBytes);
         memcpy(smaller, data, newBytes);
         free(data);
         data = smaller;
         numBytes = newBytes;
      }
   }

   void* getData() {
      void *ptr = malloc(numBytes);
      memcpy(ptr, data, numBytes);
      return ptr;
   }
};

class Decimal {
private:
   void *data;
   int numBytes;

public:
   Decimal(int exp) {
      numBytes = 1 << std::max(exp, MIN_EXP);
      data = malloc(numBytes);
   }

   ~Decimal() {
      free(data);
   }
};

#endif