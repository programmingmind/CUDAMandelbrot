#ifndef DATATYPES_H
#define DATATYPES_H

#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <iostream>

#define MIN_EXP 2

#define HIGH32 0x80000000
#define HIGH8 0x80
#define LOWBIT 0x01

inline uint32_t nextLog2(uint32_t n) {
   for (int i = 0; i < 32; i++)
      if ((1 << i) >= n)
         return i;
   return 31;
}

inline uint32_t nextBase2(uint32_t n) {
   return 1 << nextLog2(n);
}

bool topBitsSet(void *data, int len, int numBits) {
   char *d = (char *) data;
   int bytes = numBits / 8, bits = numBits % 8;

   if (numBits > len * 8)
      return true;

   for (int i = 1; i <= bytes; i++)
      if (d[len - i] != 0)
         return true;

   return numBits > 0 && (((~((1 << (8 - numBits)) - 1)) & d[len - bytes - 1]) != 0);
}

class Number {
private:
   void *data;
   int numBytes;

   bool compare(const Number& a, bool lt) {
      int lSize = numBytes >> 2, rSize = a.numBytes >> 2, len = std::max(lSize, rSize);
      uint32_t *l = (uint32_t *) data, *r = (uint32_t *) a.data;

      for (int i = len - 1; i >= 0; i--) {
         if ((i < lSize ? l[i] : 0) < (i < rSize ? r[i] : 0))
            return lt;
         if ((i < lSize ? l[i] : 0) > (i < rSize ? r[i] : 0))
            return ! lt;
      }
      return false;
   }

public:
   Number(int exp) {
      numBytes = 1 << std::max(exp, MIN_EXP);
      data = malloc(numBytes);
   }

   Number(void *bytes, int len) {
      numBytes = nextBase2(len);
      data = calloc(len, 1);
      memcpy(data, bytes, len);
   }

   Number(const Number& num) {
      numBytes = num.numBytes;
      data = malloc(numBytes);

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
      for (int i = 0; i < len; i++) {
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

      int lSize = numBytes >> 2, rSize = a.numBytes >> 2, len = resultBytes >> 2;

      uint32_t l, r, s;
      char carry = 0;

      for (int i = 0; i < len; i++) {
         l = i < lSize ? num1[i] : 0;
         r = i < rSize ? num2[i] : 0;;

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

   Number operator<<(const int a) {
      int bytes = a / 8;
      int bits = a % 8;

      int overflow = 0;
      if (topBitsSet(data, numBytes, a))
         overflow = bytes + (bits > 0 ? 1 : 0);

      Number t(nextLog2(numBytes + overflow));
      memset(t.data, 0, t.numBytes);

      char *ptr = (char *) t.data;
      memcpy(ptr + bytes, data, numBytes);

      char mask = (~((1 << (8 - bits)) - 1));
      char over = 0, tmp;
      for (int i = 0; i < t.numBytes; i++) {
         tmp = ptr[i] & mask;
         ptr[i] <<= bits;
         ptr[i] |= over >> (8 - bits);
         over = tmp;
      }

      return t;
   }

   Number operator>>(const int a) {
      int bytes = a / 8;
      int bits = a % 8;

      Number t(nextLog2(numBytes - bytes));
      memset(t.data, 0, t.numBytes);

      char *ptr = (char *) t.data;
      memcpy(ptr, ((char *) data) + bytes, numBytes - bytes);

      char mask = (1 << bits) - 1;
      char under = 0, tmp;
      for (int i = t.numBytes - 1; i >= 0; i--) {
         tmp = ptr[i] & mask;
         ptr[i] >>= bits;
         ptr[i] |= under << (8 - bits);
         under = tmp;
      }

      return t;
   }

   Number operator&(const Number& a) {
      Number n(log2(std::min(numBytes, a.numBytes)));

      memset(n.data, 0, n.numBytes);

      uint32_t *l = (uint32_t *) data, *r = (uint32_t *) a.data, *v = (uint32_t *) n.data;

      for (int i = 0; i*4 < n.numBytes; i++)
         v[i] = l[i] & r[i];

      return n;
   }

   Number operator|(const Number& a) {
      Number n(log2(std::max(numBytes, a.numBytes)));

      int lSize = numBytes >> 2, rSize = a.numBytes >> 2, len = n.numBytes >> 2;
      uint32_t *l = (uint32_t *) data, *r = (uint32_t *) a.data, *v = (uint32_t *) n.data;

      for (int i = 0; i < len; i++)
         v[i] = (i < lSize ? l[i] : 0) | (i < rSize ? r[i] : 0);

      return n;
   }

   Number operator^(const Number& a) {
      Number n(log2(std::max(numBytes, a.numBytes)));

      int lSize = numBytes >> 2, rSize = a.numBytes >> 2, len = n.numBytes >> 2;
      uint32_t *l = (uint32_t *) data, *r = (uint32_t *) a.data, *v = (uint32_t *) n.data;

      for (int i = 0; i < len; i++)
         v[i] = (i < lSize ? l[i] : 0) ^ (i < rSize ? r[i] : 0);

      return n;
   }

   bool operator==(const Number& a) {
      int lSize = numBytes >> 2, rSize = a.numBytes >> 2, len = std::max(lSize, rSize);
      uint32_t *l = (uint32_t *) data, *r = (uint32_t *) a.data;

      for (int i = 0; i < len; i++)
         if ((i < lSize ? l[i] : 0) != (i < rSize ? r[i] : 0))
            return false;
      return true;
   }

   bool operator!=(const Number& a) {
      return ! operator==(a);
   }

   bool operator>(const Number& a) {
      return compare(a, false);
   }

   bool operator<(const Number& a) {
      return compare(a, true);
   }

   bool operator>=(const Number& a) {
      return ! operator<(a);
   }

   bool operator<=(const Number& a) {
      return ! operator>(a);
   }

   // there must be a better way to do these functions...
   Number operator&(const uint32_t a) {
      uint32_t v = a;
      Number t(&v, 4);
      const Number& r = t;
      return operator&(r);
   }

   Number operator|(const uint32_t a) {
      uint32_t v = a;
      Number t(&v, 4);
      const Number& r = t;
      return operator^(r);
   }

   Number operator^(const uint32_t a) {
      uint32_t v = a;
      Number t(&v, 4);
      const Number& r = t;
      return operator^(r);
   }

   bool operator==(const uint32_t a) {
      uint32_t v = a;
      Number t(&v, 4);
      const Number& r = t;
      return operator==(r);
   }

   bool operator!=(const uint32_t a) {
      uint32_t v = a;
      Number t(&v, 4);
      const Number& r = t;
      return operator!=(r);
   }

   bool operator>(const uint32_t a) {
      uint32_t v = a;
      Number t(&v, 4);
      const Number& r = t;
      return operator>(r);
   }

   bool operator<(const uint32_t a) {
      uint32_t v = a;
      Number t(&v, 4);
      const Number& r = t;
      return operator<(r);
   }

   bool operator>=(const uint32_t a) {
      uint32_t v = a;
      Number t(&v, 4);
      const Number& r = t;
      return operator>=(r);
   }

   bool operator<=(const uint32_t a) {
      uint32_t v = a;
      Number t(&v, 4);
      const Number& r = t;
      return operator<=(r);
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

   int getSize() {
      return numBytes;
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