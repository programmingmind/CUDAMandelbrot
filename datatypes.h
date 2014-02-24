#ifndef DATATYPES_H
#define DATATYPES_H

#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <iostream>

#define MIN_BYTES 4

#define HIGH32 0x80000000
#define HIGH8 0x80
#define LOWBIT 0x01

inline bool numBase2(uint32_t n) {
   return n == 0 || ((n & (n - 1)) == 0);
}

inline uint32_t nextBase2(uint32_t n) {
   if (numBase2(n))
      return n;

   uint32_t num;
   for (int i = 0; i < 32; i++)
      if ((num = (1 << i)) >= n)
         return num;
   return num;
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

   inline Number& copyIn(Number a) {
      // the deallocater frees the pointer so just swap the pointers to handle the memory properly
      void *tmp = data;
      data = a.data;
      a.data = tmp;
      numBytes = a.numBytes;
      return *this;
   }

   bool isBase2() const {
      bool base2Seen = false;

      uint32_t *ptr = (uint32_t *) data;
      int len = numBytes >> 2;
      for (int i = 0; i < len; i++) {
         if (ptr[i] != 0) {
            if (numBase2(ptr[i])) {
               if (base2Seen)
                  return false;
               base2Seen = true;
            }
            else
               return false;
         }
      }

      return true;
   }

   // returns exponent of first high bit
   int binlog() const {
      uint32_t *ptr = (uint32_t *) data;
      int len = numBytes >> 2;

      for (int i = 0; i < len; i++)
         if (ptr[i] != 0)
            return i*32 + log2(ptr[i]);

      return 0;
   }

public:
   Number(int bytes) {
      numBytes = nextBase2(std::max(bytes, MIN_BYTES));
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
      Number n(std::max(numBytes, a.numBytes));

      int lSize = numBytes >> 2, rSize = a.numBytes >> 2, len = n.numBytes >> 2;
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
         Number t(n.numBytes + 1);
         memcpy(t.data, n.data, n.numBytes);
         ((uint32_t *)t.data)[len] = carry;

         return t;
      }

      return n;
   }

   // I know there is a bug here with the subtraction carry, need a way to propagate it furthur without modifying this.data
   Number operator-(const Number& a) {
      Number n(std::max(numBytes, a.numBytes));

      uint32_t *num1 = ((uint32_t *)data);
      uint32_t *num2 = ((uint32_t *)a.data);

      int lSize = numBytes >> 2, rSize = a.numBytes >> 2, len = n.numBytes >> 2;

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

   Number operator*(const Number& a) {
      Number p(numBytes + a.numBytes);
      memset(p.data, 0, p.numBytes);

      uint32_t *num1 = ((uint32_t *)data);
      uint32_t *num2 = ((uint32_t *)a.data);

      int lSize = numBytes >> 2, rSize = a.numBytes >> 2;
      uint64_t prod;
      for (int i = 0; i < lSize; i++) {
         for (int j = 0; j < rSize; j++) {
            prod = ((uint64_t) num1[i]) * ((uint64_t) num2[j]);
            Number t(&prod, 8);
            p += (t << ((i + j) * 32));
         }
      }

      p.trim();
      return p;
   }

   Number operator/(const Number& a) {
      // http://en.wikipedia.org/wiki/Fourier_division
      // b_i = (r_i-1,c_i+1 - 
      //        sum j=2 -> i of
      //           b_i-j+1 * a_j
      //       ) / (a_1)
      // where b = c/a and a,b,c are 1-indexed from MSByte

      if (a.isBase2())
         return operator>>(a.binlog());

      return Number(a);

      // this needs to be fixed
      // uint32_t *aPtr = (uint32_t *) a.data;
      // uint32_t *cPtr = (uint32_t *) data;
      // int aNdx = a.numBytes;

      // while (aPtr[--aNdx] == 0) ;

      // uint64_t b;
      // uint32_t *r = (uint32_t *) malloc(a.numBytes);

      // uint64_t t = (((uint64_t) cPtr[0]) << 32) | cPtr[1];
      // b = t / aPtr[0];
      // r[0] = t % aPtr[0];

      // Number n(&b, 8);
      // for (int i = 1; i <= aNdx; i++) {

      // }

      // free(r);
      // n.trim();
      // return n;
   }

   Number operator<<(const int a) {
      int bytes = a / 8;
      int bits = a % 8;

      int overflow = 0;
      if (topBitsSet(data, numBytes, a))
         overflow = bytes + (bits > 0 ? 1 : 0);

      Number t(numBytes + overflow);
      memset(t.data, 0, t.numBytes);

      char *ptr = (char *) t.data;
      memcpy(ptr + bytes, data, numBytes);

      unsigned char mask = (~((1 << (8 - bits)) - 1));
      unsigned char over = 0, tmp;
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

      Number t(numBytes - bytes);
      memset(t.data, 0, t.numBytes);

      unsigned char *ptr = (unsigned char *) t.data;
      memcpy(ptr, ((char *) data) + bytes, numBytes - bytes);

      unsigned char mask = (1 << bits) - 1;
      unsigned char under = 0, tmp;
      for (int i = t.numBytes - 1; i >= 0; i--) {
         tmp = ptr[i] & mask;
         ptr[i] >>= bits;
         ptr[i] |= under << (8 - bits);
         under = tmp;
      }

      return t;
   }

   Number operator&(const Number& a) {
      Number n(std::min(numBytes, a.numBytes));

      memset(n.data, 0, n.numBytes);

      uint32_t *l = (uint32_t *) data, *r = (uint32_t *) a.data, *v = (uint32_t *) n.data;

      for (int i = 0; i*4 < n.numBytes; i++)
         v[i] = l[i] & r[i];

      return n;
   }

   Number operator|(const Number& a) {
      Number n(std::max(numBytes, a.numBytes));

      int lSize = numBytes >> 2, rSize = a.numBytes >> 2, len = n.numBytes >> 2;
      uint32_t *l = (uint32_t *) data, *r = (uint32_t *) a.data, *v = (uint32_t *) n.data;

      for (int i = 0; i < len; i++)
         v[i] = (i < lSize ? l[i] : 0) | (i < rSize ? r[i] : 0);

      return n;
   }

   Number operator^(const Number& a) {
      Number n(std::max(numBytes, a.numBytes));

      int lSize = numBytes >> 2, rSize = a.numBytes >> 2, len = n.numBytes >> 2;
      uint32_t *l = (uint32_t *) data, *r = (uint32_t *) a.data, *v = (uint32_t *) n.data;

      for (int i = 0; i < len; i++)
         v[i] = (i < lSize ? l[i] : 0) ^ (i < rSize ? r[i] : 0);

      return n;
   }

   Number& operator+=(const Number& a) {
      return copyIn(operator+(a));
   }

   Number& operator-=(const Number& a) {
      return copyIn(operator-(a));
   }

   Number& operator*=(const Number& a) {
      return copyIn(operator*(a));
   }

   Number& operator/=(const Number& a) {
      return copyIn(operator/(a));
   }

   Number& operator<<=(const int a) {
      return copyIn(operator<<(a));
   }

   Number& operator>>=(const int a) {
      return copyIn(operator>>(a));
   }

   Number& operator&=(const Number& a) {
      return copyIn(operator&(a));
   }

   Number& operator|=(const Number& a) {
      return copyIn(operator|(a));
   }

   Number& operator^=(const Number& a) {
      return copyIn(operator^(a));
   }

   Number& operator&=(const uint32_t a) {
      return copyIn(operator&(a));
   }

   Number& operator|=(const uint32_t a) {
      return copyIn(operator|(a));
   }

   Number& operator^=(const uint32_t a) {
      return copyIn(operator^(a));
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
   Number operator+(const uint32_t a) {
      uint32_t v = a;
      Number t(&v, 4);
      const Number& r = t;
      return operator+(r);
   }

   Number operator-(const uint32_t a) {
      uint32_t v = a;
      Number t(&v, 4);
      const Number& r = t;
      return operator-(r);
   }

   Number operator*(const uint32_t a) {
      uint32_t v = a;
      Number t(&v, 4);
      const Number& r = t;
      return operator*(r);
   }

   Number operator/(const uint32_t a) {
      uint32_t v = a;
      Number t(&v, 4);
      const Number& r = t;
      return operator/(r);
   }

   Number operator+=(const uint32_t a) {
      uint32_t v = a;
      Number t(&v, 4);
      const Number& r = t;
      return operator+=(r);
   }

   Number operator-=(const uint32_t a) {
      uint32_t v = a;
      Number t(&v, 4);
      const Number& r = t;
      return operator-=(r);
   }

   Number operator*=(const uint32_t a) {
      uint32_t v = a;
      Number t(&v, 4);
      const Number& r = t;
      return operator*=(r);
   }

   Number operator/=(const uint32_t a) {
      uint32_t v = a;
      Number t(&v, 4);
      const Number& r = t;
      return operator/=(r);
   }

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

      int newBytes = std::max(nextBase2(used), (uint32_t) MIN_BYTES);
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
      numBytes = std::max(1 << exp, MIN_BYTES);
      data = malloc(numBytes);
   }

   ~Decimal() {
      free(data);
   }
};

#endif