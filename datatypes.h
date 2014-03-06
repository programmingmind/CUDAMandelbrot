#ifndef DATATYPES_H
#define DATATYPES_H

#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#define MIN_BYTES 4

#define HIGH32 0x80000000
#define HIGH8 0x80
#define LOWBIT 0x01

#define BASE 65536
#define BASE_SQR 4294967296ULL

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

uint32_t unsignedAbs(int64_t i) {
   if (i < 0)
      i *= -1;

   return *((uint32_t *) ((void *) &i));
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

   bool nonZero() const {
      uint32_t *ptr = (uint32_t *) data;
      int len = numBytes >> 2;

      while (len--)
         if (ptr[len])
            return true;

      return false;
   }

   std::pair<std::vector<uint32_t>, int> split() const {
      std::vector<uint32_t> t;
      int extra = 0;

      Number tmp(*this);

      while (tmp.nonZero()) { // since BASE is 2^n we can optimize this into bit ops
         t.insert(t.begin(), tmp.getLSU32());
         tmp /= BASE;
      }

      if (t.size() % 2 == 1) {
         t.push_back(0);
         extra++;
      }

      std::vector<uint32_t> a;
      for (int i = 0; i < t.size(); i += 2)
         a.push_back(t[i] * BASE + t[i + 1]);

      if (a.size() == 1) {
         a.push_back(0);
         extra += 2;
      }

      return std::make_pair(a, extra);
   }

   static Number comb(std::vector<int64_t> l) {
      Number n(4 * (int)(l.end() - l.begin()));
      memset(n.data, 0, n.numBytes);

      for (std::vector<int64_t>::iterator it = l.begin(); it != l.end(); it++) {
         n <<= 32;
         bool sub = *it < 0;
         uint32_t val = unsignedAbs(*it);

         if (sub)
            n -= val;
         else
            n += val;
      }

      return n;
   }

   uint32_t getLSU32() const {
      return *((uint32_t *) data);
   }

public:
   Number() {
      numBytes = MIN_BYTES;
      data = malloc(numBytes);
   }

   Number(int bytes) {
      numBytes = nextBase2(std::max(bytes, MIN_BYTES));
      data = malloc(numBytes);
   }

   Number(const void *bytes, int len) {
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

   Number operator-(const Number& a) {
      Number n(std::max(numBytes, a.numBytes));

      uint32_t *num1 = ((uint32_t *)data);
      uint32_t *num2 = ((uint32_t *)a.data);

      int lSize = numBytes >> 2, rSize = a.numBytes >> 2, len = n.numBytes >> 2;

      uint32_t l, r, s;
      char carry = 0;

      for (int i = 0; i < len; i++) {
         l = i < lSize ? num1[i] : 0;
         r = i < rSize ? num2[i] : 0;

         if (r + carry <= l) {
            // normal subtraction
            s = l - r - carry;
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

   Number operator/(const Number& aN) {
      if (! (nonZero() && aN.nonZero())) {
         return *this;
      }

      if (aN.isBase2())
         return operator>>(aN.binlog());

      std::pair<std::vector<uint32_t>, int> t;

      t = aN.split();
      std::vector<uint32_t> a = t.first;
      int aExt = t.second;

      t = split();
      std::vector<uint32_t> c = t.first;
      int cExt = t.second;

      std::vector<int64_t> b;
      int64_t tmp = c[0] * BASE_SQR + c[1];
      b.push_back(tmp / a[0]);
      int64_t r = tmp % a[0];

      int limit = c.size() - a.size() + 1;
      for (int i = 2; i < limit; i++) {
         tmp = r * BASE_SQR + c[i];

         for (int j = 1; j < a.size() && j < i; j++)
            tmp -= a[j] * b[i - j - 1];

         b.push_back(tmp / a[0]);
         r = tmp % a[0];
      }

      // double scale = pow(BASE, (aExt - cExt) - 2 * max(0, (int) (1 + a.size() - c.size())));
      // if (trunc)
      //    *((uint64_t *) res) = comb(b) * scale;
      // else
      //    *((double *) res) = comb(b) * scale;

      Number n = comb(b);
      int shift = 16 * (aExt - cExt) - 2 * std::max(0, (int) (1 + a.size() - c.size()));

      if (shift == 0)
         return n;
      else if (shift > 0)
         return n << shift;
      else
         return n >> (-1 * shift);
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

   Number operator%(const uint32_t a) {
      uint32_t *d = (uint32_t *) data;
      int len = numBytes >> 2;
      uint64_t mod = 0;

      while (len--)
         mod = ((mod<<32) + d[len]) % a;

      Number n(&mod, 4);

      return n;
   }

   // there must be a better way to do these functions...
   Number operator+(const uint32_t a) {
      Number t(&a, 4);
      const Number& r = t;
      return operator+(r);
   }

   Number operator-(const uint32_t a) {
      Number t(&a, 4);
      const Number& r = t;
      return operator-(r);
   }

   Number operator*(const uint32_t a) {
      Number t(&a, 4);
      const Number& r = t;
      return operator*(r);
   }

   Number operator*(const uint64_t a) {
      Number t(&a, 8);
      const Number& r = t;
      return operator*(r);
   }

   Number operator/(const uint32_t a) {
      Number t(&a, 4);
      const Number& r = t;
      return operator/(r);
   }

   Number operator+=(const uint32_t a) {
      Number t(&a, 4);
      const Number& r = t;
      return operator+=(r);
   }

   Number operator-=(const uint32_t a) {
      Number t(&a, 4);
      const Number& r = t;
      return operator-=(r);
   }

   Number operator*=(const uint32_t a) {
      Number t(&a, 4);
      const Number& r = t;
      return operator*=(r);
   }

   Number operator/=(const uint32_t a) {
      Number t(&a, 4);
      const Number& r = t;
      return operator/=(r);
   }

   Number operator&(const uint32_t a) {
      Number t(&a, 4);
      const Number& r = t;
      return operator&(r);
   }

   Number operator|(const uint32_t a) {
      Number t(&a, 4);
      const Number& r = t;
      return operator^(r);
   }

   Number operator^(const uint32_t a) {
      Number t(&a, 4);
      const Number& r = t;
      return operator^(r);
   }

   bool operator==(const uint32_t a) {
      Number t(&a, 4);
      const Number& r = t;
      return operator==(r);
   }

   bool operator!=(const uint32_t a) {
      Number t(&a, 4);
      const Number& r = t;
      return operator!=(r);
   }

   bool operator>(const uint32_t a) {
      Number t(&a, 4);
      const Number& r = t;
      return operator>(r);
   }

   bool operator<(const uint32_t a) {
      Number t(&a, 4);
      const Number& r = t;
      return operator<(r);
   }

   bool operator>=(const uint32_t a) {
      Number t(&a, 4);
      const Number& r = t;
      return operator>=(r);
   }

   bool operator<=(const uint32_t a) {
      Number t(&a, 4);
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
   bool negative;
   int32_t exponent;
   Number mantissa;

public:
   Decimal(unsigned int i) {
      negative = false;
      exponent = 0;
      mantissa = i;
   }

   Decimal(float f) {
      negative = false;
      exponent = 0;
      mantissa = 0;
   }

   Decimal(double d) {
      negative = false;
      exponent = 0;
      mantissa = 0;
   }

   Decimal(Number &n) {
      negative = false;
      exponent = 0;
      mantissa = n;
   }

   Decimal(const Decimal& d) {
      negative = d.negative;
      exponent = d.exponent;
      mantissa = d.mantissa;
   }

   Decimal operator+(const Decimal& a) {
      if (exponent != a.exponent) {
         Decimal tmp((exponent < a.exponent) ? *this : a);
         tmp.mantissa <<= abs(a.exponent - exponent);
         tmp.exponent = std::max(exponent, a.exponent);

         return (exponent < a.exponent) ? (tmp + a) : (operator+(tmp));
      }

      Decimal tmp(a);
      if (negative == a.negative) {
         tmp.mantissa += mantissa;
      } else {
         if (mantissa == a.mantissa)
            return Decimal((unsigned int) 0);
         else if (mantissa < a.mantissa) {
            tmp.negative = a.negative;
            tmp.mantissa -= mantissa;
         }
         else {
            tmp.negative = negative;
            tmp.mantissa = mantissa - a.mantissa;
         }
      }

      return tmp;
   }

   Decimal operator-(const Decimal& a) {
      if (exponent != a.exponent) {
         Decimal tmp((exponent < a.exponent) ? *this : a);
         tmp.mantissa <<= abs(a.exponent - exponent);
         tmp.exponent = std::max(exponent, a.exponent);

         return (exponent < a.exponent) ? (tmp - a) : (operator-(tmp));
      }

      Decimal tmp(a);
      if (negative == a.negative) {
         if (mantissa == a.mantissa)
            return Decimal((unsigned int) 0);
         else if (mantissa < a.mantissa) {
            tmp.negative = !negative;
            tmp.mantissa -= mantissa;
         } else {
            tmp.mantissa += mantissa;
         }
      } else {
         tmp.negative = negative;
         tmp.mantissa += mantissa;
      }

      return tmp;
   }

   Decimal operator*(const Decimal& a) {
      Decimal tmp(a);

      tmp.negative ^= negative;
      tmp.exponent += exponent;
      tmp.mantissa *= mantissa;
      
      return tmp;
   }

   Decimal operator/(const Decimal& a) {
      Decimal tmp(*this);

      tmp.negative ^= a.negative;
      tmp.exponent -= a.exponent;
      tmp.mantissa /= a.mantissa;

      return tmp;
   }
};

#endif