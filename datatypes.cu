#include "datatypes.h"

template <typename VecObject>
__host__
void printList(const VecObject &v) {
   typename VecObject::const_iterator it;
   for (it = v.begin(); it != v.end(); ++it)
      std::cout << *it << " ";
   std::cout << std::endl;
}

inline __host__ __device__
bool numBase2(uint32_t n) {
   return n == 0 || ((n & (n - 1)) == 0);
}

inline __host__ __device__
uint32_t nextBase2(uint32_t n) {
   if (numBase2(n))
      return n;

   uint32_t num;
   for (int i = 0; i < 32; i++)
      if ((num = (1 << i)) >= n)
         return num;
   return num;
}

__host__ __device__
bool topBitsSet(void *data, int len, int numBits) {
   char *d = (char *) data;
   int bytes = numBits / 8, bits = numBits % 8;

   if (numBits > len * 8)
      return true;

   for (int i = 1; i <= bytes; i++)
      if (d[len - i] != 0)
         return true;

   return bits > 0 && (((~((1 << (8 - bits)) - 1)) & d[len - bytes - 1]) != 0);
}

__host__ __device__
uint32_t unsignedAbs(int64_t i) {
   if (i < 0)
      i *= -1;

   return *((uint32_t *) ((void *) &i));
}

template <typename numType>
__host__ __device__
numType max(numType a, numType b) {
   return a > b ? a : b;
}

template <typename numType>
__host__ __device__
numType min(numType a, numType b) {
   return a < b ? a : b;
}

__host__ __device__
unsigned int log2(uint32_t n) {
   unsigned int i = 0;
   while (n) {
      n >>= 1;
      ++i;
   }
   return i - 1;
}

__host__ __device__
bool Number::compare(const Number& a, bool lt) {
   int lSize = numBytes >> 2, rSize = a.numBytes >> 2, len = max(lSize, rSize);
   uint32_t *l = (uint32_t *) data, *r = (uint32_t *) a.data;

   for (int i = len - 1; i >= 0; i--) {
      if ((i < lSize ? l[i] : 0) < (i < rSize ? r[i] : 0))
         return lt;
      if ((i < lSize ? l[i] : 0) > (i < rSize ? r[i] : 0))
         return ! lt;
   }
   return false;
}

inline __host__ __device__
Number& Number::copyIn(Number a) {
   // the deallocater frees the pointer so just swap the pointers to handle the memory properly
   void *tmp = data;
   data = a.data;
   a.data = tmp;
   numBytes = a.numBytes;
   return *this;
}

__host__ __device__
bool Number::isBase2() const {
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
__host__ __device__
int Number::binlog() const {
   uint32_t *ptr = (uint32_t *) data;
   int len = numBytes >> 2;

   for (int i = 0; i < len; i++)
      if (ptr[i] != 0)
         return i*32 + log2(ptr[i]);

   return 0;
}

__host__ __device__
int Number::topBytesEmpty() const {
   int len = numBytes;
   unsigned char *ptr = (unsigned char*)data;

   while (len--)
      if (ptr[len])
         return numBytes - (len + 1);

   return numBytes;
}

__host__ __device__
bool Number::nonZero() const {
   uint32_t *ptr = (uint32_t *) data;
   int len = numBytes >> 2;

   while (len--)
      if (ptr[len])
         return true;

   return false;
}

__host__ __device__
splitInfo_t Number::split() const {
   splitInfo_t info;
   info.extra = 0;

   Number tmp(*this);

   int usedBytes = numBytes - topBytesEmpty();
   int num16 = (usedBytes + 1) / 2; // ceil
   if (num16 & 1) {
      ++num16;
      ++info.extra;
      tmp <<= 16;
   }

   int num32 = (tmp.numBytes - tmp.topBytesEmpty() + 3) / 4; // ceil
   if (num32 == 1) {
      tmp <<= 32;
      ++num32;
      info.extra += 2;
   }

   info.len = num32;
   info.data = (uint32_t *) malloc(num32 * 4);
   while (num32--) {
      info.data[num32] = tmp.getLSU32();
      tmp >>= 32;
   }

   return info;
}

__host__ __device__
Number Number::comb(int64_t *l, int len) {
   Number n(4 * len);
   memset(n.data, 0, n.numBytes);

   for (int i = 0; i < len; i++) {
      n <<= 32;
      bool sub = l[i] < 0;
      uint32_t val = unsignedAbs(l[i]);

      if (sub)
         n -= val;
      else
         n += val;
   }

   return n;
}

__host__ __device__
uint32_t Number::getLSU32() const {
   return *((uint32_t *) data);
}

__host__ __device__
uint32_t Number::getLSU16() const {
   return *((uint16_t *) data);
}

__host__ __device__
Number::Number() {
   numBytes = MIN_BYTES;
   data = malloc(numBytes);
}

__host__ __device__
Number::Number(int bytes) {
   numBytes = nextBase2(max(bytes, MIN_BYTES));
   data = malloc(numBytes);
}

__host__ __device__
Number::Number(const void *bytes, int len) {
   numBytes = nextBase2(len);
   data = malloc(numBytes);
   memset(data, 0, numBytes);
   memcpy(data, bytes, len);
}

__host__ __device__
Number::Number(const Number& num) {
   numBytes = num.numBytes;
   data = malloc(numBytes);

   memcpy(data, num.data, numBytes);
}

__host__ __device__
Number::~Number() {
   free(data);
}

__host__ __device__
void Number::resize(int bytes) {
   free(data);
   numBytes = nextBase2(max(bytes, MIN_BYTES));
   data = malloc(numBytes);
}

__host__ __device__
Number& Number::operator=(const Number& a) {
   if (this == &a)
      return *this;

   memset(data, 0, numBytes);
   memcpy(data, a.data, min(numBytes, a.numBytes));
   return *this;
}

__host__ __device__
Number& Number::operator=(unsigned int a) {
   memset(data, 0, numBytes);
   ((unsigned int *)data)[0] = a;
   return *this;
}

__host__ __device__
Number& Number::operator=(uint64_t a) {
   if (numBytes < 8)
      resize(8);

   memset(data, 0, numBytes);
   ((uint64_t *)data)[0] = a;
   return *this;
}

__host__ __device__
Number Number::operator+(const Number& a) {
   Number n(max(numBytes, a.numBytes));

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

__host__ __device__
Number Number::operator-(const Number& a) {
   Number n(max(numBytes, a.numBytes));

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

__host__ __device__
Number Number::operator*(const Number& a) {
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

__host__ __device__
Number Number::operator/(const Number& aN) {
   if (! (nonZero() && aN.nonZero())) {
      return *this;
   }

   if (aN.isBase2())
      return operator>>(aN.binlog());

   splitInfo_t a, c;

   a = aN.split();
   c = split();

   int limit = c.len - a.len + 1;
   int64_t *b = (int64_t *) malloc(limit * 8);

   int64_t tmp = c.data[0] * BASE_SQR + c.data[1];
   b[0] = (tmp / a.data[0]);
   int64_t r = tmp % a.data[0];

   for (int i = 2; i < limit; i++) {
      tmp = r * BASE_SQR + c.data[i];

      for (int j = 1; j < a.len && j < i; j++)
         tmp -= a.data[j] * b[i - j - 1];

      b[i - 1] = (tmp / a.data[0]);
      r = tmp % a.data[0];
   }

   free(a.data);
   free(c.data);

   Number n = comb(b, limit);
   free(b);
   int shift = 16 * ((a.extra - c.extra) - 2 * max(0, (int) (1 + a.len - c.len)));

   if (shift == 0)
      return n;
   else if (shift > 0)
      return n << shift;
   else
      return n >> (-shift);
}

__host__ __device__
Number Number::operator<<(const int a) {
   int bytes = a / 8;
   int bits = a % 8;

   int clearBytes = topBytesEmpty();
   int overflow = max(0, bytes + (bits > 0 ? 1 : 0) - clearBytes);

   Number t(numBytes + overflow);
   memset(t.data, 0, t.numBytes);

   char *ptr = (char *) t.data;
   if (clearBytes < numBytes)
      memcpy(ptr + bytes, data, numBytes - clearBytes);

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

__host__ __device__
Number Number::operator>>(const int a) {
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

__host__ __device__
Number Number::operator&(const Number& a) {
   Number n(min(numBytes, a.numBytes));

   memset(n.data, 0, n.numBytes);

   uint32_t *l = (uint32_t *) data, *r = (uint32_t *) a.data, *v = (uint32_t *) n.data;

   for (int i = 0; i*4 < n.numBytes; i++)
      v[i] = l[i] & r[i];

   return n;
}

__host__ __device__
Number Number::operator|(const Number& a) {
   Number n(max(numBytes, a.numBytes));

   int lSize = numBytes >> 2, rSize = a.numBytes >> 2, len = n.numBytes >> 2;
   uint32_t *l = (uint32_t *) data, *r = (uint32_t *) a.data, *v = (uint32_t *) n.data;

   for (int i = 0; i < len; i++)
      v[i] = (i < lSize ? l[i] : 0) | (i < rSize ? r[i] : 0);

   return n;
}

__host__ __device__
Number Number::operator^(const Number& a) {
   Number n(max(numBytes, a.numBytes));

   int lSize = numBytes >> 2, rSize = a.numBytes >> 2, len = n.numBytes >> 2;
   uint32_t *l = (uint32_t *) data, *r = (uint32_t *) a.data, *v = (uint32_t *) n.data;

   for (int i = 0; i < len; i++)
      v[i] = (i < lSize ? l[i] : 0) ^ (i < rSize ? r[i] : 0);

   return n;
}

__host__ __device__
Number& Number::operator+=(const Number& a) {
   return copyIn(operator+(a));
}

__host__ __device__
Number& Number::operator-=(const Number& a) {
   return copyIn(operator-(a));
}

__host__ __device__
Number& Number::operator*=(const Number& a) {
   return copyIn(operator*(a));
}

__host__ __device__
Number& Number::operator/=(const Number& a) {
   return copyIn(operator/(a));
}

__host__ __device__
Number& Number::operator<<=(const int a) {
   return copyIn(operator<<(a));
}

__host__ __device__
Number& Number::operator>>=(const int a) {
   return copyIn(operator>>(a));
}

__host__ __device__
Number& Number::operator&=(const Number& a) {
   return copyIn(operator&(a));
}

__host__ __device__
Number& Number::operator|=(const Number& a) {
   return copyIn(operator|(a));
}

__host__ __device__
Number& Number::operator^=(const Number& a) {
   return copyIn(operator^(a));
}

__host__ __device__
Number& Number::operator&=(const uint32_t a) {
   return copyIn(operator&(a));
}

__host__ __device__
Number& Number::operator|=(const uint32_t a) {
   return copyIn(operator|(a));
}

__host__ __device__
Number& Number::operator^=(const uint32_t a) {
   return copyIn(operator^(a));
}

__host__ __device__
bool Number::operator==(const Number& a) {
   int lSize = numBytes >> 2, rSize = a.numBytes >> 2, len = max(lSize, rSize);
   uint32_t *l = (uint32_t *) data, *r = (uint32_t *) a.data;

   for (int i = 0; i < len; i++)
      if ((i < lSize ? l[i] : 0) != (i < rSize ? r[i] : 0))
         return false;
   return true;
}

__host__ __device__
bool Number::operator!=(const Number& a) {
   return ! operator==(a);
}

__host__ __device__
bool Number::operator>(const Number& a) {
   return compare(a, false);
}

__host__ __device__
bool Number::operator<(const Number& a) {
   return compare(a, true);
}

__host__ __device__
bool Number::operator>=(const Number& a) {
   return ! operator<(a);
}

__host__ __device__
bool Number::operator<=(const Number& a) {
   return ! operator>(a);
}

__host__ __device__
Number Number::operator%(const uint32_t a) {
   uint32_t *d = (uint32_t *) data;
   int len = numBytes >> 2;
   uint64_t mod = 0;

   while (len--)
      mod = ((mod<<32) + d[len]) % a;

   Number n(&mod, 4);

   return n;
}

__host__ __device__
Number Number::operator+(const uint32_t a) {
   Number t(&a, 4);
   return operator+(t);
}

__host__ __device__
Number Number::operator-(const uint32_t a) {
   Number t(&a, 4);
   return operator-(t);
}

__host__ __device__
Number Number::operator*(const uint32_t a) {
   Number t(&a, 4);
   return operator*(t);
}

__host__ __device__
Number Number::operator*(const uint64_t a) {
   Number t(&a, 8);
   return operator*(t);
}

__host__ __device__
Number Number::operator/(const uint32_t a) {
   Number t(&a, 4);
   return operator/(t);
}

__host__ __device__
Number Number::operator+=(const uint32_t a) {
   Number t(&a, 4);
   return operator+=(t);
}

__host__ __device__
Number Number::operator-=(const uint32_t a) {
   Number t(&a, 4);
   return operator-=(t);
}

__host__ __device__
Number Number::operator*=(const uint32_t a) {
   Number t(&a, 4);
   return operator*=(t);
}

__host__ __device__
Number Number::operator/=(const uint32_t a) {
   Number t(&a, 4);
   return operator/=(t);
}

__host__ __device__
Number Number::operator&(const uint32_t a) {
   Number t(&a, 4);
   return operator&(t);
}

__host__ __device__
Number Number::operator|(const uint32_t a) {
   Number t(&a, 4);
   return operator^(t);
}

__host__ __device__
Number Number::operator^(const uint32_t a) {
   Number t(&a, 4);
   return operator^(t);
}

__host__ __device__
bool Number::operator==(const uint32_t a) {
   Number t(&a, 4);
   return operator==(t);
}

__host__ __device__
bool Number::operator!=(const uint32_t a) {
   Number t(&a, 4);
   return operator!=(t);
}

__host__ __device__
bool Number::operator>(const uint32_t a) {
   Number t(&a, 4);
   return operator>(t);
}

__host__ __device__
bool Number::operator<(const uint32_t a) {
   Number t(&a, 4);
   return operator<(t);
}

__host__ __device__
bool Number::operator>=(const uint32_t a) {
   Number t(&a, 4);
   return operator>=(t);
}

__host__ __device__
bool Number::operator<=(const uint32_t a) {
   Number t(&a, 4);
   return operator<=(t);
}

__host__ __device__
void Number::trim() {
   char* ptr = (char *) data;
   int used;

   for (int i = numBytes - 1; i >= 0; i--) {
      if (ptr[i]) {
         used = i + 1;
         break;
      }
   }

   int newBytes = max(nextBase2(used), (uint32_t) MIN_BYTES);
   if (newBytes < numBytes) {
      void *smaller = malloc(newBytes);
      memcpy(smaller, data, newBytes);
      free(data);
      data = smaller;
      numBytes = newBytes;
   }
}

__host__ __device__
void* Number::getData() {
   void *ptr = malloc(numBytes);
   memcpy(ptr, data, numBytes);
   return ptr;
}

__host__ __device__
int Number::getSize() {
   return numBytes;
}

__host__
std::ostream& operator<<(std::ostream& os, const Number& n) {
   std::ios::fmtflags flags = os.flags();
   int width = os.width();

   int pos = n.numBytes - n.topBytesEmpty();
   if (pos < 1)
      pos = 1;

   unsigned char *ptr = (unsigned char*)n.data;

   os << "0x";
   while (pos--)
      os << std::noshowbase << std::hex << std::setw(2) << std::setfill('0') << (int)ptr[pos];

   os.width(width);
   os.flags(flags);

   return os;
}

__host__ __device__
bool Decimal::compare(const Decimal& a, bool lt) {
   if (negative && !a.negative)
      return lt;
   if (!negative && a.negative)
      return ! lt;

   if (exponent != a.exponent) {
      Decimal tmp((exponent < a.exponent) ? *this : a);
      tmp.mantissa <<= abs(a.exponent - exponent);
      tmp.exponent = max(exponent, a.exponent);

      return (exponent < a.exponent) ? tmp.compare(a, lt) : compare(tmp, lt);
   }

   if (lt)
     return mantissa < a.mantissa;

   return mantissa > a.mantissa;
}

inline __host__ __device__
Decimal& Decimal::copyIn(Decimal d) {
   negative = d.negative;
   exponent = d.exponent;
   mantissa = d.mantissa;
   return *this;
}

__host__ __device__
Decimal::Decimal(unsigned int i) {
   negative = false;
   exponent = 0;
   mantissa = i;
}

__host__ __device__
Decimal::Decimal(float f) {
   union {
      float f;
      uint32_t i;
   } q;
   q.f = f;

   negative = (q.i >> 31) != 0;
   exponent = (q.i >> 23) & ((1 << 8) - 1);
   mantissa = q.i & ((1 << 23) - 1);
}

__host__ __device__
Decimal::Decimal(double d) {
   union {
      double d;
      uint64_t i;
   } q;
   q.d = d;

   mantissa.resize(8);
   negative = q.i & (1ULL << 63);
   exponent = (q.i >> 52) & ((1 << 11) - 1);
   mantissa = (uint64_t) (q.i & ((1ULL << 52) - 1));
}

__host__ __device__
Decimal::Decimal(Number &n) {
   negative = false;
   exponent = 0;
   mantissa = n;
}

__host__ __device__
Decimal::Decimal(const Decimal& d) {
   negative = d.negative;
   exponent = d.exponent;
   mantissa = d.mantissa;
}

__host__ __device__
Decimal& Decimal::operator=(const Decimal& a) {
   if (this == &a)
      return *this;

   negative = a.negative;
   exponent = a.exponent;
   mantissa = a.mantissa;
   return *this;
}

__host__ __device__
Decimal Decimal::operator+(const Decimal& a) {
   if (exponent != a.exponent) {
      Decimal tmp((exponent < a.exponent) ? *this : a);
      tmp.mantissa <<= abs(a.exponent - exponent);
      tmp.exponent = max(exponent, a.exponent);

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

__host__ __device__
Decimal Decimal::operator-(const Decimal& a) {
   if (exponent != a.exponent) {
      Decimal tmp((exponent < a.exponent) ? *this : a);
      tmp.mantissa <<= abs(a.exponent - exponent);
      tmp.exponent = max(exponent, a.exponent);

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
         tmp.mantissa = mantissa - tmp.mantissa;
      }
   } else {
      tmp.negative = negative;
      tmp.mantissa += mantissa;
   }

   return tmp;
}

__host__ __device__
Decimal Decimal::operator*(const Decimal& a) {
   Decimal tmp(a);

   tmp.negative ^= negative;
   tmp.exponent += exponent;
   tmp.mantissa *= mantissa;

   return tmp;
}

__host__ __device__
Decimal Decimal::operator/(const Decimal& a) {
   Decimal tmp(*this);

   tmp.negative ^= a.negative;
   tmp.exponent -= a.exponent;
   tmp.mantissa /= a.mantissa;

   return tmp;
}

__host__ __device__
bool Decimal::operator>(const Decimal& a) {
   return compare(a, false);
}

__host__ __device__
bool Decimal::operator<(const Decimal& a) {
   return compare(a, true);
}

__host__ __device__
bool Decimal::operator>=(const Decimal& a) {
   return ! operator<(a);
}

__host__ __device__
bool Decimal::operator<=(const Decimal& a) {
   return ! operator>(a);
}

__host__ __device__
bool Decimal::operator==(const Decimal& a) {
   if (negative != a.negative)
      return false;

   if (exponent != a.exponent) {
      Decimal tmp((exponent < a.exponent) ? *this : a);
      tmp.mantissa <<= abs(a.exponent - exponent);
      tmp.exponent = max(exponent, a.exponent);

      return (exponent < a.exponent) ? (tmp.mantissa == a.mantissa) : (mantissa == tmp.mantissa);
   }

   return mantissa == a.mantissa;
}

__host__ __device__
bool Decimal::operator>(const uint32_t a) {
   Decimal r(a);
   return operator>(r);
}

__host__ __device__
bool Decimal::operator<(const uint32_t a) {
   Decimal r(a);
   return operator<(r);
}

__host__ __device__
bool Decimal::operator>=(const uint32_t a) {
   Decimal r(a);
   return operator>=(r);
}

__host__ __device__
bool Decimal::operator<=(const uint32_t a) {
   Decimal r(a);
   return operator<=(r);
}

__host__ __device__
Decimal& Decimal::operator+=(const Decimal& d) {
   return copyIn(operator+(d));
}

__host__ __device__
Decimal& Decimal::operator/=(const Decimal& d) {
   return copyIn(operator/(d));
}
