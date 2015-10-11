#include "BigFloat.h"

#ifdef _WIN32
#ifdef __CUDACC__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define HOST __host__
#define DEVICE __device__
#endif
#endif

typedef union {
  double d;
  struct {
    uint64_t mantissa : 52;
    uint64_t exponent : 11;
    uint64_t sign : 1;
  } parts;
} double_cast;

HOST DEVICE
int msbPos(uint64_t val) {
   if (val) {
      int pos = 64;
      uint64_t check = ((uint64_t)1) << 63;

      while ((val & check) == 0) {
         --pos;
         check >>= 1;
      }
      return pos;
   }

   return 0;
}

HOST DEVICE
BigFloat* normalize(BigFloat *val) {
   int msByte = -1;

   for (int i = 0; i < BF_SIZE - 1; i++) {
      val->data[i+1] += val->data[i] >> 32;
      val->data[i]   &= 0xFFFFFFFF;

      if (val->data[i]) {
         msByte = i;
      }
   }

   int MSB = msbPos(val->data[BF_SIZE-1]);
   if (MSB == 0) {
      if (msByte < 0) {
         val->exponent = 0;
         val->negative = 0;
         return val;
      }

      for (int i = msByte; i >= 0; --i) {
         val->data[i + (BF_SIZE-1 - msByte)] = val->data[i];
         val->data[i] = 0;
      }
      val->exponent -= (BF_SIZE-1 - msByte) * 32;

      MSB = msbPos(val->data[BF_SIZE-1]);
   }

   if (MSB > 32) {
      uint64_t toAdd = 0;
      for (int i = BF_SIZE-1; i >= 0; --i) {
         val->data[i] |= toAdd;
         toAdd = (val->data[i] & ((1 << (MSB-32)) - 1)) << 32;
         val->data[i] >>= MSB-32;
      }
      val->exponent += MSB-32;
   } else if (MSB < 32) {
      uint64_t toAdd = 0;
      for (int i = 0; i < BF_SIZE; i++) {
         val->data[i] = (val->data[i] << (32-MSB)) | toAdd;
         toAdd = val->data[i] >> 32;
         val->data[i] &= 0xFFFFFFFF;
      }
      val->exponent -= 32-MSB;
   }

   return val;
}

HOST DEVICE
BigFloat* init(BigFloat *val, uint32_t number) {
   val->negative = 0;
   val->exponent = 0;
   val->data[0] = number;
   for (int i = 1; i < BF_SIZE; i++) {
      val->data[i] = 0;
   }
   return normalize(val);
}

HOST
BigFloat* initDouble(BigFloat *val, double number) {
   double_cast dc;
   dc.d = number;

   val->negative = dc.parts.sign;
   if (dc.parts.exponent) {
      val->exponent = dc.parts.exponent - 1023 - 52;
      val->data[0] = ((uint64_t)1)<<52 | dc.parts.mantissa;
   } else {
      val->exponent = 1 - 1023 - 52;
      val->data[0] = dc.parts.mantissa;
   }

   for (int i = 1; i < BF_SIZE; i++) {
      val->data[i] = 0;
   }

   return normalize(val);
}

HOST DEVICE
void assign(BigFloat *dest, BigFloat *src) {
   dest->negative = src->negative;
   dest->exponent = src->exponent;
   for (int i = 0; i < BF_SIZE; i++) {
      dest->data[i] = src->data[i];
   }
}

HOST DEVICE
int isZero(BigFloat *val) {
   return val->data[BF_SIZE-1] == 0;
}

HOST DEVICE
int magCmp(BigFloat *one, BigFloat *two) {
   if (one->exponent != two->exponent) {
      return one->exponent < two->exponent ? LT : GT;
   }

   for (int i = BF_SIZE-1; i >= 0; --i) {
      if (one->data[i] != two->data[i]) {
         return one->data[i] < two->data[i] ? LT : GT;
      }
   }

   return EQ;
}

HOST DEVICE
int cmp(BigFloat *one, BigFloat *two) {
   if (one->negative != two->negative) {
      return one->negative ? LT : GT;
   }

   return (one->negative ? -1 : 1) * magCmp(one, two);
}

HOST DEVICE
int base2Cmp(BigFloat *val, int32_t power) {
   if (isZero(val)) {
      return LT;
   }

   power -= BF_SIZE*32 - 1;
   if (val->exponent < power) {
      return LT;
   } else if (val->exponent > power) {
      return GT;
   }

   if (val->data[BF_SIZE-1] & 0x7FFFFFFF) {
      return GT;
   }

   for (int i = BF_SIZE-2; i >= 0; i--) {
      if (val->data[i]) {
         return GT;
      }
   }

   return EQ;
}

HOST DEVICE
BigFloat* shiftL(BigFloat *val, uint64_t amount) {
   val->exponent += amount;
   return val;
}

HOST DEVICE
BigFloat* shiftR(BigFloat *val, uint64_t amount) {
   val->exponent -= amount;
   return val;
}

HOST DEVICE
BigFloat* add(BigFloat *one, BigFloat *two, BigFloat *result) {
   if (one->negative != two->negative) {
      // positive + negative = positive - positive(-1*negative)
      // negative + positive = negative - negative(-1*positive)
      two->negative ^= 1;
      (void)sub(one, two, result); // already normalizes result
      two->negative ^= 1;
      return result;
   }

   for (int i = 0; i < BF_SIZE; i++) {
      result->data[i] = 0;
   }
   result->negative = one->negative;

   BigFloat *larger = one;
   BigFloat *smaller = two;
   if (magCmp(larger, smaller) == LT) {
      larger = two;
      smaller = one;
   }
   result->exponent = larger->exponent;

   int ndxDiff = 1  + (larger->exponent - smaller->exponent) / 32;
   int bitDiff = 32 - (larger->exponent - smaller->exponent) % 32;
   
   for (int i = 0; i < BF_SIZE; i++) {
      result->data[i] += larger->data[i];

      if (i - ndxDiff + 1 >= 0) {
         uint64_t tmp = smaller->data[i] << bitDiff;

         result->data[i - ndxDiff + 1] += tmp >> 32;

         if (i - ndxDiff >= 0) {
            result->data[i - ndxDiff] += tmp & 0xFFFFFFFF;
         }
      }
   }

   return normalize(result);
}

HOST DEVICE
void carryToNdx(BigFloat *val, const int ndx) {
   int startNdx = ndx;
   while (++startNdx < BF_SIZE && val->data[startNdx] == 0) ;

   while (startNdx > ndx) {
      if (val->data[startNdx] & 0xFFFFFFFF) {
         val->data[startNdx-1] |= val->data[startNdx] << 32;
         val->data[startNdx] &= 0xFFFFFFFF00000000;
      } else {
         val->data[startNdx  ] -= 0x00000000FFFFFFFF;
         val->data[startNdx-1] |= 0xFFFFFFFF00000000;
      }
      startNdx--;
   }
}

HOST DEVICE
BigFloat* sub(BigFloat *one, BigFloat *two, BigFloat *result) {
   if (one->negative != two->negative) {
      // negative - positive = negative + negative(-1*positive)
      // positive - negative = positive + positive(-1*negative)
      two->negative ^= 1;
      (void)add(one, two, result); // already normalizes result
      two->negative ^= 1;
      return result;
   }

   for (int i = 0; i < BF_SIZE; i++) {
      result->data[i] = 0;
   }

   BigFloat *larger = one;
   BigFloat *smaller = two;
   result->negative = larger->negative;
   if (magCmp(larger, smaller) == LT) {
      larger = two;
      smaller = one;
      result->negative ^= 1;
   }
   result->exponent = larger->exponent;

   int ndxDiff = 1  + (larger->exponent - smaller->exponent) / 32;
   int bitDiff = 32 - (larger->exponent - smaller->exponent) % 32;
   
   // Because we carry from larger to smaller, take care of larger first so when
   // we carry we don't have to uncarry later
   for (int i = BF_SIZE-1; i >= 0; i--) {
      result->data[i] += larger->data[i];

      if (i - ndxDiff < BF_SIZE) {
         uint64_t tmp = smaller->data[i] << bitDiff;

         if (i - ndxDiff + 1 < BF_SIZE && i - ndxDiff + 1 >= 0) { 
            uint64_t upper = tmp >> 32;

            if (result->data[i - ndxDiff + 1] < upper) {
               carryToNdx(result, i - ndxDiff + 1);
            }
            result->data[i - ndxDiff + 1] -= upper;
         }

         if (i - ndxDiff > 0) {
            uint64_t lower = tmp & 0xFFFFFFFF;

            if (result->data[i - ndxDiff] < lower) {
               carryToNdx(result, i - ndxDiff);
            }
            result->data[i - ndxDiff] -= lower;
         }
      }
   }

   return normalize(result);
}

HOST DEVICE
BigFloat* mult(BigFloat *one, BigFloat *two, BigFloat *result, BigFloat *tmp) {
   result->negative = one->negative ^ two->negative;
   result->exponent = one->exponent + two->exponent + BF_SIZE*32;

   for (int i = 0; i < BF_SIZE; i++) {
      result->data[i] = 0;
      tmp->data[i] = 0;
   }

   for (int i = BF_SIZE - 1; i >= 0; --i) {
      for (int j = BF_SIZE - 1; j >= 0; --j) {
         uint64_t prod = one->data[i] * two->data[j];
         int topNdx = i + j - (BF_SIZE - 1);
         if (topNdx > 0) {
            result->data[topNdx  ] += prod >> 32;
            result->data[topNdx-1] += prod & 0xFFFFFFFF;
         } else {
            tmp->data[BF_SIZE + topNdx  ] += prod >> 32;
            tmp->data[BF_SIZE + topNdx-1] += prod & 0xFFFFFFFF;
         }
      }
   }

   for (int i = 0; i < BF_SIZE - 1; i++) {
      tmp->data[i+1] += tmp->data[i] >> 32;
   }
   result->data[0] += tmp->data[BF_SIZE-1] >> 32;

   return normalize(result);
}

HOST
std::ostream& operator<<(std::ostream& os, const BigFloat& bf) {
   std::ios::fmtflags flags = os.flags();
   int width = os.width();

   int pos = BF_SIZE;

   if (bf.negative) {
      os << "- ";
   }

   os << "0x";
   while (pos--)
      os << std::noshowbase << std::hex << std::setw(8) << std::setfill('0') << (uint32_t)(bf.data[pos]) << " ";

   os.width(width);
   os.flags(flags);

   os << " x 2^(" << bf.exponent << ")";

   return os;
}

HOST
BigFloat& BigFloat::operator=(const double d) {
   return *initDouble(this, d);
}

HOST
BigFloat& BigFloat::operator+=(BigFloat bf) {
   BigFloat temp;
   assign(&temp, this);
   return *add(&temp, &bf, this);
}

HOST
BigFloat BigFloat::operator*(const unsigned int i) {
   BigFloat result;
   BigFloat temp;
   BigFloat multiplier;

   mult(this, init(&multiplier, i), &result, &temp);
   return result;
}

HOST
BigFloat& BigFloat::operator>>=(const int i) {
   return *shiftR(this, i);
}

HOST
BigFloat BigFloat::operator>>(const int i) {
   BigFloat val;
   assign(&val, this);
   return *shiftR(&val, i);
}
