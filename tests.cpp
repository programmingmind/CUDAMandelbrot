#include "common.h"
#include "datatypes.h"

#include <iostream>
#include <string>

#define ALL 0
#define WARNING 1
#define ERROR 2

#define MAX_ERROR 0.0000001

static int errorLevel = ALL;

// use this for equality
template <typename DataType, typename CompareType>
void printResult(std::string prefix, DataType a, CompareType b, bool exact = true) {
   if (a == b) {
      if (errorLevel <= ALL)
         std::cout << prefix << ": " << "pass" << std::endl;
   } else if (!exact && (a >= b ? ((a - b) < MAX_ERROR) : ((a - b) > -MAX_ERROR))) {
      if (errorLevel <= WARNING)
         std::cout << prefix << ": " << "Warn (diff: " << (a - b) << ") -- " << a << "\t" << b << std::endl;
   } else {
      std::cout << prefix << ": " << "!!FAIL!! -- " << a << "\t" << b << std::endl;
   }
}

// use this for boolean comparators
void printResult(std::string prefix, bool b) {
   if (!b || errorLevel == ALL)
      std::cout << prefix << ": " << (b ? "pass" : "!!FAIL!!") << std::endl;
}

int main(int argc, char *argv[]) {
   unsigned int l = 0x56af6f4d;
   unsigned int r = 0x2308ffed;

   errorLevel = argc > 1 ? atoi(argv[1]) : ALL;

   int i = 4;
   Number four(&i, 4);
   Number sixteen = four<<2;

   printResult("partial shift left", four*four, sixteen);
   printResult("partial shift right", sixteen>>2, four);

   printResult("integer addition", four + 12, sixteen);
   printResult("integer subtraction", sixteen - 12, four);
   printResult("integer multiplication", four * 4U, sixteen);
   printResult("integer division", sixteen / 4, four);

   Number a(4);
   Number b(4);

   a = l;
   b = r;

   // std::cout << a << "\t" << b << std::endl;

   Number c(&l, 4);

   Number d(4);
   d = l;

   uint32_t *bigData = (uint32_t *) ((d<<32) | r).getData();

   printResult("left shift, or 1", bigData[0], r);
   printResult("left shift, or 2", bigData[1], l);
   free(bigData);

   Number n(&l, 4);
   n += b;
   uint32_t *nData = (uint32_t *) n.getData();
   printResult("+=", *nData, (l+r));
   printResult("+= == +", n, (a + b));
   free(nData);

   uint32_t *cData = (uint32_t *) c.getData(), *aData = (uint32_t *) a.getData();

   printResult("constructor from pointer", *cData, *aData);
   free(cData);
   free(aData);

   uint32_t *sum = (uint32_t *) (a+b).getData();
   uint32_t *diff = (uint32_t *) (a-b).getData();

   printResult("sum", *sum, (l+r));
   printResult("diff", *diff, (l-r));

   free(sum);
   free(diff);

   printResult("==", a == c);
   printResult("!=", b != c);
   printResult(">", (d<<32) > c);
   printResult("<", b < a);

   printResult("< int", b < l);
   printResult("> int", a > r);

   uint64_t prod = 0xbdd085c01afbd49ULL;
   uint64_t *prodData = (uint64_t *) (a*b).getData();
   printResult("prod", *prodData, prod);

   free(prodData);

   Number nThree(4);
   nThree = 3U;
   Number nFortyEight(4);
   nFortyEight = 48U;

   std::cout<<(nThree << 4)<<std::endl;
   std::cout<<(nThree << 64)<<std::endl;
   std::cout<<(nThree << 32)<<std::endl;
   std::cout<<(nThree << 68)<<std::endl;

   std::cout<<(nThree << 68)/480U<<std::endl;

   printResult("division #1", (a*b)/a, b);
   printResult("division #2", (a*b)/b, a);
   printResult("division #3", (nThree / nThree), 1U);
   printResult("division #4", (nFortyEight / nThree), 16U);

   uint64_t prod16 = 0x56af6f4d0ULL;
   Number n16(&prod16, 8);

   uint64_t *shiftData = (uint64_t *) (a<<4).getData();
   printResult("32 bit value << partial byte", *shiftData, prod16);
   free(shiftData);

   uint32_t *rShiftData = (uint32_t *) (n16>>4).getData();
   printResult("shift it back right", *rShiftData, l);
   free(rShiftData);

   rShiftData = (uint32_t *) (n16>>12).getData();
   printResult("shift right over a byte", *rShiftData, (l >> 8));
   free(rShiftData);

   uint64_t *prod16Data = (uint64_t *) (a*sixteen).getData();
   printResult("prod16", *prod16Data, prod16);
   free(prod16Data);

   prod16Data = (uint64_t *) (a*((uint32_t)16)).getData();
   printResult("prod16 int", *prod16Data, prod16);
   free(prod16Data);

   printResult("division by power of 2", sixteen/four, four);
   printResult("compound multiplication, division by power of 2", (a*sixteen)/four, a*four);

   // Mersenne prime for modulus M_89 = 1<<89 - 1
   uint32_t zero = 0;
   Number mod(&zero, 4);
   mod += 1;
   mod <<= 89;
   mod -= 1;

   uint32_t div = 3465847321LL;
   uint32_t remainder = 3191376254LL;

   printResult("modulus with remainder", mod % div == remainder);

   mod &= 0;
   mod += 1024;
   printResult("modulus without remainder", mod % 32 == 0);

   printResult("float vs double constructor 1", Decimal(5.0), Decimal(5.0f));
   printResult("float vs double constructor 2", Decimal(7.25), Decimal(7.25f));
   printResult("float vs double constructor 3", Decimal(0.0), Decimal(0.0f));
   printResult("float vs double constructor 4", Decimal(-1.375), Decimal(-1.375f));

   printResult("Number vs double constructor", Decimal(5.0), Decimal(5U));

   printResult("constructor with 0", Decimal(0.0) == 0U);

   Decimal dFour(4.0);
   Decimal dTwelve(12.0);
   Decimal dSixteen(16.0);

   /*std::cout<<"zero   : " << Decimal(0.0) << std::endl;
   std::cout<<"four   : " << dFour << std::endl;
   std::cout<<"twelve : " << dTwelve << std::endl;
   std::cout<<"sixteen: " << dSixteen << std::endl;*/

   printResult("decimal addition", dFour + dTwelve, dSixteen);
   printResult("decimal subtraction", dSixteen - dTwelve, dFour);
   printResult("decimal multiplication", dFour * dFour, dSixteen);
   printResult("decimal division", dSixteen / dFour, dFour);
   printResult("decimal division 2", Decimal(48U) / Decimal(3.0), dSixteen);

   printResult("decimal greater than", dTwelve > dFour);
   printResult("decimal less than", dTwelve < dSixteen);

   Decimal dAssign(0.0);
   dAssign = dFour;
   printResult("decimal assign", dAssign, dFour);

   Decimal dNegThree(-3.0);
   Decimal dNegFour(-4.0);
   Decimal dNegTwelve(-12.0);
   Decimal dNegSixteen(-16.0);
   Decimal dNegFortyEight(-48.0);

   Decimal dZero(0.0);
   printResult("decimal greater than zero", dSixteen > dZero);
   printResult("decimal neg less than zero", dNegSixteen < dZero);

   printResult("decimal neg addition", dSixteen + dNegFour, dTwelve);
   printResult("decimal neg subtraction", dFour - dNegTwelve, dSixteen);
   printResult("decimal neg multiplcation", dFour * dNegTwelve, dNegFortyEight);
   printResult("decimal neg division", dNegFortyEight / dNegThree, dSixteen);

   printResult("decimal neg greater than", dNegTwelve > dNegSixteen);
   printResult("decimal neg less than", dNegSixteen < dNegFour);

   // Mandelbrot tests
   Decimal startX = -1.50;
   Decimal startY = -1.00;
   Decimal resolution = INITIAL_RESOLUTION;

   std::cout<<startX<<"\t"<<startY<<std::endl;

   std::cout<<"resolution: " << resolution<<std::endl;
   std::cout<<resolution/WIDTH<<std::endl;
   std::cout<<(resolution*2U)/WIDTH<<std::endl;
   std::cout<<startX + resolution/WIDTH<<std::endl;

   for (unsigned int xNdx = 2; xNdx < 3; xNdx++) {
      for (unsigned int yNdx = 2; yNdx < 3; yNdx++) {
         Decimal x0(0U), y0(0U);

         x0 = startX + ((resolution * xNdx) / WIDTH);
         y0 = startY + ((resolution * yNdx) / HEIGHT);

         printResult("x0", x0, Decimal(-1.50 + ((INITIAL_RESOLUTION * xNdx) / WIDTH)));
         printResult("y0", y0, Decimal(-1.00 + ((INITIAL_RESOLUTION * yNdx) / HEIGHT)));

         std::cout << x0 << "\t" << (-1.50 + ((INITIAL_RESOLUTION * xNdx) / WIDTH)) << std::endl;
         std::cout << y0 << "\t" << (-1.00 + ((INITIAL_RESOLUTION * yNdx) / HEIGHT)) << std::endl;
      }
   }  

   return 0;
}
