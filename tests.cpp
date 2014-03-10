#include "datatypes.h"

#include <iostream>
#include <string>

void printResult(std::string prefix, bool b) {
   std::cout << prefix << ": " << (b ? "pass" : "fail") << std::endl;
}

int main() {
   unsigned int l = 0x56af6f4d;
   unsigned int r = 0x2308ffed;

   int i = 4;
   Number four(&i, 4);
   Number sixteen = four<<2;

   printResult("partial shift left", four*four == sixteen);
   printResult("partial shift right", sixteen>>2 == four);

   printResult("integer addition", four+12 == sixteen);
   printResult("integer subtraction", sixteen - 12 == four);
   printResult("integer multiplication", four* ((uint32_t)4) == sixteen);
   printResult("integer division", sixteen / 4 == four);

   Number a(4);
   Number b(4);

   a = l;
   b = r;

   std::cout << a << "\t" << b << std::endl;

   Number c(&l, 4);

   Number d(4);
   d = l;

   uint32_t *bigData = (uint32_t *) ((d<<32) | r).getData();

   printResult("left shift, or 1", bigData[0] == r);
   printResult("left shift, or 2", bigData[1] == l);
   free(bigData);

   Number n(&l, 4);
   n += b;
   uint32_t *nData = (uint32_t *) n.getData();
   printResult("+=", *nData == (l+r));
   printResult("+= == +", n == (a + b));
   free(nData);

   uint32_t *cData = (uint32_t *) c.getData(), *aData = (uint32_t *) a.getData();

   printResult("constructor from pointer", *cData == *aData);
   free(cData);
   free(aData);

   uint32_t *sum = (uint32_t *) (a+b).getData();
   uint32_t *diff = (uint32_t *) (a-b).getData();

   printResult("sum", *sum == (l+r));
   printResult("diff", *diff == (l-r));

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
   printResult("prod", *prodData == prod);

   free(prodData);

   printResult("division #1", (a*b)/a == b);
   printResult("division #2", (a*b)/b == a);

   uint64_t prod16 = 0x56af6f4d0ULL;
   Number n16(&prod16, 8);

   uint64_t *shiftData = (uint64_t *) (a<<4).getData();
   printResult("32 bit value << partial byte", *shiftData == prod16);
   free(shiftData);

   uint32_t *rShiftData = (uint32_t *) (n16>>4).getData();
   printResult("shift it back right", *rShiftData == l);
   free(rShiftData);

   rShiftData = (uint32_t *) (n16>>12).getData();
   printResult("shift right over a byte", *rShiftData == (l >> 8));
   free(rShiftData);

   uint64_t *prod16Data = (uint64_t *) (a*sixteen).getData();
   printResult("prod16", *prod16Data == prod16);
   free(prod16Data);

   prod16Data = (uint64_t *) (a*((uint32_t)16)).getData();
   printResult("prod16 int", *prod16Data == prod16);
   free(prod16Data);

   printResult("division by power of 2", sixteen/four == four);
   printResult("compound multiplication, division by power of 2", (a*sixteen)/four == a*four);

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

   Decimal dFour(4.0);
   Decimal dTwelve(12.0);
   Decimal dSixteen(16.0);

   printResult("decimal addition", dFour + dTwelve == dSixteen);
   printResult("decimal subtraction", dSixteen - dTwelve == dFour);
   printResult("decimal multiplication", dFour * dFour == dSixteen);
   printResult("decimal division", dSixteen / dFour == dFour);

   printResult("decimal greater than", dTwelve > dFour);
   printResult("decimal less than", dTwelve < dSixteen);

   Decimal dAssign(0.0);
   dAssign = dFour;
   printResult("decimal assign", dAssign == dFour);

   Decimal dNegThree(-3.0);
   Decimal dNegFour(-4.0);
   Decimal dNegTwelve(-12.0);
   Decimal dNegSixteen(-16.0);
   Decimal dNegFortyEight(-48.0);

   printResult("decimal neg addition", dSixteen + dNegFour == dTwelve);
   printResult("decimal neg subtraction", dFour - dNegTwelve == dNegSixteen);
   printResult("decimal neg multiplcation", dFour * dNegTwelve == dNegFortyEight);
   printResult("decimal neg division", dNegFortyEight / dNegThree == dSixteen);

   printResult("decimal neg greater than", dNegTwelve > dNegSixteen);
   printResult("decimal neg less than", dNegSixteen < dNegFour);

   return 0;
}
