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

   Number a(4);
   Number b(4);

   a = l;
   b = r;

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

   printResult("division by power of 2", sixteen/four == four);
   printResult("compound multiplication, division by power of 2", (a*sixteen)/four == a*four);

   return 0;
}