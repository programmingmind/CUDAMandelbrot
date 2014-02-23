#include "datatypes.h"

#include <iostream>
#include <string>

void printResult(std::string prefix, bool b) {
   std::cout << prefix << ": " << (b ? "pass" : "fail") << std::endl;
}

int main() {
   Number a(4);
   Number b(4);

   unsigned int l = 0x56af6f4d;
   unsigned int r = 0x2308ffed;

   Number c(&l, 4);

   a = l;
   b = r;

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

   return 0;
}