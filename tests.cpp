#include "datatypes.h"

#include <iostream>

int main() {
   Number a(4);
   Number b(4);

   unsigned int l = 0x56af6f4d;
   unsigned int r = 0x2308ffed;

   a = l;
   b = r;

   uint32_t *sum = (uint32_t *) (a+b).getData();
   uint32_t *diff = (uint32_t *) (a-b).getData();

   if (*sum == (l+r))
      std::cout << "Pass" << std::endl;
   else
      std::cout << "Fail" << std::endl;

   if (*diff == (l-r))
      std::cout << "Pass" << std::endl;
   else
      std::cout << "Fail" << std::endl;

   free(sum);
   free(diff);

   return 0;
}