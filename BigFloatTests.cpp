#include <iostream>

#include "BigFloat.h"
#include "common.h"

using namespace std;

int main() {
   BigFloat a;
   BigFloat b;
   BigFloat c;
   BigFloat d;

   a = -1.50;
   b = 2.00;
   cout << "a: " << a << endl;
   cout << "b: " << b << endl;

   init(&a, 5);
   init(&b, 2000000000);
   add(&a, &b, &c);
   sub(&c, &a, &d);

   cout << "a: " << a << endl;
   cout << "b: " << b << endl;
   cout << "c: " << c << endl;
   cout << "d: " << d << endl;

   BigFloat temp;
   BigFloat multTemp;

   mult(&a, &b, &c, &multTemp);
   sub(&c, &a, &d);

   cout << "c: " << c << endl;
   cout << "d: " << d << endl;

   init(&temp, 1);
   cout << "1? " << base2Cmp(&temp, 1) << endl;
   init(&temp, 2);
   cout << "2? " << base2Cmp(&temp, 1) << endl;
   init(&temp, 3);
   cout << "3? " << base2Cmp(&temp, 1) << endl;

   uint32_t it=0; 
   BigFloat x0;
   BigFloat y0;
   BigFloat x;
   BigFloat y;
   BigFloat xSqr;
   BigFloat ySqr;

   init(&x0, 0);
   init(&y0, 0);
   assign(&x, &x0);
   assign(&y, &y0);

   while (it < MAX && base2Cmp(add(mult(&x, &x, &xSqr, &multTemp), mult(&y, &y, &ySqr, &multTemp), &temp), 2) != GT) {
      (void)add(&y0, shiftL(mult(&x, &y, &temp, &multTemp), 1), &y);
      (void)add(sub(&xSqr, &ySqr, &temp), &x0, &x);
      it++;
   }

   cout << "it: " << it << endl;

   return 0;
}
