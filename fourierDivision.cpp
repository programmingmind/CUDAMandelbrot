#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>

#include <inttypes.h>
#include <math.h>
#include <string.h>

using namespace std;

#define MAX_ERROR 0.0000001

#ifdef SMALL
#define BASE 10
#define BASE_SQR 100
#else
#define BASE 65536
#define BASE_SQR 4294967296ULL
#endif


template <typename VecObject>
void printList(const VecObject &v) {
   #ifndef DEBUG
   return;
   #endif

   typename VecObject::const_iterator it;
   for (it = v.begin(); it != v.end(); ++it)
      cout << *it << " ";
   cout << endl;
}

pair<vector<uint32_t>, int> split(uint64_t number) {
   vector<uint32_t> t;
   int extra = 0;

   while (number > 0) {
      t.insert(t.begin(), number % BASE);
      number /= BASE;
   }

   if (t.size() % 2 == 1) {
      t.push_back(0);
      extra++;
   }

   vector<uint32_t> a;
   for (int i = 0; i < t.size(); i += 2)
      a.push_back(t[i] * BASE + t[i + 1]);

   if (a.size() == 1) {
      a.push_back(0);
      extra += 2;
   }

   return make_pair(a, extra);
}

uint64_t comb(vector<int64_t> l) {
   uint64_t n = 0;
   for (vector<int64_t>::iterator it = l.begin(); it != l.end(); it++)
      n = n*BASE_SQR + *it;
   return n;
}

void div(uint64_t dividend, uint64_t divisor, void *res, bool trunc) {
   if (divisor == 0 || dividend == 0) {
      memset(res, (divisor == 0 ? 0xFF : 0), 8);
      return;
   }

   pair<vector<uint32_t>, int> t;

   t = split(divisor);
   vector<unsigned int> a = t.first;
   int aExt = t.second;
   printList(a);

   t = split(dividend);
   vector<unsigned int> c = t.first;
   int cExt = t.second;
   printList(c);

   vector<int64_t> b;
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

   printList(b);

   double scale = pow(BASE, (aExt - cExt) - 2 * max(0, (int) (1 + a.size() - c.size())));
   if (trunc)
      *((uint64_t *) res) = comb(b) * scale;
   else
      *((double *) res) = comb(b) * scale;

   // *((double *) res) = ((double) comb(b)) * pow(BASE, aExt - cExt) / pow(BASE_SQR, 1 + a.size() - c.size());
}

void test(uint64_t dividend, uint64_t divisor, bool trunc = true) {
   double fd, d;
   uint64_t fi, i;

   if (divisor == 0) {
      cout << "Divide by 0" << endl;
      div(dividend, divisor, &fi, trunc);
      cout << "handled " << (~fi == 0 ? "correctly" : "incorrectly") << endl << endl;
      return;
   }

   if (trunc) {
      div(dividend, divisor, &fi, trunc);
      i = dividend / divisor;
   } else {
      div(dividend, divisor, &fd, trunc);
      d = ((double) dividend) / divisor;
   }

   double err = trunc ? (fi > i ? (fi - i) : (i - fi)) : fabs(fd - d);

   cout << dividend << " / " << divisor << " = " << (trunc ? i : d) << endl;
   cout << (err < MAX_ERROR ? "pass" : "fail") << " error: " << err << endl << endl;
}

int main() {
   test(499084777422ULL, 6534);
   test(100000, 314159, false);
   test(875352, 6543);
   test(876342615243ULL, 73524377);
   test(654, 23);
   test(87897432432ULL, 67676237);
   test(6633554627152354545ULL, 856352525354556ULL);
   test(100000000000ULL, 314159265359ULL, false);
   test(12345678, 123);
   test(12345678, 12345);
   test(12345678, 1234567);
   test(0, 65567536576355ULL);
   test(0, 67868764857673ULL, false);
   test(767565, 0);
   test(7657785, 0, false);
   test(854848695528570185ULL, 1454337869);
   test(854848695528570185ULL, 587792365);
   return 0;
}