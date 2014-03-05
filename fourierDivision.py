def split(n, base):
    t = []
    extra = False
    
    while (n > 0):
        t.insert(0, n % base)
        n = int(n / base)

    if (len(t) % 2 == 1):
        t.append(0)
        extra = True

    a = []
    for i in range(0, len(t), 2):
        a.append(t[i] * base + t[i + 1])
    if (len(a) == 1):
        a.append(0)

    return a, extra

def comb(l, base):
    n = 0
    for e in l:
        n = n*(base**2) + e
    return n

def div(dividend, divisor, base, trunc, round = False):
    c, cExt = split(dividend, base)
    a, aExt = split(divisor, base)

    factor = 1
    if aExt:
        factor *= base
    if cExt:
        factor /= base
    
    b = []
    tmp = (c[0] * (base**2) + c[1])
    b.append(int(tmp / a[0]))
    r = tmp % a[0]

    for i in range(2, len(c) - len(a) + 1):
        tmp = r * (base**2) + c[i]
        for j in range(1, min(len(a), i)):
            tmp -= a[j] * b[i - j - 1]
        b.append(int(tmp / a[0]))
        r = tmp % a[0]

    res = comb(b, base) * factor
    if (len(a) >= len(c)):
        res /= (base**2)**(1 + len(a) - len(c))

    if (trunc):
        return int(res) + (1 if round and r >= (base / 2) else 0)

    return res

def test(dividend, divisor, base = 10, trunc = True):
    dq = div(dividend, divisor, base, trunc)
    q = dividend/divisor
    if (trunc):
        q = int(q)

    print(str(dividend) + " / " + str(divisor) + " = " + str(q))
    
    if (dq == q):
        print("pass")
    else:
        print("fail: " + str(dq) + " diff : " + str(abs(dq - q)))

    print()

bases = [10, 16, 32768]
for base in bases:
    print("Testing with base: " + str(base) + "\n")
    
    test(499084777422, base, 6534)
    test(100000, 314159, base, False)
    test(875352, 6543, base)
    test(876342615243, 73524377, base)
    test(654, 23, base)
    test(87897432432, 67676237, base)
    test(6633554627152354545, 856352525354556, base)
    test(100000000000, 314159265359, base, False)
    test(12345678, 123, base)
    test(12345678, 12345, base)
    test(12345678, 1234567, base)
