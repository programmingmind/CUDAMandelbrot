def split(n):
    s = str(n)
    if (len(s) % 2 == 1):
        s += "0"
    a = []
    for i in range(0, int(len(s)/2)):
        a.append(int (s[i * 2 : (i+1) * 2]))
    return a

def comb(l):
    n = 0
    for e in l:
        n = n*100 + e
    return n

def div(dividend, divisor, trunc = True, round = False):
    c = split(dividend)
    a = split(divisor)

    factor = 1
    if (len(str(dividend)) % 2 == 1):
        factor /= 10
    if (len(str(divisor)) % 2 == 1):
        factor *= 10
    
    b = []
    tmp = (c[0] * 100 + c[1])
    b.append(int(tmp / a[0]))
    r = tmp % a[0]

    for i in range(2, len(c) - len(a) + 1):
        tmp = r * 100 + c[i]
        for j in range(1, min(len(a), i)):
            tmp -= a[j] * b[i - j - 1]
        b.append(int(tmp / a[0]))
        r = tmp % a[0]

    res = comb(b) * factor
    if (len(a) >= len(c)):
        res /= (100**(1 + len(a) - len(c)))

    if (trunc):
        return int(res) + (1 if round and r >= 50 else 0)

    return res

def test(dividend, divisor, trunc = True):
    dq = div(dividend, divisor, trunc)
    q = dividend/divisor
    if (trunc):
        q = int(q)

    print(str(dividend) + " / " + str(divisor) + " = " + str(q))
    
    if (dq == q):
        print("pass")
    else:
        print("fail: " + str(dq))

test(499084777422, 6534)
test(100000, 314159, False)
test(875352, 6543)
test(876342615243, 73524377)
test(654, 23)
