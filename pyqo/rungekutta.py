import numpy

class RungeKutta:
    a = None
    b = None
    c = None
    s = None
    def __init__(self, a, b, c):
        assert len(a)+1 == len(b) == len(c)
        self.s = s = len(b)
        a_ = [[]]
        for i in range(1,s):
            assert len(a[i-1]) == i
            a_.append(numpy.array(a[i-1]))
        self.a = a_
        self.b = numpy.array(b)
        self.c = numpy.array(c)

    def K(self, f, y0, t0, h, y1=None):
        K = []
        if y1 is None:
            y1 = y0.copy()
        for i in range(self.s):
            t = t0 + self.c[i]*h
            y1[:] = y0
            for j, k in enumerate(K):
                y1 += h*self.a[i][j]*k
            K.append(f(t,y1))
        return K

    def step(self, f, t0, h, y0):
        y1 = y0.copy()
        K = self.K(f, y0, t0, h, y1)
        y1[:] = y0
        for i in range(self.s):
            y1 += h*self.b[i]*K[i]
        return y1

class EmbeddedRungeKutta(RungeKutta):
    def __init__(self, a, b, c, d, order):
        RungeKutta.__init__(self, a, b, c)
        assert len(d) == self.s
        self.d = numpy.array(d)
        self.order = order

    def step(self, f, y0, t0, h):
        y1 = y0.copy()
        K = self.K(f, y0, t0, h, y1)
        e = y0.copy()
        e[:] = 0
        y1[:] = y0
        for i in range(self.s):
            y1 += h*self.b[i]*K[i]
            e += h*self.d[i]*K[i]
        return y1, e

    def integrate(self, f, y0, T, h0=None, atol=1e-7, rtol=1e-7, h_min=1e-9, S=0.95):
        result = [y0]
        y = y0.copy()
        t = T[0]
        t_next = T[1]
        i = 1
        if h0 is None:
            h = min((atol + abs(numpy.dot(y0.conj().flat, y0.flat))*rtol)*1e4, t_next - t)
        else:
            h = h_start
        while True:
            # Make one step
            y1, delta = self.step(f, y, t, h)
            # Calculate error and new stepsize
            scale = atol + abs(y1)*rtol
            err = (((abs(delta)/scale)**2).sum()/delta.size)**(0.5)
            mu = S*(1/err)**(1./self.order)
            mu = min(max(0.2, mu), 10)
            h_next = mu*h
            if h_next < h_min:
                raise Exception("Minimal stepsize reached.")
            if err > 1:
                # Repeat step with smaller stepsize
                h = h_next
                continue
            y = y1
            t = t+h
            h = h_next
            if t == t_next:
                result.append(y.copy())
                i += 1
                if i==len(T):
                    break
                t_next = T[i]
            if t + h > t_next:
                h = t_next-t
            elif t + 4*h/3 > t_next:
                h = 2*h/3
        return result


def RK1(dtype=float):
    _0 = dtype("0")
    _1 = dtype("1")
    b = (_1,)
    c = (_0,)
    a = []
    return RungeKutta(a,b,c)

def RK2(dtype=float):
    _0 = dtype("0")
    _1 = dtype("1")
    _2 = dtype("2")
    a = ((_1,),)
    b = (_1/_2, _1/_2)
    c = (_0, _1)
    return RungeKutta(a,b,c)

def RK3(dtype=float):
    _0 = dtype("0")
    _1 = dtype("1")
    _2 = dtype("2")
    _3 = dtype("3")
    _6 = dtype("6")
    a = ((_1/_2,), (-_1, _2))
    b = (_1/_6, _2/_3, _1/_6)
    c = (_0, _1/_2, _1)
    return RungeKutta(a,b,c)

def RK4(dtype=float):
    _0 = dtype("0")
    _1 = dtype("1")
    _2 = dtype("2")
    _3 = dtype("3")
    _6 = dtype("6")
    a = ((_1/_2,), (_0, _1/_2), (_0, _0, _1))
    b = (_1/_6, _1/_3, _1/_3, _1/_6)
    c = (_0, _1/_2, _1/_2, _1)
    return RungeKutta(a,b,c)

# Heun-Euler
def RK1_2(dtype=float):
    _0 = dtype("0")
    _1 = dtype("1")
    _2 = dtype("2")
    a = ((_1,),)
    b = (_1/_2, _1/_2)
    c = (_0, _1)
    d = (-_1/_2, _1/_2)
    return EmbeddedRungeKutta(a,b,c,d,order=2)

# Bogacki-Shampine
def RK2_3(dtype=float):
    _0 = dtype("0")
    _1 = dtype("1")
    _2 = dtype("2")
    _3 = dtype("3")
    _4 = dtype("4")
    _7 = dtype("7")
    _8 = dtype("8")
    _9 = dtype("9")
    _24 = dtype("24")
    a = ((_1/_2,), (_0, _3/_4), (_2/_9, _1/_3, _4/_9))
    b = (_2/_9, _1/_3, _4/_9, _0)
    c = (_0, _1/_2, _3/_4, _1)
    d = (_2/_9-_7/_24, _1/_3-_1/_4, _4/_9-_1/_3, -_1/_8)
    return EmbeddedRungeKutta(a,b,c,d,order=3)

# Fehlberg
def RK4_5(dtype=float):
    _0 = dtype("0")
    _1 = dtype("1")
    _2 = dtype("2")
    _3 = dtype("3")
    _4 = dtype("4")
    _5 = dtype("5")
    _8 = dtype("8")
    _9 = dtype("9")
    _11 = dtype("11")
    _12 = dtype("12")
    _13 = dtype("13")
    _16 = dtype("16")
    _25 = dtype("25")
    _27 = dtype("27")
    _32 = dtype("32")
    _40 = dtype("40")
    _50 = dtype("50")
    _55 = dtype("55")
    _135 = dtype("135")
    _216 = dtype("216")
    _439 = dtype("439")
    _513 = dtype("513")
    _845 = dtype("845")
    _1408 = dtype("1408")
    _1859 = dtype("1859")
    _1932 = dtype("1932")
    _2197 = dtype("2197")
    _2565 = dtype("2565")
    _3544 = dtype("3544")
    _3680 = dtype("3680")
    _4104 = dtype("4104")
    _6656 = dtype("6656")
    _7200 = dtype("7200")
    _7296 = dtype("7296")
    _12825 = dtype("12825")
    _28561 = dtype("28561")
    _56430 = dtype("56430")
    a = ((_1/_4,),
         (_3/_32, _9/_32),
         (_1932/_2197, -_7200/_2197, _7296/_2197),
         (_439/_216, -_8, _3680/_513, -_845/_4104),
         (-_8/_27, _2, -_3544/_2565, _1859/_4104, -_11/_40))
    b = (_16/_135, _0, _6656/_12825, _28561/_56430,-_9/_50, _2/_55)
    c = (_0, _1/_4, _3/_8, _12/_13, _1, _1/_2)
    d = (_16/_135-_25/_216, _0, _6656/_12825-_1408/_2565,
         _28561/_56430-_2197/_4104,-_9/_50+_1/_5, _2/_55)
    return EmbeddedRungeKutta(a,b,c,d,order=5)

if __name__ == "__main__":
    b = []
    f = lambda t, y:t
    #print(RK4_5().d)
    print(RK4_5().integrate(f, numpy.array([0], dtype=float), [0,0.1,0.2,1], rtol=1e-12, atol=1e-12))
