import numpy

from . import basis

def part_factorial(X, n):
    if n==0:
        return X**0
    return X*part_factorial(X-1, n-1)

class FockBasis(basis.ONBasis):
    rank = 1
    N0 = None
    N1 = None
    dtype = None

    def __init__(self, N0, N1=None, dtype=None):
        if N1 is None:
            self.N0 = 0
            self.N1 = N0
        else:
            self.N0 = N0
            self.N1 = N1
        self.dtype = dtype

    def basis_vector(self, i):
        from ..statevector import StateVector
        assert self.N0 <= i < self.N1
        if self.dtype is None:
            X = numpy.zeros(self.N1-self.N0, dtype=complex)
            X[i-self.N0] = 1
        else:
            X = numpy.empty(self.N1-self.N0, dtype=self.dtype)
            for j in range(0, len(X)):
                X[j] = self.dtype(0)
            X[i-self.N0] = self.dtype(1)
        return StateVector(X, basis=self)

    def identity(self):
        from ..operators import Operator as op
        N = self.N1-self.N0
        if self.dtype is None:
            X = numpy.eye(N)
        else:
            X = numpy.array(list(self.dtype(1))*N)
            X = numpy.diag(X)
        return op(X, basis=self)

    def number(self):
        from ..operators import Operator as op
        if self.dtype is None:
            X = numpy.arange(self.N0, self.N1)
        else:
            X = numpy.array(list(self.dtype(i) for i in range(self.N0,self.N1)))
        return op(numpy.diag(X), basis=self)

    def destroy(self, pow=1):
        from ..operators import Operator as op
        N0 = self.N0 + pow
        N1 = self.N1
        if self.dtype is None:
            X = numpy.arange(N0, N1)
        else:
            X = numpy.array(list(self.dtype(i) for i in range(N0,N1)))
        X = part_factorial(X, pow)**(0.5)
        return op(numpy.diag(X, pow), basis=self)

    def create(self, pow=1):
        return self.destroy(pow=pow).T

    def coherent_state(self, alpha, dtype=complex):
        from ..statevector import StateVector as sv
        x = numpy.empty(self.N1, dtype=dtype)
        x[0] = dtype("1")
        for n in range(1, self.N1):
            x[n] = x[n-1]*alpha/(n)**(0.5)
        if isinstance(dtype, complex):
            a = numpy.exp(-numpy.abs(alpha)**2/2.)
        else:
            import mpmath
            a = mpmath.exp(-mpmath.norm(alpha)**2/2)
        x = a*x[self.N0:self.N1]
        return sv(x, basis=self)

def coherent_state(alpha, N, dtype=complex):
    from .. import statevector
    x = numpy.empty(N, dtype=dtype)
    x[0] = 1
    for n in range(1,N):
        x[n] = x[n-1]*alpha/numpy.sqrt(n)
    x = numpy.exp(-numpy.abs(alpha)**2/2.)*x
    return statevector.StateVector(x)