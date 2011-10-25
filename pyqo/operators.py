from functools import reduce
import numpy
from . import statevector

class Operator(numpy.ndarray):
    r"""
    A class representing an operator in a specific basis.

    *Usage*
        >>> A = Operator(((0,1), (1,0)))
        >>> print(A)

    *Arguments*
        * *data*
            Anything that can be used to create a numpy (ij..) x (ij..) array,
            e.g. a nested tuple or another numpy array.

        * Any other argument that a numpy array takes. E.g. ``copy=False`` can
          be used so that the Operator shares the data storage with the
          given numpy array.
    """
    def __new__(cls, data, **kwargs):
        array = numpy.array(data, dtype=complex, **kwargs)
        shape = array.shape
        rank = len(shape)
        if rank%2 != 0 or shape[:(rank//2)] != shape[(rank//2):]:
            raise ValueError("Operators map from H->H and therefor must"
                             "be square!")
        return numpy.asarray(array).view(cls)

    def __str__(self):
        clsname = self.__class__.__name__
        rank = len(self.shape)
        dims = " x ".join(map(str, self.shape[:rank//2]))
        array = numpy.ndarray.__str__(self)
        return "%s\n%s -> %s\n%s" % (clsname, dims, dims, array)

    @property
    def T(self):
        rank = len(self.shape)
        ind = tuple(range(rank))
        return self.transpose(ind[rank//2:]+ind[:rank//2])

    @property
    def H(self):
        return self.T.conj()

    def tensor(self, array):
        if not isinstance(array, Operator):
            array = Operator(array, copy=False)
        op = numpy.multiply.outer(self, array)
        rank1 = len(self.shape)
        m = rank1//2
        rank2 = len(array.shape)
        n = rank2//2
        R = lambda a,b:tuple(range(a,b))
        perm = R(0,m) + R(rank1, rank1+n) + \
               R(m, rank1) + R(rank1+n,rank1+rank2)
        return Operator(numpy.transpose(op, perm), copy=False)

    __xor__ = tensor

    def __mul__(self, other):
        if isinstance(other, Operator):
            if self.shape == other.shape or self.shape == other.shape*2:
                rank = len(self.shape)
                return self.__class__(numpy.tensordot(self,other,rank//2))
            elif self.shape*2 == other.shape:
                rank = len(self.shape)
                return self.__class__(numpy.tensordot(self,other,rank))
            else:
                raise ValueError("Operators are dimensional incompatible")
        elif isinstance(other, statevector.StateVector):
            if self.shape != other.shape*2:
                raise ValueError("Operator and state vector are dimensional"
                                 "incompatible")
            rank = len(self.shape)
            return other.__class__(numpy.tensordot(self,other,rank//2))
        else:
            return numpy.ndarray.__mul__(self, other)


sigmax = Operator( ((0,1),(1,0)) )
sigmay = Operator( ((0,-1j),(1j,0)) )
sigmaz = Operator( ((1,0),(0,-1)) )
sigmap = Operator( ((0,0),(1,0)) )
sigmam = Operator( ((0,1),(0,0)) )

def identity(x):
    if isinstance(x, int):
        return Operator(numpy.eye(x))
    elif isinstance(x, Operator):
        shape = x.shape
        rank = len(shape)
        id = (Operator(numpy.eye(N)) for N in shape[:rank//2])
        return reduce(Operator.tensor,id)
    else:
        raise TypeError("Unsupported argument type.")

def destroy(N):
    return Operator(numpy.diag(numpy.arange(1,N)**(0.5),1))

def create(N):
    return Operator(numpy.diag(numpy.arange(1,N)**(0.5),-1))

def number(N):
    return Operator(numpy.diag(numpy.arange(N)))

def spre(op):
    return op^identity(op)

def spost(op):
    return identity(op)^op.T

def liouvillian(H, J=()):
    L = -1j*(spre(H) - spost(H))
    for j in J:
        n = j.H*j/2.
        L += spre(j)*spost(j.H) - spost(n) - spre(n)
    return L

def ptrace(op, indices):
    n = len(op.shape)//2
    if isinstance(indices, int):
        indices = (indices,)
    for i in indices:
        assert 0<=i<n
        op = op.trace(axis1=i, axis2=i+n)
    return Operator(op)

def qfunc(state, X, Y, g=2**(1/2)):
    rank = len(state.shape)
    n = state.shape[0]
    if rank == 1:
        @numpy.vectorize
        def Q(a):
            c = statevector.coherent(n,a)
            return numpy.abs(numpy.dot(c.conj(), state))**2
    if rank == 2:
        @numpy.vectorize
        def Q(a):
            c = statevector.coherent(n,a)
            return numpy.dot(c.conj(), numpy.dot(state,c))
    else:
        ValueError("The given state has a too high rank.")
    alpha = g*(numpy.array(X) + 1j*numpy.array(Y))/2
    return Q(alpha)



