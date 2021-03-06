from functools import reduce
import itertools
import numpy

from . import ndarray
from . import statevector
from . import bases
from .utils import list_functions

class BaseOperator:
    pass

class Operator(ndarray.Array, BaseOperator):
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
    @staticmethod
    def _check(array):
        shape = array.shape
        rank = array.ndim
        if rank%2 != 0 or shape[:(rank//2)] != shape[(rank//2):]:
            raise ValueError("Operators must be square!")

    def __str__(self):
        clsname = self.__class__.__name__
        rank = self.ndim
        dims = " x ".join(map(str, self.shape[:rank//2]))
        array = numpy.ndarray.__str__(self)
        return "%s\n%s -> %s\n%s" % (clsname, dims, dims, array)

    def _apply(self, func, left=True, right=True):
        rank = self.ndim // 2
        dims = map(range, self.shape[:rank])
        d = self.copy()
        if left:
            for comb in itertools.product(*dims):
                slice = (Ellipsis,)*rank + comb
                d[slice] = func(d[slice])
        d_H = d.H
        if right:
            for comb in itertools.product(*dims):
                slice = (Ellipsis,)*rank + comb
                d[slice] = func(d_H[slice])
        return d

    def dual(self, left=True, right=True):
        if self.basis is None or self.basis.is_ON:
            return self
        else:
            return self._apply(self.basis.dual, left, right)

    def inverse_dual(self, left=True, right=True):
        if self.basis is None or self.basis.is_ON:
            return self
        else:
            return self._apply(self.basis.inverse_dual, left, right)

    @property
    def T(self):
        rank = self.ndim
        ind = tuple(range(rank))
        return self.transpose(ind[rank//2:]+ind[:rank//2])

    @property
    def H(self):
        return self.T.conj()

    def dagger(self):
        if self.basis is None:
            return self.H
        else:
            return self.basis.dagger(self)

    def ptrace(self, indices):
        if isinstance(indices, int):
            indices = (indices,)
        else:
            indices = list_functions.sorted_list(indices, True)
        if len(indices)==0:
            return self
        rank = self.ndim//2
        assert indices[-1] < rank
        mixed = self.inverse_dual(left=True, right=False)
        for i in indices:
            mixed = mixed.trace(axis1=i, axis2=i + mixed.ndim//2)
        if self.basis is None:
            b = None
        else:
            b = self.basis.ptrace(indices)
        return self.__class__(mixed, basis=b).dual(left=True, right=False)

    def tensor(self, array):
        if not isinstance(array, Operator):
            array = Operator(array, copy=False)
        op = numpy.multiply.outer(self, array)
        rank1 = self.ndim
        m = rank1//2
        rank2 = array.ndim
        n = rank2//2
        R = lambda a,b:tuple(range(a,b))
        perm = R(0,m) + R(rank1, rank1+n) + \
               R(m, rank1) + R(rank1+n,rank1+rank2)
        b = bases.compose_bases(self.basis, rank1, array.basis, rank2)
        return self.__class__(numpy.transpose(op, perm), basis=b, copy=False)

    __xor__ = tensor

    def __mul__(self, other):
        if isinstance(other, DensityOperator):
            if self.shape != other.shape:
                raise ValueError("Dimensions incompatible!")
            if self.basis != other.basis:
                raise ValueError("Bases incompatible!")
            rank = self.ndim
            return other.__class__(numpy.tensordot(self, other, rank//2),
                    basis=self.basis)
        elif isinstance(other, Operator):
            if self.shape != other.shape:
                raise ValueError("Dimensions incompatible!")
            if self.basis != other.basis:
                raise ValueError("Bases incompatible!")
            rank = self.ndim
            return self.__class__(numpy.tensordot(self, other, rank//2),
                    basis=self.basis)
        elif isinstance(other, statevector.StateVector):
            if self.shape != other.shape*2:
                raise ValueError("Operator and state vector are dimensional"
                                 "incompatible")
            if self.basis != other.basis:
                raise ValueError("Bases incompatible!")
            rank = self.ndim
            return other.__class__(numpy.tensordot(self,other,rank//2),
                    basis=self.basis)
        else:
            return numpy.ndarray.__mul__(self, other)

    def identity(self):
        if self.basis is None:
            raise NotImplementedError()
        else:
            return self.basis.identity()

    def __pow__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        if other==0:
            return self.identity()
        else:
            op = self
            for i in range(other-1):
                op = op * self
            return op

class DensityOperator(Operator):
    def ptrace_do(self, indices):
        if isinstance(indices, int):
            indices = (indices,)
        else:
            indices = list_functions.sorted_list(indices, True)
        rank = len(shape)//2
        assert indices[-1] < rank
        mixed = self.dual(left=True, right=False)
        for i in indices[::-1]:
            mixed = mixed.trace(axis1=i, axis2=i + mixed.ndim//2)
        return mixed.inverse_dual(left=True, right=False)


sigmax = Operator( ((0,1),(1,0)) )
sigmay = Operator( ((0,-1j),(1j,0)) )
sigmaz = Operator( ((1,0),(0,-1)) )
sigmap = Operator( ((0,0),(1,0)) )
sigmam = Operator( ((0,1),(0,0)) )

def identity(x):
    if isinstance(x, int):
        return Operator(numpy.eye(x))
    elif isinstance(x, Operator):
        rank = x.ndim
        id = (Operator(numpy.eye(N)) for N in x.shape[:rank//2])
        return Operator(reduce(Operator.tensor,id), basis=x.basis)
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

def qfunc(state, X, Y=None):
    rank = state.ndim
    n = state.shape[0]
    if isinstance(state, statevector.StateVector):
        assert state.ndim == 1
        if isinstance(state.basis, bases.CoherentBasis):
            @numpy.vectorize
            def Q(a):
                c = state.basis.coherent_scalar_product(a, state.basis.states)
                return numpy.abs(numpy.dot(c, state))**2/numpy.pi
        else:
            @numpy.vectorize
            def Q(a):
                N = state.shape[0]
                if state.basis is None:
                    c = bases.fock_basis.FockBasis(N).coherent_state(a)
                    c.basis = None
                else:
                    c = state.basis.coherent_state(a)
                return numpy.abs(numpy.dot(c.conj(), state))**2/numpy.pi
    elif isinstance(state, DensityOperator):
        assert state.ndim == 2
        if isinstance(state.basis, bases.CoherentBasis):
            @numpy.vectorize
            def Q(a):
                c = state.basis.coherent_scalar_product(a.tolist(), state.basis.states)
                c = statevector.StateVector(c)
                return numpy.dot(c.conj(), numpy.dot(state,c))/numpy.pi
        else:
            @numpy.vectorize
            def Q(a):
                N = state.shape[0]
                if state.basis is None:
                    c = bases.fock_basis.FockBasis(N).coherent_state(a)
                    c.basis = None
                else:
                    c = state.basis.coherent_state(a)
                return numpy.dot(c.conj(), state*c)/numpy.pi
    else:
        raise ValueError("The given state has a too high rank.")
    alpha = numpy.array(X) if Y is None else (numpy.array(X) + 1j*numpy.array(Y))
    return Q(alpha)



