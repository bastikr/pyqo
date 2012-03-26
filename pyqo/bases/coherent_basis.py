"""
Provides functionality to calculate in coherent bases.
"""
import itertools
import numpy

try:
    import mpmath
except:
    print("mpmath not found - Calculating coherent bases not possible")

from . import basis
from .. import ndarray
from ..utils import lattice

class CoherentBasis(basis.Basis):
    """
    A class representing a basis defined by selected coherent vectors.

    It calculates the transformation matrix and provides methods to calculate
    in this basis.

    *Arguments*
        * *states*
            A vector of complex numbers defining the basis states.
        * *is_dual*
            If False this basis consists of coherent states, otherwise it's
            the dual basis.
    """
    rank = 1
    states = None
    lattice = None
    bra_is_dual = None
    ket_is_dual = None
    _trafo = None
    _inv_trafo = None

    def __init__(self, states, ket_is_dual=False, bra_is_dual=False, **args):
        if isinstance(states, lattice.Lattice):
            self.states = ndarray.Array(states.states())
            self.lattice = states
        else:
            self.states = ndarray.Array(states)
        self.bra_is_dual = bra_is_dual
        self.ket_is_dual = ket_is_dual
        self._trafo = args.get("_trafo", None)
        self._inv_trafo = args.get("_inv_trafo", None)

    def __repr__(self):
        name = "%s.%s" % (self.__module__, self.__class__.__name__)
        if self.lattice is None:
            states = repr(self.states)
        else:
            states = repr(self.lattice)
        return "%s(%s, bra_is_dual=%s, ket_is_dual=%s)" %\
                    (name, states, self.bra_is_dual, self.ket_is_dual)

    def __eq__(self, other):
        if self is other:
            return True
        if self.__class__ != other.__class__:
            return False
        if self.bra_is_dual != other.bra_is_dual:
            return False
        if self.ket_is_dual != other.ket_is_dual:
            return False
        if self.states == other.states:
            return False
        return True

    def __add__(self, other):
        if isinstance(other, CoherentBasis):
            # TODO: Implement addition of lattices
            #if self.lattice is not None and other.lattice is not None:
            #    arg = (self.lattice + other.lattice)
            #else:
            #    arg = numpy.concatenate(self.states, other.states)
            arg = numpy.concatenate((self.states, other.states))
            return self.__class__(arg)
        else:
            return NotImplemented

    @property
    def trafo(self):
        if self._trafo is None:
            self._trafo = self.coherent_scalar_product(self.states, self.states)
        return self._trafo

    @property
    def inv_trafo(self):
        if self._inv_trafo is None:
            M = mpmath.matrix(self.trafo.tolist())
            self._inv_trafo = ndarray.Array((M**-1).tolist())
        return self._inv_trafo

    @staticmethod
    def create_hexagonal_grid_rings(center, d, rings):
        """
        Create a new coherent basis with basis states on a hexagonal grid.

        The basis states are choosen in rings around the center on a hexagonal
        grid with lattice constant d.

        *Arguments*
            * *center*
                A complex number defining the center of the rings.

            * *d*
                A real number defining the lattice constant.

            * *rings*
                An integer giving the number of rings.
        """
        center = mpmath.mpmathify(center)
        d = mpmath.mpf(d)
        lat = lattice.HexagonalLattice(d, center, dtype=mpmath.mpf)
        for i in range(-rings,rings+1):
            if i<0:
                start = -rings-i
                end = rings
            else:
                start = -rings
                end = rings-i
            for j in range(start,end+1):
                lat.select((i, j))
        return CoherentBasis(lat)

    @staticmethod
    def create_hexagonal_grid_nearestN(origin, d, point, N):
        """
        Create a new coherent basis with basis states on a hexagonal grid.

        As basis states the nearest N grid points to the center are choosen.

        *Arguments*
            * *origin*
                A complex number defining the origin of the grid.

            * *d*
                A real number defining the lattice constant.

            * *point*
                A complex number used as reference point to which the nearest
                N lattice points are selected.

            * *N*
                An integer giving the number of basis states.
        """
        origin = mpmath.mpmathify(origin)
        point = mpmath.mpmathify(point)
        d = mpmath.mpf(d)
        lat = lattice.HexagonalLattice(d, origin, dtype=mpmath.mpf)
        lat.select_nearestN(point, N)
        return CoherentBasis(lat)


    @staticmethod
    def coherent_scalar_product(alpha, beta):
        """
        Calculate the scalar product of two coherent states.

        *Argument*
            * *alpha*, *beta*
                Complex numbers or vectors of complex numbers, defining the
                coherent states.
        """
        def sp(a,b):
            return mpmath.exp(-(abs(a)**2 + abs(b)**2)/2 + mpmath.conj(a)*b)

        if isinstance(alpha, (numpy.ndarray, list, tuple)):
            alpha = ndarray.Array(alpha)
        else:
            alpha = ndarray.Array([alpha])
        if isinstance(beta, (numpy.ndarray, list, tuple)):
            beta = ndarray.Array(beta)
        else:
            beta = ndarray.Array([beta])
        assert alpha.ndim == 1
        assert beta.ndim == 1
        r = ndarray.Array(numpy.empty((alpha.size, beta.size),
                          dtype=mpmath.mpc))
        for i,j in itertools.product(range(alpha.size), range(beta.size)):
            r[i,j] = sp(alpha[i], beta[j])
        if r.size == 1:
            return r[0,0]
        elif 1 in r.shape:
            return r.reshape(-1)
        else:
            return r

    def dual_basis(self, bra=True, ket=True):
        """
        Calculate the dual basis.
        """
        if self.lattice is None:
            states = self.states
        else:
            states = self.lattice.copy()
        return self.__class__(states, bra_is_dual=bra^self.bra_is_dual,
                              ket_is_dual= ket^self.ket_is_dual,
                              _trafo=self._trafo, _inv_trafo=self._inv_trafo)

    def to_dualbasis_coordinates(self, array):
        """
        Calculate the dual vector of a vector given in this basis.
        """
        return numpy.dot(array, self.trafo.T)

    def to_coherentbasis_coordinates(self, array):
        """
        Calculate the coordinates of a vector given in the dual basis.
        """
        return numpy.dot(array, self.inv_trafo.T)

    def coherent_state(self, alpha):
        """
        Calculate the coordinates of the given coherent state in this basis.
        """
        from ..statevector import StateVector
        b = self.coherent_scalar_product(self.states, alpha)
        if self.ket_is_dual:
            sv = StateVector(self.to_dualbasis_coordinates(b), basis=self)
        else:
            sv = StateVector(b, basis=self)
        return sv

    def create(self, pow=1):
        from ..operators import Operator as op
        # down-up version
        alpha = self.states.conj()**pow
        dbra, dket = self.bra_is_dual, self.ket_is_dual
        if not dbra and not dket:
            array = self.inv_trafo*alpha
        elif not dbra and dket:
            array = numpy.dot(self.inv_trafo*alpha, self.trafo)
        elif dbra and not dket:
            array = numpy.diag(alpha)
        elif dbra and dket:
            array = a_dag.reshape((-1,1))*self.trafo
        else:
            assert False
        return op(array, basis=self)

    def destroy(self, pow=1):
        from ..operators import Operator as op
        # up-down version
        alpha = self.states**pow
        dbra, dket = self.bra_is_dual, self.ket_is_dual
        if not dbra and not dket:
            array = alpha.reshape((-1,1))*self.inv_trafo
        elif not dbra and dket:
            array = numpy.diag(alpha)
        elif dbra and not dket:
            array = numpy.dot(self.trafo*alpha, self.inv_trafo)
        elif dbra and dket:
            array = self.trafo*alpha
        else:
            assert False
        return op(array, basis=self)

    def create_destroy(self, pow1=1, pow2=1):
        from ..operators import Operator as op
        # down-down versions
        alpha1 = self.states.conj()**pow1
        alpha2 = self.states**pow2
        adag_a = alpha1.reshape((-1,1))*self.trafo*alpha2
        dbra, dket = self.bra_is_dual, self.ket_is_dual
        if not dbra and not dket:
            array = numpy.dot(numpy.dot(self.inv_trafo, adag_a), self.inv_trafo)
        elif not dbra and dket:
            array = numpy.dot(self.inv_trafo, adag_a)
        elif dbra and not dket:
            array = numpy.dot(adag_a, self.inv_trafo)
        elif dbra and dket:
            array = adag_a
        else:
            assert False
        return op(array, basis=self)

    def basis_change_func(self, basis):
        # TODO: Improve this function.
        from . import fock_basis
        if isinstance(basis, CoherentBasis):
            def f1(psi):
                if self.ket_is_dual:
                    return numpy.dot(self.inv_trafo, psi)
                return psi
            def f2(psi):
                if not basis.ket_is_dual:
                    return numpy.dot(basis.inv_trafo, psi)
                return psi
            new_states = basis.states
            old_states = self.states
            T = CoherentBasis.coherent_scalar_product(new_states, old_states)
            return lambda psi:f2(numpy.dot(T, f1(psi)))
        elif isinstance(basis, fock_basis.FockBasis):
            states = self.states
            A = numpy.empty((basis.N1-basis.N0, len(states)), dtype=mpmath.mpc)
            for i, alpha in enumerate(states):
                A[:,i] = basis.coherent_state(alpha, dtype=mpmath.mpc)
            def f1(psi):
                if self.ket_is_dual:
                    return numpy.dot(self.inv_trafo, psi)
                else:
                    psi
            return lambda psi:numpy.dot(A, f1(psi))
        else:
            raise NotImplementedError()


