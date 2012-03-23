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
from . import number_basis
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
    """
    rank = 1
    states = None
    lattice = None
    _trafo = None
    _inv_trafo = None

    def __init__(self, states):
        if isinstance(states, lattice.Lattice):
            self.states = ndarray.Array(states.states())
            self.lattice = states
        else:
            self.states = ndarray.Array(states)

    def __repr__(self):
        clsname = "%s.%s" % (self.__module__, self.__class__.__name__)
        if self.lattice is None:
            return "%s(%s)" % (clsname, repr(self.states))
        else:
            return "%s(%s)" % (clsname, repr(self.lattice))

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
        """
        is_matrix = False
        if isinstance(alpha, mpmath.matrix):
            rows = max((alpha.rows,alpha.cols))
            is_matrix = True
        else:
            alpha = mpmath.matrix([alpha])
            rows = 1
        if isinstance(beta, mpmath.matrix):
            cols = max((beta.rows,beta.cols))
            is_matrix = True
        else:
            beta = mpmath.matrix([beta])
            cols = 1
        r = mpmath.matrix(rows,cols)
        for i,j in itertools.product(range(rows),range(cols)):
            alpha_i = alpha[i]
            beta_j = beta[j]
            r[i,j] = mpmath.exp(
                        -mpmath.mpf(1)/2*(abs(alpha_i)**2 + abs(beta_j)**2)\
                        + mpmath.conj(alpha_i)*beta_j\
                        )
        if is_matrix:
            return numpy.array(r.tolist())
        else:
            return r[0,0]
        """

    def dual(self, psi):
        """
        Calculate the dual vector of a vector given in this basis.
        """
        return numpy.dot(psi, self.trafo.T)

    def inverse_dual(self, psi):
        """
        Calculate the coordinates of a vector given in the dual basis.
        """
        return numpy.dot(psi, self.inv_trafo.T)

    def coherent_state(self, alpha, raise_index=False):
        """
        Calculate the coordinates of the given coherent state in this basis.
        """
        from .. import statevector
        b = self.coherent_scalar_product(self.states, alpha)
        if raise_index:
            sv = statevector.StateVector(self.inverse_dual(b), basis=self)
        else:
            sv = statevector.StateVector(b, basis=self)

    def create(self, pow=1, raise_left=False, raise_right=False):
        from ..operators import Operator as op
        # down-up version
        alpha = self.states.conj()**pow
        if raise_left and raise_right:
            return op(self.inv_trafo*alpha, basis=self)
        elif raise_left and not raise_right:
            return op(numpy.dot(self.inv_trafo*alpha, self.trafo), basis=self)
        elif not raise_left and raise_right:
            return op(numpy.diag(alpha), basis=self)
        elif not raise_left and not raise_right:
            return op(a_dag.reshape((-1,1))*self.trafo, basis=self)

    def destroy(self, pow=1, raise_left=False, raise_right=False):
        from ..operators import Operator as op
        # up-down version
        alpha = self.states**pow
        if raise_left and raise_right:
            return op(alpha.reshape((-1,1))*self.inv_trafo, basis=self)
        elif raise_left and not raise_right:
            return op(numpy.diag(alpha), basis=self)
        elif not raise_left and raise_right:
            return op(numpy.dot(self.trafo*alpha, self.inv_trafo), basis=self)
        elif not raise_left and not raise_right:
            return op(self.trafo*alpha, basis=self)

    def create_destroy(self, pow1=1, pow2=1, raise_left=False, raise_right=False):
        from ..operators import Operator as op
        # down-down versions
        alpha1 = self.states.conj()**pow1
        alpha2 = self.states**pow2
        adag_a = alpha1.reshape((-1,1))*self.trafo*alpha2
        if raise_left and raise_right:
            return op(numpy.dot(numpy.dot(self.inv_trafo, adag_a), self.inv_trafo),
                      basis=self)
        elif raise_left and not raise_right:
            return op(numpy.dot(self.inv_trafo, adag_a), basis=self)
        elif not raise_left and raise_right:
            return op(numpy.dot(adag_a, self.inv_trafo), basis=self)
        elif not raise_left and not raise_right:
            return op(adag_a, basis=self)

    def transform_func(self, basis):
        if isinstance(basis, CoherentBasis):
            new_states = self.states
            old_states = basis.states
            T = CoherentBasis.coherent_scalar_product(new_states, old_states)
            inv_trafo = self.inv_trafo
            return lambda psi:numpy.dot(inv_trafo, numpy.dot(T, psi))
        elif isinstance(basis, number_basis.NumberBasis):
            states = self.states
            A = numpy.empty((basis.N1-basis.N0, len(states)), dtype=mpmath.mpc)
            for i, alpha in enumerate(states):
                A[:,i] = basis.coherent_state(alpha, dtype=mpmath.mpc)
            return lambda psi:numpy.dot(A, psi)
        else:
            raise NotImplementedError()


