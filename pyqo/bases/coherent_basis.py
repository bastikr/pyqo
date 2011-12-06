"""
Provides functionality to calculate in coherent bases.
"""
import itertools
import numpy

try:
    import mpmath
except:
    print("mpmath not found - Calculating coherent bases not posible")

from . import basis
from .. import ndarray

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
    _trafo = None
    _inv_trafo = None

    def __init__(self, states):
        self.states = ndarray.Array(states)

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
    def create_hexagonal_grid(center, d, rings):
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
        basis = []
        for i in range(-rings,rings+1):
            if i<0:
                start = -rings-i
                end = rings
            else:
                start = -rings
                end = rings-i
            for j in range(start,end+1):
                basis.append(center+d*i+d/2*j+1j*mpmath.sqrt(3)/2*d*j)
        return CoherentBasis(basis)

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
        #if psi.cols == 1:
        #    return mp.cholesky_solve(self.trafo, psi)
        #r = mp.matrix(psi.rows,psi.cols)
        #for i in range(psi.cols):
        #    r[:,i] = self.dual_reverse(psi[:,i])
        #return r

    def coordinates(self, alpha):
        """
        Calculate the coordinates of the given coherent state in this basis.
        """
        b = self.coherent_scalar_product(self.states, alpha)
        return self.inverse_dual(b)

