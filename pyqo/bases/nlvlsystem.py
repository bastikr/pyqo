from . import basis
import numpy

class NLevelSystem(basis.ONBasis):
    rank = 1
    energies = None
    N = None
    dtype = None

    def __init__(self, energies, dtype=complex):
        self.energies = energies
        self.N = len(energies)
        self.dtype = dtype

    def zero(self):
        """
        Create a statevector with only zero entries.
        """
        from ..statevector import StateVector
        X = numpy.zeros(self.N, dtype=self.dtype)
        if X.dtype is numpy.dtype("object"):
            X[:] = self.dtype(0)
        return StateVector(X, basis=self, dtype=self.dtype)

    def basis_vector(self, i):
        """
        Create a statevector where only the i-th level is occupied.
        """
        assert i < self.N
        X = self.zero()
        X[i] = self.dtype(1)
        return X

    def identity(self):
        from ..operators import Operator
        if self.dtype is complex:
            X = numpy.eye(self.N)
        else:
            X = numpy.array([self.dtype(1)]*self.N)
            X = numpy.diag(X)
        return Operator(X, basis=self)

    def H0(self):
        from ..operators import Operator
        return Operator(numpy.diag(self.energies), basis=self)

    def couple(self, i, j, g):
        from ..operators import Operator
        state_i = self.basis_vector(i)
        state_j = self.basis_vector(j)
        op = g*Operator(state_i^state_j.conj(), basis=self)
        return op + op.H

    def project_to_from(self, i, j):
        from ..operators import Operator
        state_i = self.basis_vector(i)
        state_j = self.basis_vector(j)
        return Operator(state_i^state_j.conj(), basis=self)


