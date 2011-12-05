import unittest
import numpy as np

from . import statevector as sv
from . import operators
from . import bases

class TestStateVector(unittest.TestCase):
    def test_creation(self):
        psi = sv.StateVector([[1,2,3],[1,0,4]])
        self.assertIsInstance(psi, sv.StateVector)
        self.assertIs(psi.basis, None)

    def test_dual(self):
        psi = sv.StateVector([[1,2,3],[1,0,4]])
        dual = psi.dual()
        inv_dual = psi.inverse_dual()
        self.assertIs(psi, dual)

    def test_norm(self):
        psi = sv.StateVector([1j,0,0])
        self.assertTrue(psi.norm()-1<1e-15)

    def test_normalize(self):
        b = bases.ONBasis(1)
        psi = sv.StateVector([0.1j,0,0], basis=b)
        psi2 = psi.normalize()
        self.assertTrue(abs(psi2[0] - 1j)<1e-15)
        self.assertIs(psi2.basis, b)

    def test_DO(self):
        b = bases.Basis(1)
        psi = sv.StateVector([1,0,0], basis=b)
        do = psi.DO
        self.assertIs(do.basis, b)
        self.assertIsInstance(do, operators.DensityOperator)

    def test_ptrace(self):
        b = bases.ONBasis(2)
        psi = sv.StateVector([[1j,1], [2,0]], basis=b).normalize()
        rho = psi.ptrace(1)
        op = psi.DO.ptrace(1)
        self.assertTrue((rho-op).sum() < 1e-14)
        self.assertIsInstance(rho.basis, bases.ONBasis)
        self.assertIsInstance(op.basis, bases.ONBasis)
        self.assertEqual(rho.basis.rank, 1)
        self.assertEqual(op.basis.rank, 1)
        self.assertIsInstance(rho, operators.DensityOperator)

    def test_tensor(self):
        b = bases.ONBasis(1)
        psi1 = sv.StateVector([1,2,1], basis=b).normalize()
        psi2 = sv.StateVector([1j,0,-0.1,4], basis=b).normalize()
        psi = psi1 ^ psi2
        self.assertIsInstance(psi, sv.StateVector)
        self.assertEqual(psi.shape, (3,4))
        self.assertIsInstance(psi.basis, bases.ONBasis)
        self.assertEqual(psi.basis.rank, 2)


