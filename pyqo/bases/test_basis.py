import unittest
import numpy

from . import basis

class BasisTestCase(unittest.TestCase):
    def test_creation(self):
        b = basis.Basis(3)
        self.assertEqual(3, b.rank)
        self.assertRaises(AssertionError, basis.Basis, -1)

    def test_ptrace(self):
        b = basis.Basis(3)
        b_sub1 = b.ptrace(0)
        b_sub2 = b.ptrace((1,2))
        b_sub3 = b.ptrace((0,1,2))
        self.assertEqual(b_sub1.rank, 2)
        self.assertEqual(b_sub2.rank, 1)
        self.assertIs(b_sub3, None)

    def test_equality(self):
        self.assertEqual(basis.Basis(1), basis.Basis(1))
        self.assertNotEqual(basis.Basis(1), basis.Basis(2))


class ONBasisTestCase(unittest.TestCase):
    def test_dual(self):
        b = basis.ONBasis(1)
        psi = [1,2]
        self.assertIs(b.dual(psi), psi)
        self.assertIs(b.inverse_dual(psi), psi)


class CompositeBasisTestCase(unittest.TestCase):
    def test_creation(self):
        b1 = basis.ONBasis(2)
        b2 = basis.Basis(1)
        b = basis.CompositeBasis((b1,b2))
        self.assertEqual(b.rank, 3)

    def test_combine(self):
        b1 = basis.ONBasis(1)
        b2 = basis.ONBasis(2)
        b = (b1 ^ b2)
        self.assertIsInstance(b, basis.CompositeBasis)
        self.assertEqual(b.rank, 3)

    def test_dual(self):
        b1 = basis.ONBasis(1)
        b2 = basis.ONBasis(2)
        b = basis.CompositeBasis((b1, b2))
        psi = numpy.arange(30).reshape((5,2,3))
        psi_dual = b.dual(psi)
        psi_idual = b.inverse_dual(psi)
        self.assertTrue((psi_dual==psi).all())
        self.assertTrue((psi_dual==psi).any())

    def test_ptrace(self):
        b1 = basis.ONBasis(3)
        b2 = basis.Basis(1)
        b = b1 ^ b2
        b_sub1 = b.ptrace(3)
        b_sub2 = b.ptrace((0,1,2))
        b_sub3 = b.ptrace(0)
        b_sub4 = b.ptrace((0,1,2,3))
        self.assertIs(b_sub1, b1)
        self.assertIs(b_sub2, b2)
        self.assertIsInstance(b_sub3.bases[0], basis.ONBasis)
        self.assertIsInstance(b_sub3.bases[1], basis.Basis)
        self.assertEqual(b_sub3.bases[0].rank, 2)
        self.assertIs(b_sub4, None)

    def test_compose(self):
        b1 = basis.ONBasis(1)
        b2 = basis.Basis(2)
        b = basis.compose_bases(None, 1, None, 2)
        self.assertIs(b, None)
        b = basis.compose_bases(b1, 1, None, 2)
        self.assertIsInstance(b, basis.CompositeBasis)
        self.assertEqual(b.rank, 3)

    def test_equality(self):
        b1 = basis.ONBasis(1)
        b2 = basis.Basis(2)
        self.assertEqual(b1^b2, b1^b2)
        self.assertNotEqual(b1^b2, b1)


