import unittest

from . import operators as op


class OperatorTestCase(unittest.TestCase):
    def test_creation(self):
        A = op.Operator(((1,2), (3,4)))
        self.assertEqual(len(A.shape),2)
        self.assertRaises(ValueError, op.Operator, ((1,2,3),(4,3,2)))

    def test_tensor(self):
        op1 = op.Operator(((1,2), (3,4)))
        op2 = op.Operator(((1,4,0), (3,4,1), (2,1,3)))
        t = op1 ^ op2
        self.assertEqual(t.shape, (2,3,2,3))

    def test_ptrace(self):
        x,y,z = op.sigmax, op.sigmay, op.sigmaz
        I = op.identity(x)
        self.assertEqual((I^y).ptrace(0).tolist(), (2*y).tolist())

    def test_mul(self):
        x,y,z = op.sigmax, op.sigmay, op.sigmaz
        self.assertEqual((x*y).tolist(),
                         (1j*z).tolist())
        self.assertEqual((y * op.identity(2)).tolist(),
                         y.tolist())
        self.assertEqual(((x^z)*(y^y)).tolist(),
                         ((x*y)^(z*y)).tolist())

