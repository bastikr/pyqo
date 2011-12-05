import unittest
import numpy
import mpmath

from . import CoherentBasis

class TestCoherentBasis(unittest.TestCase):
    def test_create_hex_grid(self):
        b = CoherentBasis.create_hexagonal_grid(0,1.2,1)
        self.assertEqual(len(b.states), 7)
        trafo = b.trafo
        inv_trafo = b.inv_trafo
        self.assertIsInstance(trafo, numpy.ndarray)
        self.assertIsInstance(inv_trafo, numpy.ndarray)
        I1 = numpy.abs(numpy.dot(trafo, inv_trafo))
        I2 = numpy.array(mpmath.eye(7).tolist())
        self.assertTrue(((I1-I2)<1e-14).all())

