import mpmath
import numpy

TYPE = mpmath.mpc

METHODS = {
    "conjugate": numpy.vectorize(mpmath.conj),
    "conj": numpy.vectorize(mpmath.conj),
}

PROPERTIES = {
    "imag": numpy.vectorize(mpmath.im, otypes=[mpmath.mpf]),
    "real": numpy.vectorize(mpmath.re, otypes=[mpmath.mpf]),
}

