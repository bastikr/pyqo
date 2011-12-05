import numpy

from . import datatypes
from . import utils

DEFAULT_DTYPE = [complex]

class Array(numpy.ndarray):
    def __new__(cls, data, basis=None, **kwargs):
        #if "dtype" not in kwargs:
        #    print("no dtype")
        #    kwargs["dtype"] = DEFAULT_DTYPE[0]
        #print("__new__ dtype:", kwargs["dtype"])
        array = numpy.array(data, **kwargs)
        if array.dtype in (numpy.float16, numpy.float32,
                                    numpy.float64, numpy.float128):
            kwargs["dtype"] = complex
            array = numpy.array(data, **kwargs)
        array = numpy.asarray(array).view(cls)
        array.basis = basis
        cls._check(array)
        return array

    def __array_finalize__(self, obj):
        if obj is not None and obj.size != 0:
            dtype = type(obj.flat[0])
        elif self.size != 0:
            dtype = type(self.flat[0])
        else:
            dtype = None
        for d in datatypes.types:
            if issubclass(dtype, d):
                methods, properties = datatypes.types[d]
                utils.add_methods(self, methods)
                utils.add_properties(self, properties)
                break
        if obj is None:
            return
        self.basis = getattr(obj, "basis", None)

    @staticmethod
    def _check(array):
        pass

    @property
    def imag(self):
        if hasattr(self, "_imag"):
            return self._imag()
        else:
            return numpy.ndarray.imag.__get__(self)

    @property
    def real(self):
        if hasattr(self, "_real"):
            return self._real()
        else:
            return numpy.ndarray.real.__get__(self)
