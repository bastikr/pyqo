import numpy

from . import datatypes
from . import utils


class Array(numpy.ndarray):
    def __new__(cls, data, basis=None, **kwargs):
        array = numpy.array(data, **kwargs)
        """
        if array.dtype in (numpy.float16, numpy.float32,
                                    numpy.float64, numpy.float128):
            kwargs["dtype"] = complex
            array = numpy.array(data, **kwargs)
        array = numpy.asarray(array).view(cls)
        """
        array = array.view(cls)
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

    def __mul__(self, other):
        result = (self.view(numpy.ndarray) * other)
        if isinstance(result, numpy.ndarray):
            result = result.view(self.__class__)
            result.basis = self.basis
        return result

    def __rmul__(self, other):
        result = (other * self.view(numpy.ndarray))
        if isinstance(result, numpy.ndarray):
            result = result.view(self.__class__)
            result.basis = self.basis
        result.basis = self.basis
        return result

    def __div__(self, other):
        result = (self.view(numpy.ndarray) / other)
        if isinstance(result, numpy.ndarray):
            result = result.view(self.__class__)
            result.basis = self.basis
        return result

    def __rdiv__(self, other):
        result = (other / self.view(numpy.ndarray))
        if isinstance(result, numpy.ndarray):
            result = result.view(self.__class__)
            result.basis = self.basis
        result.basis = self.basis
        return result


    def __add__(self, other):
        result = (self.view(numpy.ndarray) + other)
        if isinstance(result, numpy.ndarray):
            result = result.view(self.__class__)
            result.basis = self.basis
        result.basis = self.basis
        return result

    def __radd__(self, other):
        result = (other + self.view(numpy.ndarray))
        if isinstance(result, numpy.ndarray):
            result = result.view(self.__class__)
            result.basis = self.basis
        result.basis = self.basis
        return result

    def __sub__(self, other):
        result = (self.view(numpy.ndarray) - other)
        if isinstance(result, numpy.ndarray):
            result = result.view(self.__class__)
            result.basis = self.basis
        result.basis = self.basis
        return result

    def __rsub__(self, other):
        result = (other - self.view(numpy.ndarray))
        if isinstance(result, numpy.ndarray):
            result = result.view(self.__class__)
            result.basis = self.basis
        result.basis = self.basis
        return result

