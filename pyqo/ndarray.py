import numpy
import types

from . import datatypes


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
        if obj is None:
            return
        if obj.size != 0:
            dtype = type(obj.flat[0])
        elif self.size != 0:
            dtype = type(self.flat[0])
        else:
            dtype = None
        if dtype is not None:
            for d in datatypes.types:
                if issubclass(dtype, d):
                    methods, properties = datatypes.types[d]
                    add_methods(self, methods)
                    add_properties(self, properties)
                    break
        self.basis = getattr(obj, "basis", None)

    def __repr__(self):
        clsname = "%s.%s" % (self.__module__, self.__class__.__name__)
        def f(a):
            if isinstance(a, list):
                return "[%s]" % ",".join(map(f, a))
            else:
                return repr(a)
        return "%s(%s, basis=%s)" % (clsname, f(self.tolist()),
                                     repr(self.basis))

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


def add_methods(array, methods):
    for name, func in methods.items():
        f = types.MethodType(func, array)
        setattr(array, name, f)

def add_properties(array, properties):
    for name, func in properties.items():
        f = types.MethodType(func, array)
        setattr(array, "_"+name, f)

