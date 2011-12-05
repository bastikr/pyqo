from . import operators

class DynamicOperator(operators.BaseOperator):
    def __init__(self, func, t_min, t_max, basis=None):
        self.basis = basis
        self.__call__ = func
        self.t_min = t_min
        self.t_max = t_max
