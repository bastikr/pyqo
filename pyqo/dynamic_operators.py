from . import operators

class DynamicOperator(operators.BaseOperator):
    def __init__(self, func, basis=None):
        self.basis = basis
        self.__call__ = func
