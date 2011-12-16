import numpy

class Lattice:
    origin = None
    basis_vectors = None
    selected = None
    def __init__(self, origin, basis_vectors):
        self.origin = origin
        self.basis_vectors = basis_vectors
        self.selected = []

    def select(self, indices):
        if indices not in self.selected:
            self.selected.append(indices)

    def clear_selection(self):
        self.selected = []

    def state(self, index):
        return numpy.dot(index, self.basis_vectors) + self.origin

    def states(self, indices=None):
        if indices is None:
            indices = self.selected
        return tuple(self.state(i) for i in indices)

    def neighbors(self, max_order):
        assert max_order >= 1
        def f(indices, order):
            S = set()
            for i in indices[-1]:
                S |= self._neighbors(i)
            for i in indices:
                S -= i
            indices.append(S)
            if order == 1:
                return indices
            else:
                return f(indices, order-1)
        indices = set(self.selected)
        n = f([indices], max_order)
        return (tuple(self.selected),) + tuple(map(tuple, n[1:]))


    def copy(self):
        raise NotImplementedError()


class HexagonalLattice(Lattice):
    d = None
    dtype = None
    def __init__(self, d, origin=0, dtype=float):
        self.d = d
        self.dtype = dtype
        v0 = (dtype("1") + 1j*dtype("0"))*d
        v1 = (dtype("1")/2 + 1j*dtype("3")**(dtype("1")/2)/2)*d
        Lattice.__init__(self, origin, numpy.array([v0,v1]))

    def _neighbors(self, index):
        x = index[0]
        y = index[1]
        n1 = (x+1,y)
        n2 = (x,y+1)
        n3 = (x-1,y+1)
        n4 = (x-1,y)
        n5 = (x,y-1)
        n6 = (x+1,y-1)
        return set([n1,n2,n3,n4,n5,n6])

    def copy(self):
        lat = self.__class__(self.d, self.origin, self.dtype)

class SquareLattice(Lattice):
    d = None
    dtype = None
    def __init__(self, d, origin=0, dtpye=complex):
        self.d = d
        self.dtype = dtype
        v0 = (dtype("1") + 1j*dtype("0"))*d
        v1 = (dtype("0") + 1j*dtype("1"))*d
        Lattice.__init__(self, origin, numpy.array([v0,v1]))

    def _neighbors(self, index):
        x = index[0]
        y = index[1]
        n1 = (x+1,y)
        n2 = (x+1,y+1)
        n3 = (x,y+1)
        n4 = (x-1,y+1)
        n5 = (x-1,y)
        n6 = (x-1,y-1)
        n7 = (x,y-1)
        n8 = (x+1,y-1)
        return set([n1,n2,n3,n4,n5,n6,n7,n8])

    def copy(self):
        lat = self.__class__(self.d, self.origin, self.dtype)
