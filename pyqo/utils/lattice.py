import itertools
import numpy

class Lattice:
    origin = None
    basis_vectors = None
    selected = None
    def __init__(self, origin, basis_vectors, selected=None):
        self.origin = origin
        self.basis_vectors = basis_vectors
        self.selected = [] if selected is None else list(selected)

    def __repr__(self):
        clsname = "%s.%s" % (self.__module__, self.__class__.__name__)
        return "%s(%s, %s, %s)" % (clsname, repr(self.origin),
                    repr(self.basis_vectors), repr(self.selected))

    @staticmethod
    def scalar_product(a, b):
        if isinstance(a, numpy.ndarray):
            return numpy.dot(a,b)
        else:
            return a.real*b.real + a.imag*b.imag

    def select(self, indices):
        if indices not in self.selected:
            self.selected.append(indices)

    def select_sequence(self, indices):
        for ind in indices:
            self.select(ind)

    def select_nearestN(self, point, N):
        # Find nearest grid point:
        #   (1) find dual vectors for basis
        T = []
        for xi in self.basis_vectors:
            T.append([])
            for xj in self.basis_vectors:
                T[-1].append(self.scalar_product(xi,xj))
        T = numpy.array(T)
        T_inv = numpy.linalg.inv(T)
        duals = []
        for i in range(len(self.basis_vectors)):
            duals.append(sum(T_inv[i,j]*xj\
                         for j, xj in enumerate(self.basis_vectors)))
        #   (2) multiplicate them with the given point
        coordinates = tuple(int(self.scalar_product(x, point)) for x in duals)
        # For a sufficient amount of neighbor points calculate the distance
        # to the given point and choose only the N nearest.
        lat = self.copy()
        lat.clear_selection()
        lat.select(coordinates)
        neighbors_indices = lat.neighbors(int(N/2+1))
        neighbors_indices = tuple(itertools.chain(*neighbors_indices))
        neighbors = self.states(neighbors_indices)
        d = (self.scalar_product(x-point, x-point)\
                                        for x in neighbors)
        d = numpy.array(tuple(d))
        key = numpy.argsort(d)
        selection = []
        for k in key[:N]:
            selection.append(neighbors_indices[k])
        self.select_sequence(selection)

    def clear_selection(self):
        self.selected = []

    def state(self, index):
        return sum(tuple(index[i]*self.basis_vectors[i] for i in range(len(index)))) + self.origin
        #return numpy.dot(index, self.basis_vectors) + self.origin

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
        selected = None if self.selected is None else tuple(self.selected)
        return self.__class__(self.origin, self.basis_vectors, selected)


class HexagonalLattice(Lattice):
    d = None
    dtype = None
    def __init__(self, d, origin=0, selection=None, dtype=float):
        self.d = d
        self.dtype = dtype
        v0 = (dtype("1") + 1j*dtype("0"))*d
        v1 = (dtype("1")/2 + 1j*dtype("3")**(dtype("1")/2)/2)*d
        Lattice.__init__(self, origin, numpy.array([v0,v1]), selection)

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
        selected = None if self.selected is None else tuple(self.selected)
        return self.__class__(self.d, self.origin, selected, self.dtype)

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
