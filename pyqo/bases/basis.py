import itertools
import numpy

class Basis:
    rank = None

    def __init__(self, rank):
        assert rank > 0
        self.rank = rank

    def dual(self, psi):
        return NotImplementedError()

    def inverse_dual(self, psi):
        return NotImplementedError()

    def ptarce(self, indices):
        return NotImplementedError()

    def scalar_product(self, psi1, psi2):
        assert psi1.shape == psi2.shape
        return numpy.tensordot(self.dual(psi1).conj(), psi2, psi1.ndim)

    def norm(self, psi):
        return numpy.sqrt(numpy.abs(self.scalar_product(psi, psi)))

    def ptrace(self, indices):
        if isinstance(indices, int):
            indices = (indices,)
        new_rank = self.rank-len(indices)
        if new_rank == 0:
            return None
        else:
            return self.__class__(new_rank)

    def __xor__(self, other):
        if not isinstance(other, Basis):
            return NotImplemented
        else:
            return CompositeBasis((self, other))

    def __eq__(self, other):
        if self.__class__==other.__class__ and self.rank==other.rank:
            return True
        else:
            return False

    def __ne__(self, other):
        return not self==other


class CompositeBasis(Basis):
    bases = None
    def __init__(self, bases):
        rank = 0
        for b in bases:
            rank += b.rank
        Basis.__init__(self, rank=rank)
        self.bases = bases

    def apply(self, psi, func):
        d = psi.copy()
        rank_pos = 0
        dims = tuple(map(range, psi.shape))
        for b in self.bases:
            for front in itertools.product(*dims[:rank_pos]):
                for back in itertools.product(*dims[rank_pos+b.rank:]):
                    slice = front + (Ellipsis,)*b.rank + back
                    d[slice] = func(b, d[slice])
            rank_pos += b.rank
        return d

    def dual(self, psi):
        return self.apply(psi, lambda basis, slice: basis.dual(slice))

    def inverse_dual(self, psi):
        return self.apply(psi, lambda basis, slice: basis.inverse_dual(slice))

    def ptrace(self, indices):
        if isinstance(indices, int):
            indices = (indices,)
        n = numpy.cumsum((0,) + tuple(b.rank for b in self.bases))
        new_bases = []
        for i, b in enumerate(self.bases):
            t = tuple(j+n[i] in indices for j in range(b.rank))
            if all(t):
                pass
            elif not any(t):
                new_bases.append(b)
            else:
                new_bases.append(b.ptrace(numpy.arange(b.rank)[numpy.array(t)]))
        if len(new_bases) == 0:
            return None
        elif len(new_bases) == 1:
            return new_bases[0]
        else:
            return self.__class__(new_bases)

    def __eq__(self, other):
        if self.__class__ == other.__class__ and self.rank == other.rank:
            l = len(self.bases)
            return all(self.bases[i]==other.bases[i] for i in range(l))
        else:
            return False


class ONBasis(Basis):
    def dual(self, psi):
        return psi

    def inverse_dual(self, psi):
        return psi

    def __xor__(self, other):
        if isinstance(other, ONBasis):
            return ONBasis(self.rank + other.rank)
        else:
            return Basis.__xor__(self, other)


def compose_bases(basis1, rank1, basis2, rank2):
    if basis1 is None and basis2 is None:
        basis = None
    else:
        if basis1 is None:
            basis1 = ONBasis(rank1)
        if basis2 is None:
            basis2 = ONBasis(rank2)
        basis = basis1 ^ basis2
    return basis

