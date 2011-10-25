import numpy

class StateVector(numpy.ndarray):
    r"""
    A class representing a state vector in a specific basis.

    *Usage*
        >>> sv = StateVector((1, 3, 7, 2), norm=True)
        >>> sv = StateVector(numpy.arange(12).reshape(3,4))
        >>> print(sv)
        StateVector(3 x 4)
        >>> print(repr(sv))
        StateVector([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])

    *Arguments*
        * *data*
            Anything that can be used to create a numpy array, e.g. a nested
            tuple or another numpy array.

        * *norm* (optional)
            If set True the StateVector will be automatically normalized.
            (Default is False)

        * Any other argument that a numpy array takes. E.g. ``copy=False`` can
          be used so that the StateVector shares the data storage with the
          given numpy array.

    Most useful is maybe the tensor product which lets you easily calculate
    state vectors for combined systems::

        >>> sv1 = StateVector((1,2,3))
        >>> sv2 = StateVector((3,4,0), norm=True)
        >>> sv = sv1 ^ sv2
        >>> print(sv)
        StateVector(3 x 3)
        [[ 0.6,  0.8,  0. ],
         [ 1.2,  1.6,  0. ],
         [ 1.8,  2.4,  0. ]]

    The tensor product is abbreviated by the "^" operator. But be aware that
    this operator follows the built-in operator precedence - that means "+",
    "*" etc. have **higher** precedence!
    """
    def __new__(cls, data, norm=False, **kwargs):
        array = numpy.array(data, dtype=complex, **kwargs)
        if norm:
            array = normalize(array)
        return numpy.asarray(array).view(cls)

    def __str__(self):
        clsname = self.__class__.__name__
        dims = " x ".join(map(str, self.shape))
        array = numpy.ndarray.__str__(self)
        return "%s(%s)\n%s" % (clsname, dims, array)

    def norm(self):
        r"""
        Calculate the norm of the StateVector.

        *Usage*
            >>> sv = StateVector((1,2,3,4,5), norm=True)
            >>> print(sv.norm())
            1.0
        """
        return numpy.abs(numpy.tensordot(self, self.conj(), len(self.shape)))

    def normalize(self):
        r"""
        Return a normalized StateVector.

        *Usage*
            >>> sv = StateVector((1,2,1,3,1))
            >>> print(sv.norm())
            4.0
            >>> nsv = sv.normalize()
            >>> print(nsv.norm())
            1.0
        """
        return self/self.norm()

    @property
    def DO(self):
        """
        Calculate the density operator |Psi><Psi|.
        """
        # statevector.py and operators.py have circular import
        from . import operators
        return operators.Operator(self ^ self.conj())

    def ptrace(self, indices):
        r"""
        Calculate the reduced density operator tr_ind{|Psi><Psi|}.

        *Usage*
            >>> sv1 = StateVector((0,1,2,1,0), norm=True)
            >>> sv2 = StateVector((1,0,1), norm=True)
            >>> sv = sv1^sv2
            >>> sqtensor = sv.ptrace(1)

        *Arguments*
            * *indices*
                An integer or a collection of integers specifying which
                subsystems should be traced out.

        The result of this method is the same as generating the density
        operator corresponding to the state vector and tracing
        over the subsystems specified by the given incices.
        """
        if isinstance(indices, int):
            a = (indices,)
        else:
            a = _sorted_list(indices, True)
        # statevector.py and operators.py have circular import
        from . import operators
        return operators.Operator(numpy.tensordot(self, self.conj(), (a,a)))

    def tensor(self, other):
        r"""
        Return the tensor product between this and the given StateVector.

        *Usage*
            >>> sv = StateVector((0,1,2), norm=True)
            >>> print(repr(sv.tensor(StateVector((3,4), norm=True))))
            StateVector([[ 0.        ,  0.        ],
                   [ 0.26832816,  0.35777088],
                   [ 0.53665631,  0.71554175]])
            >>> print(sv.tensor((3,4))) # Not normalized!
            StateVector([[ 0.        ,  0.        ],
                   [ 1.34164079,  1.78885438],
                   [ 2.68328157,  3.57770876]])

        *Arguments*
            * *array*
                Some kind of array (E.g. StateVector, numpy.array, list, ...).

        As abbreviation ``sv1^sv2`` can be written instead of
        ``sv1.tensor(sv2)``. But be aware that the operator precedence of
        ``^`` follows the python rules - that means ``sv1 ^ sv2 + sv3`` is
        the same as ``sv1 ^ (sv2 + sv3)``.
        """
        assert isinstance(other, StateVector)
        return self.__class__(numpy.multiply.outer(self, other))

    __xor__ = tensor


# Helper functions for StateVector

def _dim2str(dimensions):
    """
    Return the corresponding dimension string for the given nested tuple.
    """
    dims = []
    for d in dimensions:
        dims.append("(%s,%s)" % d)
    return " x ".join(dims)

def _conjugate_indices(indices, ndim):
    """
    Return all numbers from 0 to ndim which are not in indices.
    """
    if isinstance(indices, int):
        indices = (indices,)
    return set(range(ndim)).difference(indices)

def _sorted_list(iterable, reverse=False):
    """
    Transform an iterable to a sorted list.
    """
    a = list(iterable)
    a.sort()
    if reverse:
        a.reverse()
    return a


# Functions creating commonly used state vectors.

def zero(x):
    """
    Creates a StateVector of a shape defined by x where all entries are 0.

    *Usage*
        >>> sv = zero(3)
        >>> print(sv)

        >>> sv = zero((2,3))
        >>> print(sv)

        >>> sv2 = zero(sv)
        >>> print(sv2)

    *Arguments*
        * *x*
            A tuple defining the shape of the StateVector or an
            StateVector which will be used to determine the shape.
    """
    if isinstance(x, StateVector):
        return StateVector(numpy.zeros_like(x))
    else:
        return StateVector(numpy.zeros(x))

def basis(x, index):
    """
    Creates a StateVector of a shape defined by x where only one entry is 1.

    *Usage*
        >>> sv = basis(4,0)
        >>> print(sv)

        >>> sv = basis(sv, 2)
        print(sv)

    *Arguments*
        * *x*
            A tuple defining the shape of the StateVector or an
            StateVector which will be used to determine the shape.

        * *index*
            A tuple or an integer specifying which entry is zero.
    """
    e = zero(x)
    e[index] = 1
    return e

def gaussian(x0, k0, sigma, fin):
    r"""
    Generate a StateVector with a normal distribution.

    *Usage*
        >>> sv = gaussian(x0=0.3, k0=4, sigma=0.6, fin=7)
        >>> print sv
        StateVector(128)

    *Arguments*
        * *x0*
            Center in the real space.

        * *k0*
            Center in the k-space.

        * *sigma*
            Width in the real space :math:`(\sigma = \sqrt{Var(x)})`.

        * *fin*
            :math:`2^{fin}` determines the amount of sample points.

    *Returns*
        * *sv*
            A :class:`pyqo.statevector.StateVector` representing this
            gaussian wave packet in the k-space.

    The generated StateVector is normalized and given in the k-space. It
    is the Fourier transformed of the following expression:

        .. math::

            \Psi(x) = \frac {1} {\sqrt[4]{2 \pi}} *
                            e^{-\frac {x^2} {4*{\Delta x}^2}}
    """
    N = 2**fin
    L = 2*numpy.pi
    if 6.*sigma > L:
        print("Warning: Sigma might be too big.")
    dx = L/float(N)
    if sigma < dx:
        print("Warning: Sigma might be too small.")
    fft = numpy.fft.fft
    fftshift = numpy.fft.fftshift
    kc = numpy.pi/dx
    K = numpy.linspace(-kc, kc, N, endpoint=False)
    X_transl = numpy.linspace(-L/2., L/2., N, endpoint=False)
    X = X_transl + x0
    phase = numpy.exp(1j*X*k0)
    Norm = 1/(2*numpy.pi)**(1./4)/numpy.sqrt(sigma)
    f_transl = Norm*numpy.exp(-X_transl**2/(4*sigma**2))*phase
    F_transl = fftshift(fft(f_transl))*dx/numpy.sqrt(2*numpy.pi)
    F = F_transl*numpy.exp(-1j*(x0-L/2)*K)
    return StateVector(F)

def coherent(N, alpha):
    r"""
    Generate a coherent StateVector in the Fock space.

    *Usage*
        >>> sv = coherent(N=20, alpha=2)
        >>> print(sv)
        StateVector(20)

    *Arguments*
        * *alpha*
            A complex number specifying the coherent state.

        * *N*
            A number determining the dimension of the Fock space.

    *Returns*
        * *sv*
            A :class:`pyqo.statevector.StateVector`.

    The coherent state is given by the formula:

        .. math::

            |\alpha\rangle = e^{-\frac {|\alpha|^2} {2}} \sum_{n=0}^{N}
                                \frac {\alpha^n} {\sqrt{n!}} |n\rangle

    Calculation is done using the recursive formula:

        .. math::

            a_0 = e^{- \frac {|\alpha|^2} {2}}

        .. math::

            a_n = a_{n-1} * \frac {\alpha} {\sqrt n}
    """
    x = numpy.empty(N, dtype=complex)
    x[0] = 1
    for n in range(1,N):
        x[n] = x[n-1]*alpha/numpy.sqrt(n)
    x = numpy.exp(-numpy.abs(alpha)**2/2.)*x
    return StateVector(x)


def adjust(array, length):
    """
    Adjust the dimensionality of a 1D array.
    """
    import scipy.interpolate
    X_old = numpy.linspace(0,1,len(array))
    f = scipy.interpolate.interp1d(X_old, array)
    X_new = numpy.linspace(0,1,length)
    return StateVector(f(X_new))

