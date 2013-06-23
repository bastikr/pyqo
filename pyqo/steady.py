import numpy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from . import algorithms, operators


def spre(op):
    op = algorithms.as_matrix(op)
    id = numpy.eye(op.shape[0])
    return scipy.sparse.kron(op, id)

def spost(op):
    op = algorithms.as_matrix(op)
    id = numpy.eye(op.shape[0])
    return scipy.sparse.kron(id, op.T)

def liouvillian(H, J=()):
    L = -1j*(spre(H) - spost(H))
    for j in J:
        n = j.H*j/2.
        L = L+spre(j)*spost(j.H) - spost(n) - spre(n)
    return L

def steadystate(H, J, v0=None, tol=1e-6):
    L = liouvillian(H,J)
    #v = algorithms.as_vector(vac)
    if v0 is None:
        v = numpy.random.randn(L.shape[0])
    else:
        v = algorithms.as_vector(v0)

    N = lambda x:scipy.linalg.norm(x, numpy.inf)
    while N(L*v) > tol:
        v = scipy.sparse.linalg.spsolve(L,v)
        v /= N(v)
    rho_ss = operators.DensityOperator(v.reshape(H.shape), basis=H.basis)
    rho_ss /= numpy.diag(algorithms.as_matrix(v)).sum()
    return rho_ss

