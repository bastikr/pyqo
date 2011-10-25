import numpy
import scipy.linalg, scipy.integrate
from . import statevector, operators

def as_vector(x):
    return x.reshape(-1)

def as_matrix(x):
    rank = len(x.shape)
    if rank==1:
        dim = int(x.shape[0]**(1./2))
    else:
        dim = numpy.array(x.shape[:rank//2]).prod()
    return x.reshape((dim,dim))

def expect(op, state):
    # Handle multiple operators/states
    if isinstance(op, (list, tuple)):
        return tuple(map(lambda x:expect(x,state), op))
    elif isinstance(state, (list,tuple)):
        return tuple(map(lambda x:expect(op,x), state))
    assert isinstance(op, operators.Operator)
    # Calculate expectation value for given operator and state
    if isinstance(state, statevector.StateVector):
        return numpy.tensordot(state.conj(), op*state, len(state.shape))
    elif isinstance(state, operators.Operator):
        return as_matrix(op*state).trace()
    else:
        raise TypeError("Type of the given state is not supported.")

class ExponentialSeries:
    eigvals = None
    U = None
    hermitian = None
    k = None

    def __init__(self, H, k=1j, hermitian=False):
        self.k = k
        self.hermitian = hermitian
        H = as_matrix(H)
        if hermitian:
            self.eigvals, self.U = scipy.linalg.eigh(H)
        else:
            self.eigvals, self.U = scipy.linalg.eig(H)

    def __call__(self, state, T):
        psi0 = as_vector(state)
        assert len(psi0) == self.U.shape[0]
        U = self.U
        if self.hermitian:
            psi0_ = numpy.dot(U.T.conj(), psi0)
        else:
            psi0_ = numpy.linalg.solve(U, psi0)
        if hasattr(T, "__iter__"):
            result = [state]
            t0 = T[0]
            for t in T[1:]:
                dt = t - t0
                psit_ = numpy.exp(1./self.k*dt*self.eigvals)*psi0_
                psit = numpy.dot(U, psit_)
                result.append(state.__class__(psit.reshape(state.shape)))
            return result
        else:
            psit_ = numpy.exp(1./self.k*T*self.eigvals)*psi0_
            psit = numpy.dot(U, psit_)
            return state.__class__(psit.reshape(state.shape))

def steady(H, J=None):
    if J:
        L = as_matrix(operators.liouvillian(H, J))
        w, U = scipy.linalg.eig(L)
        U_0 = U[:,w.argmin()]
        return operators.Operator(U_0.reshape(H.shape))
        #y = numpy.linalg.solve(L, numpy.zeros(L.shape[0]))
        #return operators.Operator(y.reshape(H.shape))
    else:
        H_ = as_matrix(H)
        y = numpy.linalg.solve(H_, numpy.zeros(H_.shape[0]))
        rank = len(H.shape)//2
        return statevector.StateVector(y.reshape(H.shape[:rank]))

def _as_density_operator(psi, shape):
    if psi.shape == shape:
        return psi
    elif psi.shape*2 == shape:
        return operators.Operator(psi^psi.conj())
    else:
        raise ValueError("Psi has uncompatible dimensionality.")

def solve_es(H, psi, T, J=None):
    if J:
        psi = _as_density_operator(psi, H.shape)
        L = operators.liouvillian(H, J)
        es = ExponentialSeries(L, k=1, hermitian=False)
        return es(psi, T)
    else:
        es = ExponentialSeries(H, k=1j, hermitian=True)
        if isinstance(psi, operators.Operator):
            assert psi.shape == H.shape
            raise NotImplementedError()
        elif psi.shape*2 == H.shape:
            return es(psi, T)
        else:
            raise ValueError("Psi has uncompatible dimensionality.")

def solve_ode(H, psi, T, J=None):
    if J:
        dot = numpy.dot
        rho = _as_density_operator(psi, H.shape)
        rho_ = as_vector(rho)
        H_ = as_matrix(H)
        def f(t, y):
            y = as_matrix(y)
            result = -1j*(dot(H_, y) - dot(y, H_))
            for j in J:
                j = as_matrix(j)
                n = j.H * j/2
                result += dot(j, dot(y, j.H)) - dot(y, n) - dot(n, y)
            return as_vector(result)
        integrator = scipy.integrate.ode(f).set_integrator('zvode')
        integrator.set_initial_value(rho_,T[0])
        result = [rho]
        for t in T[1:]:
            integrator.integrate(t)
            if not integrator.successful():
                raise ValueError("Integration went wrong.")
            result.append(operators.Operator(integrator.y.reshape(H.shape)))
        return result
    else:
        assert isinstance(psi, statevector.StateVector)
        H_ = as_matrix(H)
        psi_ = as_vector(psi)
        def f(t, y):
            return -1j*numpy.dot(H_, y)
        integrator = scipy.integrate.ode(f).set_integrator('zvode')
        integrator.set_initial_value(psi_,T[0])
        result = [psi]
        t0 = T[0]
        for t in T[1:]:
            dt = t-t0
            integrator.integrate(dt)
            if not integrator.successful():
                raise ValueError("Integration went wrong.")
            result.append(psi.__class__(integrator.y.reshape(psi.shape)))
        return result

"""
def solve_mc(H, psi, T, J=None, trajectories=10):
    N = 0
    for j in J:
        N += J.H * J
    H_nH = H - 1j./2*N
    H_nH_ = as_matrix(H_nH)
    psi_ = as_vector(psi)
    def f(t, y):
        return -1j*numpy.dot(H_nH_, y)
    integrator = scipy.integrate.ode(f).set_integrator('zvode')
    integrator.set_initial_value(psi_, T[0])
    result = [psi]
    t0 = T[0]
    for t in T[1:]:
        while
        dt = t-t0
        integrator.integrate(dt)
        if not integrator.successful():
            raise ValueError("Integration went wrong.")
        result.append(psi.__class__(integrator.y.reshape(psi.shape)))
    return result
"""

