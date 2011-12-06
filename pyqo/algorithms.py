import random

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
    # Handle multiple operators
    if isinstance(op, (list, tuple)):
        return tuple(map(lambda x:expect(x,state), op))
    # Handle multiple trajectories
    if isinstance(state, Ensemble):
        result = []
        N = len(state)
        for states in zip(*state):
            result.append(sum(expect(op, states))/N)
        return tuple(result)
    # Handle multiple states
    if isinstance(state, (list,tuple)):
        return tuple(map(lambda x:expect(op,x), state))
    assert isinstance(op, operators.BaseOperator)
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
        return psi.DO
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

def calculate_H_nH(H, J):
    if J is None:
        return H
    N = 0
    for j in J:
        N += j.H * j
    return H - 1j*N/2


def solve_ode(H, psi, T, J=None):
    if J:
        dot = numpy.dot
        rho = _as_density_operator(psi, H.shape)
        rho_ = as_vector(rho)
        H_nH = as_matrix(calculate_H_nH(H,J))
        def f(t,y):
            y = as_matrix(y)
            result = -1j*(dot(H_nH, y) - dot(y, H_nH.H))
            for j in J:
                j = as_matrix(j)
                result += dot(j, dot(y, j.H))
            return as_vector(result)
        integrator = scipy.integrate.ode(f).set_integrator('zvode')
        integrator.set_initial_value(rho_,T[0])
        result = [rho]
        for t in T[1:]:
            integrator.integrate(t)
            if not integrator.successful():
                raise ValueError("Integration went wrong.")
            result.append(operators.DensityOperator(integrator.y.reshape(H.shape)))
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

class TimeStepError(Exception):
    new_dt = None
    def __init__(self, new_dt):
        self.new_dt = new_dt

class IntegrationState:
    t = None
    H = None
    J = None
    jumps = None
    jump_probabilities = None
    psi = None
    H_nH = None

    def __init__(self, psi, t):
        self.psi = psi
        self.t_last = None
        self.t = t
        self.t_min = {
            "adaptive": -float("inf"),
            "H": -float("inf"),
            "J": -float("inf"),
            "jump": None,
            "T": None}
        self.t_max = self.t_min.copy()

    def next_t(self):
        return min(self.t_max.values())

    def copy(self):
        s = IntegrationState()
        s.H = self.H
        s.J = None if self.J is None else self.J.copy()
        s.t = self.t
        s.t_last = self.t_last
        s.t_min = self.t_min.copy()
        s.t_max = self.t_max.copy()
        s.psi = self.psi.copy()


class TimeStepManager:
    backup = None

    def __call__(self, state, H, J, T, adaptiveManager, dp_max):
        # Adapt system
        if adaptiveManager is None:
            state.t_min["adaptive"] = float("inf")
            state.t_max["adaptive"] = float("inf")

        elif t in T or state.t_min["adaptive"] < state.t:
            dt = None if self.backup is None else t - self.backup.t
            try:
                changed, t_min, t_max = adaptiveManager.adapt(dt, state.t, psi, H, J)
            except TimeStepError as ts_error:
                state = self.backup.copy()
                state.t_max["adaptive"] = t_max
                state.t_min["adaptive"] = t_min
                return state
            if changed:
                self.backup = state.copy()
                psi = adaptiveManager.adaptStateVector(psi)
                H, J = adaptiveManager.adaptOperators(H, J)
            state.t_min["adaptive"] = t_min
            state.t_max["adaptive"] = t_max

        # Adapt H
        if state.t_min["H"] < state.t:
            from . import dynamic_operators
            if isinstance(H, dynamic_operators.DynamicOperator):
                state.H = H(t)
                state.t_min["H"] = H.t_min(t)
                state.t_max["H"] = H.t_max(t)
            else:
                state.H = H
                state.t_min["H"] = float("inf")
                state.t_max["H"] = float("inf")
            state.H_nH = None

        # Adapt J
        if J is None:
            state.t_min["J"] = float("inf")
            state.t_max["J"] = float("inf")
        elif state.t_min["J"] < state.t:
            from . import dynamic_operators
            state.J = []
            t_min_J = []
            t_max_J = []
            for j in J:
                # Change J
                if isinstance(j, dynamic_operators.DynamicOperator):
                    state.J.append(j(state.t))
                    t_min_J.append(j.t_min(state.t))
                    t_max_J.append(H.t_max(state.t))
                else:
                    state.J.append(j)
                    t_min_J.append(float("inf"))
                    t_max_J.append(float("inf"))
            state.H_nH = None
            state.t_min["J"] = min(t_min_J)
            state.t_max["J"] = min(t_max_J)

        # Jump
        if J is None:
            state.t_min["jump"] = float("inf")
            state.t_max["jump"] = float("inf")
        else:
            jump_results, jump_probabilities = jump(J, state.psi)
            if jump_probabilities[-1] == 0:
                if state.t_last is None:
                    dt_max = dp_max
                else:
                    assert state.t_max["jump"] is not None
                    dt_max = (state.t_max["jump"] - state.t_last)*1.2
            else:
                dt_max = dp_max/jump_probabilities[-1]
            state.t_max["jump"] = state.t + dt_max
            state.jumps = jump_results
            state.jump_probabilities = jump_probabilities

        state.t_max["T"] = T[T>state.t][0]
        return state

def jump(J, psi):
    jump_results = []
    jump_norms = []
    for j in J:
        jump_results.append(j*psi)
        jump_norms.append(jump_results[-1].norm()**2)
    jump_norms = numpy.cumsum(jump_norms)
    return jump_results, jump_norms

def integrate(H_nH, psi, dt):
    H_nH_ = as_matrix(H_nH)
    psi_ = as_vector(psi)
    def f(t, y):
        return -1j*numpy.dot(H_nH_, y)
    integrator = scipy.integrate.ode(f).set_integrator('zvode')
    integrator.set_initial_value(psi_, 0)
    integrator.integrate(dt)
    if not integrator.successful():
        raise ValueError("Integration went wrong.")
    return psi.__class__(integrator.y.reshape(psi.shape), dtype=psi.dtype)

def solve_mc_single(H, psi, T, J=None, adapt=None, time_manager=None, dp_max=1e-2, seed=0):
    if time_manager is None:
        time_manager = TimeStepManager()
    if isinstance(T, (float, int)):
        T = [0, T]
        results = []
    else:
        results = [psi.copy()]
    rand_gen = random.Random(seed)
    state = IntegrationState(psi, T[0])
    while state.t < T[-1]:
        # In principle also the last step has to be checked!
        state = time_manager(state, H, J, T, adapt, dp_max)
        next_t = state.next_t()
        rand_number = rand_gen.random()/(next_t - state.t)
        P = state.jump_probabilities
        if P is not None and rand_number < P[-1]:
            # Quantum Jump
            state.psi = state.jumps[(P<rand_number).sum()]
        else:
            # non hermitian time evolution
            if state.H_nH is None:
                state.H_nH = calculate_H_nH(H, J)
            state.psi = integrate(state.H_nH, state.psi, next_t - state.t)
        state.psi.renorm()
        state.t_last = state.t
        state.t = next_t
        if state.t in T:
            results.append(state.psi.copy())
    return results

class Ensemble(list):
    pass

def solve_mc(H, psi, T, J=None, trajectories=100, seed=0):
    rand_gen = random.Random(seed)
    results = Ensemble()
    for i in range(trajectories):
        traj_seed = rand_gen.random()
        results.append(solve_mc_single(H, psi, T, J, seed=traj_seed))
    return results

