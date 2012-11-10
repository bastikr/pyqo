import random

import numpy
try:
    import mpmath
except ImportError:
    mpmath = None
import scipy.linalg, scipy.integrate

from . import statevector, operators
from .utils import rungekutta

def as_vector(x):
    return x.reshape(-1)

def as_matrix(x):
    rank = x.ndim
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
    if not isinstance(state, (statevector.StateVector, operators.Operator)):
        return tuple(map(lambda x:expect(op,x), state))
    assert isinstance(op, operators.BaseOperator)
    # Calculate expectation value for given operator and state
    if isinstance(state, statevector.StateVector):
        return numpy.tensordot(state.conj(), op*state, state.ndim).item()
    elif isinstance(state, operators.DensityOperator):
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
        rank = H.ndim//2
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
    import bases
    if J is None:
        return H
    N = 0
    for j in J:
        N += j.dagger() * j
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
            result.append(operators.DensityOperator(integrator.y.reshape(H.shape), basis=psi.basis))
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
            result.append(psi.__class__(integrator.y.reshape(psi.shape), basis=psi.basis))
        return result

class MasterTimeStepManager:
    backup = None

    def __call__(self, state, T, adaptiveManager):
        if self.backup is None:
            self.backup = state.copy()

        was_valid = True
        # Adapt system
        if adaptiveManager is None:
            state.t_min["adaptive"] = float("inf")
            state.t_max["adaptive"] = float("inf")
        elif state.t in T or state.t_min["adaptive"] < state.t:
            t_last = float("NaN") if self.backup.t==state.t else self.backup.t
            try:
                basis, t_min, t_max = adaptiveManager.adapt(t_last, state.t, state.psi)
            except TimeStepError as ts_error:
                print("Too big change for basis.")
                state = self.backup.copy()
                state.t_max["adaptive"] = ts_error.new_t_max
                state.t_min["adaptive"] = ts_error.new_t_min
            else:
                state.t_min["adaptive"] = t_min
                state.t_max["adaptive"] = t_max
                if basis is not None:
                    print("Change basis.")
                    state.H, state.J = adaptiveManager.adapt_operators(state.t, basis)
                    state.H_nH = None
                    state.psi = adaptiveManager.adapt_statevector(state.psi, basis)
                    state.jumped = False
                    self.backup = state.copy()


        state.t_min["H"] = float("inf")
        state.t_max["H"] = float("inf")
        state.t_min["J"] = float("inf")
        state.t_max["J"] = float("inf")
        state.t_min["jump"] = float("inf")
        state.t_max["jump"] = float("inf")

        T_remaining = T[T>state.t]
        if len(T_remaining)>0:
            state.t_max["T"] = T_remaining[0]
        return state


def integrate_master(H_nH, rho, dt, J):
    dot = numpy.dot
    def f(t,y):
        result = -1j*(H_nH*y - y*H_nH.H)
        for j in J:
            result += j*y*j.H
        return result
    dtype = mpmath.mpf if rho.dtype==mpmath.mpf else float
    return rungekutta.RK4_5(dtype).integrate(f, rho, (0,dt), rtol=1e-6, atol=1e-6)[-1]

def solve_master(H, rho, T, J=None, adapt=None, time_manager=None):
    if isinstance(rho, statevector.StateVector):
        rho = rho.DO
    if J is None:
        J = []
    if time_manager is None:
        time_manager = MasterTimeStepManager()
    results = [rho.copy()]
    T_calculated = [T[0]]
    state = IntegrationState(T[0], rho, H, J)
    while True:
        state = time_manager(state, T, adapt)
        if state.t in T[1:]:
            if state.t not in T_calculated:
                T_calculated.append(state.t)
                results.append(state.psi.copy())
            if state.t == T[-1]:
                break
        next_t = state.next_t()
        if state.H_nH is None:
            state.H_nH = calculate_H_nH(state.H, state.J)
        state.psi = integrate_master(state.H_nH, state.psi, next_t - state.t, state.J)
        state.t_last = state.t
        state.t = next_t
    return results


class TimeStepError(Exception):
    new_t_min = None
    new_t_max = None
    def __init__(self, new_t_min, new_t_max):
        self.new_t_min = new_t_min
        self.new_t_max = new_t_max

class IntegrationState:
    t = None
    t_last = None
    t_min = None
    t_max = None
    H = None
    J = None
    jumped = False
    jumps = None
    jump_probabilities = None
    psi = None
    H_nH = None

    def __init__(self, t, psi, H=None, J=None):
        self.t = t
        self.psi = psi
        self.H = H
        self.J = J
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
        s = IntegrationState(self.t, self.psi.copy())
        s.H = self.H
        s.jumped = self.jumped
        s.J = None if self.J is None else self.J[:]
        s.t_last = self.t_last
        s.t_min = self.t_min.copy()
        s.t_max = self.t_max.copy()
        return s


class TimeStepManager:
    backup = None

    def __call__(self, state, T, adaptiveManager, dp_max):
        if self.backup is None:
            self.backup = state.copy()

        was_valid = True
        # Adapt system
        if adaptiveManager is None:
            state.t_min["adaptive"] = float("inf")
            state.t_max["adaptive"] = float("inf")
        elif state.t in T or state.t_min["adaptive"] < state.t:
            t_last = float("NaN") if self.backup.t==state.t else self.backup.t
            try:
                basis, t_min, t_max = adaptiveManager.adapt(t_last, state.t,
                                            state.psi, force_adapt=state.jumped)
            except TimeStepError as ts_error:
                print("Too big change for basis.")
                state = self.backup.copy()
                state.t_max["adaptive"] = ts_error.new_t_max
                state.t_min["adaptive"] = ts_error.new_t_min
            else:
                state.t_min["adaptive"] = t_min
                state.t_max["adaptive"] = t_max
                if basis is not None:
                    print("Change basis.")
                    state.H, state.J = adaptiveManager.adapt_operators(state.t, basis)
                    state.H_nH = None
                    state.psi = adaptiveManager.adapt_statevector(state.psi, basis)
                    state.jumped = False
                    self.backup = state.copy()


        state.t_min["H"] = float("inf")
        state.t_max["H"] = float("inf")
        """
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
        """
        state.t_min["J"] = float("inf")
        state.t_max["J"] = float("inf")
        """
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
        """

        # Jump
        if state.J is None:
            state.t_min["jump"] = float("inf")
            state.t_max["jump"] = float("inf")
        else:
            jump_results, jump_probabilities = jump(state.J, state.psi)
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
        T_remaining = T[T>state.t]
        if len(T_remaining)>0:
            state.t_max["T"] = T_remaining[0]
        return state

def jump(J, psi):
    jump_results = []
    jump_norms = []
    for j in J:
        jump_results.append(j*psi)
        jump_norms.append(jump_results[-1].norm()**2)
    jump_norms = numpy.cumsum(jump_norms)
    return jump_results, jump_norms

"""
def rk4(f, y0, t, h):
    k1 = h*f(t, y0)
    k2 = h*f(t+h/2, y0+k1/2)
    k3 = h*f(t+h/2, y0+k2/2)
    k4 = h*f(t+h, y0+k3)
    return y0 + (k1 + 2*k2 + 2*k3 + k4)/6
"""

def integrate_mp(H_nH, psi, dt):
    def f(t, y):
        return -1j*H_nH*y
    return rungekutta.RK4_5(mpmath.mpf).integrate(f, psi, (0,dt), rtol=1e-6, atol=1e-6)[-1]

def integrate(H_nH, psi, dt):
    if mpmath is not None and isinstance(psi.flat[0], (mpmath.mpc, mpmath.mpf)):
        return integrate_mp(H_nH, psi, dt)
    H_nH_ = as_matrix(H_nH)
    psi_ = as_vector(psi)
    def f(t, y):
        return -1j*numpy.dot(H_nH_, y)
    integrator = scipy.integrate.ode(f).set_integrator('zvode', nsteps=10000)
    integrator.set_initial_value(psi_, 0)
    integrator.integrate(dt)
    if not integrator.successful():
        raise ValueError("Integration went wrong.")
    return psi.__class__(integrator.y.reshape(psi.shape), basis=psi.basis,
                         dtype=psi.dtype)

class Trajectory(list):
    pass

def solve_mc_single(H, psi, T, J=None, adapt=None, time_manager=None, dp_max=1e-2, seed=0):
    if time_manager is None:
        time_manager = TimeStepManager()
    T_calculated = [T[0]]
    results = Trajectory([psi.copy()])
    rand_gen = random.Random(seed)
    state = IntegrationState(T[0], psi, H, J)
    while True:
        state = time_manager(state, T, adapt, dp_max)
        if state.t in T[1:]:
            if state.t not in T_calculated:
                T_calculated.append(state.t)
                results.append(state.psi.copy())
            if state.t == T[-1]:
                break
        next_t = state.next_t()
        rand_number = rand_gen.random()/(next_t - state.t)
        P = state.jump_probabilities
        if P is not None and rand_number < P[-1]:
            # Quantum Jump
            state.psi = state.jumps[(P<rand_number).sum()]
            state.jumped = True
        else:
            # non hermitian time evolution
            if state.H_nH is None:
                state.H_nH = calculate_H_nH(state.H, state.J)
            #try:
            state.psi = integrate(state.H_nH, state.psi, next_t - state.t)
            #except:
            #    print("Integration aborted.")
            #    return results
        state.psi.renorm()
        state.t_last = state.t
        state.t = next_t
        #print(state.t)
    return results

class Ensemble(list):
    @property
    def DO(self):
        if len(self) == 0:
            return self
        elif isinstance(self[0], Trajectory):
            return map(lambda e:Ensemble(e).DO, zip(*self))
        else:
            rho = 0
            for state in self:
                rho += state.DO
        return rho/len(self)

def solve_mc(H, psi, T, J=None, trajectories=100, seed=0):
    rand_gen = random.Random(seed)
    results = Ensemble()
    for i in range(trajectories):
        traj_seed = rand_gen.random()
        results.append(solve_mc_single(H, psi, T, J, seed=traj_seed))
    return results

