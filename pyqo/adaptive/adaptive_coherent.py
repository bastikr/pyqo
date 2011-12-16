import numpy

from . import adaptive

from ..operators import qfunc
from ..import statevector
from ..import operators
from ..bases import coherent_basis
from ..algorithms import TimeStepError

class AM_Coherent(adaptive.AdaptivityManager):
    H_func = None
    J_func = None
    def __init__(self, H_func=None, J_func=None):
        self.H_func = H_func
        self.J_func = J_func

    def adapt(self, t_last, t, psi):
        lattice = psi.basis.lattice
        dt = t-t_last

        # Consider 3 different groups:
        #   (1) current basis states
        #   (2) nearest neighbor states
        #   (3) next nearest neighbor states
        ind0, ind1, ind2 = lattice.neighbors(2)
        states0 = numpy.array(lattice.states())
        states1 = numpy.array(lattice.states(ind1))
        states2 = numpy.array(lattice.states(ind2))

        # Calculate Q-function for these states.
        Q0 = qfunc(psi, states0)
        Q1 = qfunc(psi, states1)
        Q2 = qfunc(psi, states2)
        order0 = Q0.argsort()
        order1 = Q1.argsort()

        # Determine which old basis states have to be replaced by
        # nearest neighbor states.
        subs = {}
        min_q = Q0[order0[-1]]
        for i in range(min(len(Q2),len(Q1))):
            i0 = order0[i]
            i1 = order1[-(i+1)]
            if Q0[i0] < Q1[i1]:
                subs[ind0[i0]] = ind1[i1]
                min_q = Q1[i1]
            else:
                break

        # Find out if next nearest neighbor states would be better than this
        # new basis.
        if Q2.max() > min_q:
            if subs:
                raise TimeStepError(t_last+dt/3, t_last+dt/2)
            else:
                raise TimeStepError(t_last+dt/10, t_last+dt/5)


        # Estimate point of time when basis has to change again
        l = len(subs)
        if l==0:
            dt_min = 1.2*dt
            dt_max = 1.6*dt
        elif 1<=l<2:
            dt_min = 0.8*dt
            dt_max = 1.2*dt
        elif 2<=l<4:
            dt_min = 0.5*dt
            dt_max = 0.8*dt
        elif 4<=l<6:
            dt_min = 0.2*dt
            dt_max = 0.5*dt
        else:
            raise TimeStepError(t_last+0.2*dt, t_last+0.5*dt)
        t_min = t+dt_min
        t_max = t+dt_max
        if l==0:
            return None, t_min, t_max

        # Create new lattice consisting of states with highest values of
        # Q-function.
        new_lattice = psi.basis.lattice.copy()
        lattice.clear_selection()
        for i in ind0:
            if i in subs:
                lattice.select(subs[i])
            else:
                lattice.select(i)
        basis = coherent_basis.CoherentBasis(lattice)
        return basis, t_min, t_max

    def adapt_operators(self, t, basis):
        # Calculate operators in new basis
        H = None if self.H_func is None else self.H_func(t, basis)
        J = None if self.J_func is None else self.J_func(t, basis)
        return (H,J)

    def adapt_statevector(self, psi, basis):
        f = basis.transform_func(psi.basis)
        if isinstance(psi, statevector.StateVector):
            psi_new = psi.__class__(f(psi), basis=basis, dtype=psi.dtype)
            psi_new.renorm()
            return(psi_new)
        else:
            raise NotImplementedError()

