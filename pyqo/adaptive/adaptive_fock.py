from . import adaptive
from ..algorithms import TimeStepError

class AM_Fock(adaptive.AdaptivityManager):
    H_func = None
    J_func = None

    N_min = None
    N_max = None
    soft_limit = None
    hard_limit = None
    def __init__(self, H_func, J_func, N_min, N_max,
                 soft_limit=1e-7, hard_limit=1e-6):
        self.H_func = H_func
        self.J_func = J_func
        self.N_min = N_min
        self.N_max = N_max
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit

    def adapt(self, t_last, t, psi):
        raise NotImplementedError()

    def adapt_operators(self, t_last, state):
        # Calculate operators in new basis
        H = None if self.H_func is None else self.H_func(t, basis)
        J = None if self.J_func is None else self.J_func(t, basis)
        return (H,J)

    def adapt_statevector(self, state):
        if isinstance(psi, statevector.StateVector):
            psi_new = psi.change_basis(basis)
            psi_new.renorm()
            return psi_new
        else:
            raise NotImplementedError()
