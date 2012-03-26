import imp
qo = imp.load_module("pyqo", *imp.find_module("pyqo", [".."]))
import numpy as np
import mpmath as mp
import pylab

mp.mp.dps = 64
N = 50

# Parameters of system
T = np.linspace(0, 6, 20)
mu = 0.35
kappa = 1

# Choose initial fock basis and state
basis = qo.bases.FockBasis(0, N)
psi_0 = basis.basis_vector(0)

lattice = qo.utils.lattice.HexagonalLattice(d=1.3)

# Choose initial coherent basis and state
#basis = qo.bases.CoherentBasis.create_hexagonal_grid_nearestN(0, 1.7, 0, 19)
#basis = basis.dual_basis(bra=True, ket=False)
#psi_0 = basis.coherent_state(mp.mpf("0"))

# Define Hamiltonian and jump operator
def create_H(t, basis):
    adag = basis.create(2)
    a = basis.destroy(2)
    return mu*(adag+a)

def create_J(t, basis):
    return [kappa*basis.destroy()]
#
# Hamiltonian for initial basis
H_0 = create_H(0, basis)
J_0 = create_J(0, basis)

# Adaptivity manager (governs evolution of basis)

class AM_Hybrid(qo.adaptive.AdaptivityManager):
    limit = None
    phase = None
    lat = None
    N = None
    AM_coherent = None
    def __init__(self, AM_coherent, limit, N, lat):
        self.AM_coherent = AM_coherent
        self.limit = limit
        self.N = N
        self.phase = 0
        self.lat = lat

    def adapt(self, t_last, t, psi, force_adapt=False):
        if self.phase == 0:
            h = t - t_last
            if abs(psi[-1])>self.limit:
                self.phase = 1
                basis = qo.bases.coherent_basis.find_coherent_basis(
                                    psi, N, self.lat)
                basis = basis.dual_basis(bra=True, ket=False)
                return basis, t + 0.8*h, t + 1.2*h
            else:
                return None, t + 0.8*h, t + 1.2*h
        else:
            return self.AM_coherent.adapt(t_last, t, psi)

    def adapt_operators(self, *args, **kwargs):
        # Also valid for fock basis:)
        return self.AM_coherent.adapt_operators(*args, **kwargs)

    def adapt_statevector(self, *args, **kwargs):
        # Also valid for fock basis:)
        return self.AM_coherent.adapt_statevector(*args, **kwargs)

AM_coherent = qo.adaptive.AM_Coherent(create_H, create_J)
AM = AM_Hybrid(AM_coherent, 1e-6, 30, lattice)
# Solve master equation
# (0, 0.5, 11)
psi_t = qo.solve_mc_single(H_0, psi_0, T, J_0, adapt=AM)
#psi_t = qo.solve_mc_single(H_0, psi_0, T, J_0)
#psi_t = qo.solve_mc_single(H, psi_0, T, J)


# Visualization
X,Y = np.meshgrid(np.linspace(-7,7,40), np.linspace(-7,7,40))
log = np.vectorize(mp.log)

i = 0
for psi  in psi_t:
    pylab.figure()

    # Calculate Q-function
    Q = qo.qfunc(psi, X, Y)
    Q_numpy = np.array(np.abs(Q).tolist(), dtype=float)

    if isinstance(psi.basis, qo.bases.CoherentBasis):
        # Basis states
        states = np.array(psi.basis.states, dtype=complex)
        z = np.array(log(np.abs(psi.dual().conj()*psi)), dtype=float)
        pylab.scatter(states.real, states.imag, c=z)
    pylab.imshow(Q_numpy, origin="lower", extent=(-7,7,-7,7))
    pylab.contour(X,Y,Q)
    i += 1
    #pylab.savefig("presentation/reduced_parametric_oscillator/%s.png" % i, dpi=300)

pylab.show()

