import imp
qo = imp.load_module("pyqo", *imp.find_module("pyqo", [".."]))
import numpy as np
import mpmath as mp
import pylab

mp.mp.dps = 64

# Parameter of system
omega = mp.mpf("1")
eta = mp.mpf("0.2")

# Choose initial coherent basis and state
basis = qo.bases.CoherentBasis.create_hexagonal_grid_rings(2, 1.3, 3)
basis = basis.dual_basis(bra=True, ket=False)
psi_0 = basis.coherent_state(mp.mpf("2.1"))

# Define Hamiltonian (dependent on basis)
def create_H(t, basis):
    return omega*basis.create_destroy(1,1)

def create_J(t, basis):
    return [eta*basis.destroy()]

# Hamiltonian for initial basis
H_0 = create_H(0, basis)
J_0 = create_J(0, basis)

# Adaptivity manager (governs evolution of basis)
AM = qo.adaptive.AM_Coherent(create_H, create_J)

# Solve master equation
T = np.linspace(0, np.pi/2, 6)
#T = np.linspace(0, 0.2, 6)
psi_t = qo.solve_mc_single(H_0, psi_0, T, J_0, adapt=AM)

# Visualization
exp_a = []

X,Y = np.meshgrid(np.linspace(-6,6), np.linspace(-6,6))
log = np.vectorize(mp.log)
print len(psi_t)
for i, psi in enumerate(psi_t):
    pylab.subplot(2,3,i+1)
    b = psi.basis
    # Calculate <a>
    a = psi.basis.destroy()
    exp_a.append(qo.expect(a, psi))
    exp_a_numpy = np.array(exp_a, dtype=complex)

    # Calculate Q-function
    Q = qo.qfunc(psi, X, Y)
    Q_numpy = np.array(Q.tolist(), dtype=float)

    # Basis states
    states = np.array(psi.basis.states, dtype=complex)
    z = np.array(log(np.abs(psi.dual().conj()*psi)), dtype=float)

    pylab.scatter(states.real, states.imag, c=z)
    pylab.imshow(Q_numpy, origin="lower", extent=(-6,6,-6,6))
    pylab.contour(X,Y,Q)
    pylab.plot(exp_a_numpy.real, exp_a_numpy.imag)

pylab.show()
