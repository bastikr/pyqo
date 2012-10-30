import imp
qo = imp.load_module("pyqo", *imp.find_module("pyqo", [".."]))
import numpy as np
import mpmath as mp
import pylab

mp.mp.dps = 128

# Parameter of system
omega = mp.mpf("1")
kappa = mp.mpf("0")
eta = mp.mpf("-5")

# Choose initial coherent basis and state
basis = qo.bases.CoherentBasis.create_hexagonal_grid_nearestN(2.1, 0.7, 2.1, 28)
basis = basis.dual_basis(bra=True, ket=False)
psi_0 = basis.coherent_state(mp.mpf("2"))

# Define Hamiltonian (dependent on basis)
def create_H(t, basis):
    return omega*basis.create_destroy(1,1) +\
           eta*(basis.create() + basis.destroy())

def create_J(t, basis):
    return [kappa*basis.destroy()]

# Hamiltonian for initial basis
H_0 = create_H(0, basis)
J_0 = create_J(0, basis)

# Adaptivity manager (governs evolution of basis)
AM = qo.adaptive.AM_Coherent(create_H, create_J)

# Solve master equation
T = np.linspace(0, 2*np.pi, 6)
#T = np.linspace(0, 0.2, 6)
psi_t = qo.solve_mc_single(H_0, psi_0, T, J_0, adapt=AM)

# Visualization
exp_a = []

X,Y = np.meshgrid(np.linspace(0,12), np.linspace(0,12))

log = np.vectorize(mp.log)
print len(psi_t)
for i, psi in enumerate(psi_t):
    pylab.subplot(2,3,i+1)
    b = psi.basis
    # Calculate <a>
    a = b.destroy()
    exp_a.append(np.dot(psi.dual().conj(), a*psi))

    exp_a_numpy = np.array(exp_a, dtype=complex)

    # Calculate Q-function
    Q = qo.qfunc(psi, X, Y)
    Q_numpy = np.array(Q.tolist(), dtype=float)

    # Basis states
    states = np.array(psi.basis.states, dtype=complex)
    z = np.array(log(np.abs(psi.dual().conj()*psi)), dtype=float)

    pylab.scatter(states.real, states.imag, c=z)
    pylab.imshow(Q_numpy, origin="lower", extent=(0,12,0,12))
    pylab.contour(X,Y,Q)
    pylab.plot(exp_a_numpy.real, exp_a_numpy.imag)
    sp0 = b.coherent_scalar_product(mp.mpf("2"), b.states)
    sp1 = b.coherent_scalar_product(mp.conj(exp_a[-1]), b.states)
    error_sv0 = 2*(1-(np.dot(sp0, psi)).real)
    error_sv1 = 2*(1-(np.dot(sp1, psi)).real)
    print(error_sv0, error_sv1)

print(exp_a_numpy)
pylab.show()
