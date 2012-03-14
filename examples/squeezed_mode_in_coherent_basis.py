import imp
qo = imp.load_module("pyqo", *imp.find_module("pyqo", [".."]))
import numpy as np
import mpmath as mp
import pylab

# 128
mp.mp.dps = 128

# Parameter of system
# (1.2, 1.4)
mu = mp.mpf("1.5")
kappa = mp.mpf("1.5")

# Choose initial coherent basis and state
# (0, 1.5, 4)
# (0, 1.6, 3)
basis = qo.bases.CoherentBasis.create_hexagonal_grid_nearestN(0, 1.7, 0, 19)
psi_0 = basis.coordinates(mp.mpf("0"))

# Define Hamiltonian (dependent on basis)
def create_H(t, basis):
    adag = basis.create(2, True, False)
    a = basis.destroy(2, True, False)
    return mu*(adag+a)

def create_J(t, basis):
    return [kappa*basis.destroy(1, True, False)]

# Hamiltonian for initial basis
H_0 = create_H(0, basis)
J_0 = create_J(0, basis)

# Adaptivity manager (governs evolution of basis)
AM = qo.adaptive.AM_Coherent(create_H, create_J)

# Solve master equation
# (0, 0.5, 11)
T = np.linspace(0, 0.7, 15)
psi_t = qo.solve_mc_single(H_0, psi_0, T, J_0, adapt=AM)
#psi_t = qo.solve_mc_single(H_0, psi_0, T, J_0)

# Visualization
exp_a = []

X,Y = np.meshgrid(np.linspace(-14,14), np.linspace(-14,14))
log = np.vectorize(mp.log)

#"""
i = 0
for psi  in psi_t:
    i+=1
    pylab.figure()
    b = psi.basis
    # Calculate <a>
    a = qo.Operator(b.trafo * b.states, basis=b)
    exp_a.append(qo.expect(a, psi))
    exp_a_numpy = np.array(exp_a, dtype=complex)

    # Calculate Q-function
    Q = qo.qfunc(psi, X, Y)
    Q_numpy = np.array(Q.tolist(), dtype=float)

    # Basis states
    states = np.array(psi.basis.states, dtype=complex)
    z = np.array(log(np.abs(psi.dual().conj()*psi)), dtype=float)

    pylab.scatter(states.real, states.imag, c=z)
    pylab.imshow(Q_numpy, origin="lower", extent=(-14,14,-14,14))
    pylab.contour(X,Y,Q)
    pylab.plot(exp_a_numpy.real, exp_a_numpy.imag)
#"""

"""
print len(psi_t)
for i, psi in enumerate(psi_t[-6:]):
    pylab.subplot(2,3,i+1)
    b = psi.basis
    # Calculate <a>
    a = qo.Operator(b.trafo * b.states, basis=b)
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
"""
pylab.show()
