import imp
qo = imp.load_module("pyqo", *imp.find_module("pyqo", [".."]))
import numpy as np
import mpmath as mp
import pylab

mp.mp.dps = 32

# Parameter of system
omega = 1.

# Choose initial coherent basis and state
basis = qo.bases.NumberBasis(20)
psi_0 = basis.coherent_state(2)

# Hamiltonian
H = omega*basis.create()*basis.destroy()

# Solve master equation
T = np.linspace(0, 2*np.pi, 6)

psi_t = qo.solve_mc_single(H, psi_0, T)

# Visualization
exp_a = []

X,Y = np.meshgrid(np.linspace(-6,6), np.linspace(-6,6))
log = np.vectorize(mp.log)
for i, psi in enumerate(psi_t):
    pylab.subplot(2,3,i+1)
    b = psi.basis
    # Calculate <a>
    a = basis.destroy()
    exp_a.append(qo.expect(a, psi))
    exp_a_numpy = np.array(exp_a, dtype=complex)
    print(exp_a_numpy)

    # Calculate Q-function
    Q = qo.qfunc(psi, X, Y)
    Q_numpy = np.array(Q.tolist(), dtype=float)

    # Basis states
    #states = np.array(psi.basis.states, dtype=complex)
    z = np.array(log(np.abs(psi.dual().conj()*psi)), dtype=float)

    #pylab.scatter(states.real, states.imag, c=z)
    pylab.imshow(Q_numpy, origin="lower", extent=(-6,6,-6,6))
    pylab.contour(X,Y,Q)
    pylab.plot(exp_a_numpy.real, exp_a_numpy.imag)

pylab.show()
