import imp
qo = imp.load_module("pyqo", *imp.find_module("pyqo", [".."]))
import numpy as np
import mpmath as mp
import pylab

#mp.mp.dps = 16
N = 60

# Parameter of system
#mu = mp.mpf("1.2")
#kappa = mp.mpf("1.2")
mu = 0.35
kappa = 1

# Choose initial coherent basis and state
basis = qo.bases.NumberBasis(0, N)
psi_0 = basis.basis_vector(0)

# Define Hamiltonian and jump operator
a = basis.destroy(1)
H = mu*(basis.create(2) + basis.destroy(2))
J = [kappa*a]

# Solve master equation
T = np.linspace(0, 5, 5)
psi_t = qo.solve_mc(H, psi_0, T, J, trajectories=10).DO
#psi_t = qo.solve_mc_single(H, psi_0, T, J)
#psi_t = qo.solve_master(H, psi_0, T, J)

# Visualization
exp_a = []

X,Y = np.meshgrid(np.linspace(-7,7), np.linspace(-7,7))
log = np.vectorize(mp.log)

#"""
i = 0
for psi  in psi_t:
    pylab.figure()
    # Calculate <a>
    exp_a.append(qo.expect(a, psi))
    exp_a_numpy = np.array(exp_a, dtype=complex)

    # Calculate Q-function
    Q = qo.qfunc(psi, X, Y)
    Q_numpy = np.array(np.abs(Q).tolist(), dtype=float)

    # Basis states
    pylab.imshow(Q_numpy, origin="lower", extent=(-7,7,-7,7))
    pylab.contour(X,Y,Q)
    i += 1
    pylab.savefig("presentation/reduced_parametric_oscillator/%s.png" % i, dpi=300)
    #pylab.plot(exp_a_numpy.real, exp_a_numpy.imag)
    print()
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
