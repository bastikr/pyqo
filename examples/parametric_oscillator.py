import imp
qo = imp.load_module("pyqo", *imp.find_module("pyqo", [".."]))
import numpy as np
import mpmath as mp
import pylab

#mp.mp.dps = 16
N1 = 40
N2 = 40

# Parameter of system
#mu = mp.mpf("1.2")
#kappa = mp.mpf("1.2")
eta = 1
E1 = 1
E2 = 0
kappa1 = 0.5
kappa2 = 0.5

# Choose initial coherent basis and state
basis1 = qo.bases.NumberBasis(0, N1)
psi1_0 = basis1.basis_vector(0)
basis2 = qo.bases.NumberBasis(0, N2)
psi2_0 = basis2.basis_vector(0)
psi_0 = psi1_0^psi2_0

# Define Hamiltonian and jump operator
a1 = basis1.destroy(1)
a2 = basis2.destroy(1)
id1 = basis1.identity()
id2 = basis2.identity()

H_int = 1j*eta/2*((basis1.create(2)^basis2.destroy(1)) -\
        (basis1.destroy(2)^basis2.create(1)))
H_pump1 = 1j*E1*((basis1.create(1)^id2) - (basis1.destroy(1)^id2))
H_pump2 = 1j*E2*((id1^basis2.create(1)) - (id1^basis2.destroy(1)))
H = H_int + H_pump1 + H_pump2
J = [kappa1*a1^id2, kappa2*id1^a2]

# Solve master equation
T = np.linspace(0, 10, 20)
psi_t = qo.solve_mc(H, psi_0, T, J, trajectories=1).DO
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
    #exp_a.append(qo.expect(a, psi))
    #exp_a_numpy = np.array(exp_a, dtype=complex)

    # Calculate Q-function
    Q1 = qo.qfunc(psi.ptrace(1), X, Y)
    Q_numpy1 = np.array(np.abs(Q1).tolist(), dtype=float)
    Q2 = qo.qfunc(psi.ptrace(0), X, Y)
    Q_numpy2 = np.array(np.abs(Q2).tolist(), dtype=float)

    # Basis states
    pylab.subplot(2,1,1)
    pylab.imshow(Q_numpy1, origin="lower", extent=(-7,7,-7,7))
    pylab.contour(X,Y,Q1)
    pylab.subplot(2,1,2)
    pylab.imshow(Q_numpy2, origin="lower", extent=(-7,7,-7,7))
    pylab.contour(X,Y,Q2)
    i += 1
    pylab.savefig("presentation/parametric_oscillator/%s.png" % i, dpi=300)
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
