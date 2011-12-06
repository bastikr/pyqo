import imp
qo = imp.load_module("pyqo", *imp.find_module("pyqo", [".."]))
import numpy as np

N = 10 # dimension of field Hilbert space
delta_c = 1
delta_a = 1
g = 1
gamma = 0.1
kappa = 0.1
R = 0.5

# Field
id_f = qo.identity(N)
a = qo.destroy(N)
at = qo.create(N)
n = qo.number(N)

# Atom
id_a = qo.identity(2)

# Initial state
psi_0 = qo.basis_vector(N,0) ^ qo.basis_vector(2,1)

# Hamiltonian
H = delta_c*(at*a^id_a) + delta_a*(id_f^qo.sigmap*qo.sigmam) + g*(a^qo.sigmap) + g*(at^qo.sigmam)

# Jump operators
j1 = gamma**(1./2)*(id_f^qo.sigmam)
j2 = kappa**(1./2)*(a^id_a)
j3 = R**(1./2)*(id_f^qo.sigmap)
J = [j1, j2, j3]

# Solve Master equation
T = np.linspace(0, 3*np.pi, 20)
rho = qo.solve_ode(H, psi_0, T, J)
#rho = qo.solve_mc_single(H, psi_0, T, J, seed=0)
#rho = qo.solve_es(H, psi_0, T, J)

# Expectation values
n_exp, e_exp = qo.expect((n^id_a, id_f^qo.sigmap*qo.sigmam), rho)

# Calculate Q-function and photon number distribution
x_min, x_max = -5, 5
y_min, y_max = -5, 5
x = np.linspace(x_min, x_max, 30)
y = np.linspace(y_min, y_max, 30)
X, Y = np.meshgrid(x,y)

Q = []
F = []
for rho_t in rho:
    rho_f = rho_t.ptrace(1)
    Q.append(np.abs(qo.qfunc(rho_f,X,Y)))
    F.append(np.abs(np.diag(rho_f)))

# Visualization
import pylab
pylab.figure(1)
pylab.subplot(211)
pylab.xlabel("time")
pylab.ylabel(r"$\langle n \rangle$")
pylab.plot(T, np.abs(n_exp))
pylab.ylim(ymin=0)
pylab.subplot(212)
pylab.xlabel("time")
pylab.ylabel(r"$\langle P_1 \rangle$")
pylab.ylim((0,1))
pylab.plot(T, np.abs(e_exp))
pylab.show()

from scipy.misc import factorial

F_x = np.arange(0, N)
def fplot(fig, step):
    axes = fig.add_subplot(111)
    axes.clear()
    pylab.plot(F_x, F[step], "o")
    n_ = np.abs(n_exp[step])
    pylab.plot(F_x, np.exp(-n_)*n_**F_x/factorial(F_x))
qo.animate(len(rho), fplot)

def qplot(fig,step):
    axes = fig.add_subplot(111)
    axes.clear()
    axes.imshow(Q[step], interpolation='bilinear', origin='lower',
                extent=(x_min, x_max, y_min, y_max))

qo.animate(len(rho), qplot)
