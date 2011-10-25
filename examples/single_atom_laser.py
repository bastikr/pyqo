import imp
qo = imp.load_module("pyqo", *imp.find_module("pyqo", [".."]))
import numpy as np

N = 10 # dimension of field Hilbert space
δ_c = 1
δ_a = 1
g = 1
γ = 0.2
κ = 0.1
R = 0.5

# Field
id_f = qo.identity(N)
a = qo.destroy(N)
at = qo.create(N)
n = qo.number(N)

# Atom
id_a = qo.identity(2)
σ_p = qo.sigmap
σ_m = qo.sigmam

# Initial state
ψ_0 = qo.basis(N,0) ^ qo.basis(2,1)

# Hamiltonian
H = δ_c*(at*a^id_a) + δ_a*(id_f^σ_p*σ_m) + g*(a^σ_p) + g*(at^σ_m)

# Jump operators
j1 = γ**(1./2)*(id_f^σ_m)
j2 = κ**(1./2)*(a^id_a)
j3 = R**(1./2)*(id_f^σ_p)
J = [j1, j2, j3]

# Solve Master equation
T = np.linspace(0, 12*np.pi, 80)
ρ = qo.solve_ode(H, ψ_0, T, J)
#ρ = qo.solve_es(H, ψ_0, T, J)

# Expectation values
n_exp, e_exp = qo.expect((n^id_a, id_f^σ_p*σ_m), ρ)

# Calculate Q-function and photon number distribution
x_min, x_max = -5, 5
y_min, y_max = -5, 5
x = np.linspace(x_min, x_max, 30)
y = np.linspace(y_min, y_max, 30)
X, Y = np.meshgrid(x,y)

Q = []
F = []
for ρ_t in ρ:
    ρ_f = qo.ptrace(ρ_t,1)
    Q.append(np.abs(qo.qfunc(ρ_f,X,Y)))
    F.append(np.abs(np.diag(ρ_f)))

# Visualization
import pylab
pylab.figure(1)
pylab.subplot(211)
pylab.xlabel("time")
pylab.ylabel(r"$\langle n \rangle$")
pylab.plot(T, n_exp)
pylab.ylim(ymin=0)
pylab.subplot(212)
pylab.xlabel("time")
pylab.ylabel(r"$\langle P_1 \rangle$")
pylab.ylim((0,1))
pylab.plot(T, e_exp)
pylab.show()

from scipy.misc import factorial

F_x = np.arange(0, N)
def fplot(fig, step):
    axes = fig.add_subplot(111)
    axes.clear()
    pylab.plot(F_x, F[step], "o")
    n_ = np.abs(n_exp[step])
    pylab.plot(F_x, np.exp(-n_)*n_**F_x/factorial(F_x))
qo.animate(len(ρ), fplot)

def qplot(fig,step):
    axes = fig.add_subplot(111)
    axes.clear()
    axes.imshow(Q[step], interpolation='bilinear', origin='lower',
                extent=(x_min, x_max, y_min, y_max))

qo.animate(len(ρ), qplot)
