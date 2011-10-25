import imp
qo = imp.load_module("pyqo", *imp.find_module("pyqo", [".."]))
import numpy as np

N = 10 # dimension of field Hilbert space
δ_c = 1
δ_a = 2
g = 1
γ = 0.1
κ = 0.1

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

# Solve Master equation
T = np.linspace(0, 2*np.pi, 30)
ρ = qo.solve_ode(H, ψ_0, T, [γ**(1/2)*(id_f^σ_m), κ**(1/2)*(a^id_a)])

# Expectation values
n_exp = qo.expect(n^id_a, ρ)
e_exp = qo.expect(id_f^σ_p*σ_m, ρ)

# Q-function
x = np.linspace(-4,4,40)
y = np.linspace(-4,4,40)
X, Y = np.meshgrid(x,y)


# Visualization
import pylab
pylab.figure(1)
pylab.subplot(211)
pylab.xlabel("time")
pylab.ylabel(r"$\langle n \rangle$")
pylab.plot(T, n_exp)
pylab.subplot(212)
pylab.xlabel("time")
pylab.ylabel(r"$\langle P_1 \rangle$")
pylab.plot(T, e_exp)
pylab.show()

Q = []
for ρ_t in ρ:
    ρ_f = qo.ptrace(ρ_t,1)
    Q.append(np.abs(qo.qfunc(ρ_f,X,Y)))

def qplot(fig,step):
    axes = fig.add_subplot(111)
    axes.clear()
    axes.imshow(Q[step])

qo.animate(len(ρ), qplot)
