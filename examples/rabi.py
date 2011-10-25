import imp
qo = imp.load_module("pyqo", *imp.find_module("pyqo", [".."]))
import numpy as np

σ_z = qo.sigmaz
σ_p = qo.sigmap
σ_m = qo.sigmam

Δ = 2
Ω = 1
φ = np.pi

H = 1./2*(- Δ * σ_z + Ω * np.exp(1j*φ) * σ_m + Ω * np.exp(-1j*φ) * σ_p)

ψ_0 = qo.basis(2,0)

T = np.linspace(0,2*np.pi,30)
ψ = qo.solve_ode(H, ψ_0, T)

e0 = qo.expect(σ_p*σ_m, ψ)
e1 = qo.expect(σ_m*σ_p, ψ)

import pylab
pylab.plot(T, e0, T, e1)
pylab.show()

