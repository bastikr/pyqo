import imp
qo = imp.load_module("pyqo", *imp.find_module("pyqo", [".."]))
import numpy as np

Delta = 2
Omega = 1
phi = np.pi

H = 1./2*(- Delta * qo.sigmaz\
          + Omega * np.exp(1j*phi) * qo.sigmam\
          + Omega * np.exp(-1j*phi) * qo.sigmap)

psi_0 = qo.basis(2,0)

T = np.linspace(0,2*np.pi,30)
psi = qo.solve_ode(H, psi_0, T)

e0 = qo.expect(qo.sigmap*qo.sigmam, psi)
e1 = qo.expect(qo.sigmam*qo.sigmap, psi)

import pylab
pylab.plot(T, e0, T, e1)
pylab.xlabel("Time")
pylab.ylabel("Occupation")
pylab.show()

