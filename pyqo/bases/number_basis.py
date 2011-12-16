import numpy

def coherent_state(alpha, N, dtype=complex):
    from .. import statevector
    x = numpy.empty(N, dtype=dtype)
    x[0] = 1
    for n in range(1,N):
        x[n] = x[n-1]*alpha/numpy.sqrt(n)
    x = numpy.exp(-numpy.abs(alpha)**2/2.)*x
    return statevector.StateVector(x)
