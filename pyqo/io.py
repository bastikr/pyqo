import pyqo
import numpy

def save(path, obj):
    f = open(path, "w")
    f.write(repr(obj))
    f.close()

def load(path, namespace=None):
    f = open(path)
    buf = f.read()
    f.close()
    ns = {"pyqo": pyqo}
    ns.update(numpy.__dict__)
    if "mpc" in buf:
        import mpmath
        ns.update(mpmath.__dict__)
    if namespace is not None:
        ns.update(namespace)
    return eval(buf, ns)

