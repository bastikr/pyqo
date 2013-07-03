import os
import pyqo
import numpy

def guess_type(x):
    try:
        return int(x)
    except ValueError:
        pass
    try:
        return float(x)
    except ValueError:
        pass
    return x

def filename2dict(path, types=None):
    directory, name = os.path.split(path)
    assert name
    if name.endswith(".npz"):
        name = name[:-4]
    d = {}
    for pair in name.split(";"):
        key, value = pair.split("=")
        if types is None:
            value = guess_type(value)
        else:
            value = types[key](value)
        d[key] = value
    return d

def dict2filename(d):
    keys = d.keys()
    keys.sort()
    return ";".join(str(k)+"="+str(d[k]) for k in keys)

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

