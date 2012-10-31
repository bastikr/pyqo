try:
    from . import mpc

    types = {
        mpc.TYPE: (mpc.METHODS, mpc.PROPERTIES),
    }
except:
    types = {}
    print("mpmath not available.")
