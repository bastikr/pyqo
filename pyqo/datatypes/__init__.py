try:
    from . import mpc

    types = {
        mpc.TYPE: (mpc.METHODS, mpc.PROPERTIES),
    }
except:
    print("mpmath not available.")
