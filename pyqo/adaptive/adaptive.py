class AdaptivityManager:
    def adapt(self, t_last, t, psi):
        raise NotImplementedError()

    def adapt_operators(self, t_last, state):
        raise NotImplementedError()

    def adapt_statevector(self, state):
        raise NotImplementedError()


class AM_Composite(AdaptivityManager):
    def __init__(self):
        raise NotImplementedError()
