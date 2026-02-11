
class Detrender:
    def __init__(self,kappa): self.kappa=kappa
    def window_size(self,s): return int(self.kappa*s)
    def detrend_point(self,x,t,s): raise NotImplementedError
