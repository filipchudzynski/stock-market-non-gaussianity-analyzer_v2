
import numpy as np

def intermittency_variance(log_energy):
    '''
    I(s) = Var[L(s,t)]
    '''
    return np.var(log_energy, axis=1)
