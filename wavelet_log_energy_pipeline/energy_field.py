
import numpy as np

def wavelet_energy(coeffs):
    '''
    Energy field E(s,t) = |W(s,t)|^2
    '''
    return np.abs(coeffs)**2

def log_energy_field(energy, eps=1e-12):
    '''
    Log-energy field L(s,t) = log(E)
    '''
    return np.log(energy + eps)
