
import numpy as np
import pywt

def compute_cwt(signal, scales, wavelet='morl'):
    '''
    Continuous Wavelet Transform using PyWavelets.
    Returns coefficients W(s,t)
    '''
    coeffs, freqs = pywt.cwt(signal, scales, wavelet)
    return coeffs
