
import numpy as np

def local_energy(coeffs):
    return coeffs ** 2

def sliding_baseline(energy, window):
    kernel = np.ones(window) / window
    return np.convolve(energy, kernel, mode='same')

def log_energy_field(coeffs, s, kappa=10):
    window = max(int(kappa * s), 5)
    E = local_energy(coeffs)
    baseline = sliding_baseline(E, window)
    eps = 1e-12
    return np.log((E + eps) / (baseline + eps))
