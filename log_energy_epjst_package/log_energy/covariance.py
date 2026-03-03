
import numpy as np

def log_energy_covariance(U, max_lag):
    N = len(U)
    cov = []
    for tau in range(1, max_lag):
        c = np.cov(U[:-tau], U[tau:])[0,1]
        cov.append(c)
    return np.array(cov)
