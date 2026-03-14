
import numpy as np

def log_energy_covariance(Ls, max_lag=100):
    '''
    Temporal covariance of log-energy for one scale
    '''
    cov = []
    for lag in range(1, max_lag+1):
        v = np.mean(Ls[:-lag]*Ls[lag:])
        cov.append(v)
    return np.array(cov)

def estimate_lambda2(cov, lags):
    '''
    Estimate λ² from slope of covariance vs log(lag)
    '''
    x = np.log(lags)
    slope, _ = np.polyfit(x, cov, 1)
    return -slope
