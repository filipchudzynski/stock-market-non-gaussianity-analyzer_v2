
import numpy as np

def sliding_energy(returns, scale):
    """Compute sliding-window energy E_s(t) = sum_{k=0}^{s-1} r_{t-k}^2"""
    r2 = returns**2
    kernel = np.ones(scale)
    energy = np.convolve(r2, kernel, mode="valid")
    return energy

def log_energy(returns, scale, eps=1e-12):
    """Compute centered log-energy field"""
    E = sliding_energy(returns, scale)
    L = np.log(E + eps)
    L_centered = L - np.mean(L)
    return L_centered

def log_energy_covariance(L, max_lag):
    """Compute covariance C(Δt) = <L(t)L(t+Δt)>"""
    cov = []
    for lag in range(1, max_lag + 1):
        v = np.mean(L[:-lag] * L[lag:])
        cov.append(v)
    return np.array(cov)

def estimate_lambda2_from_covariance(L, max_lag=100):
    """Estimate λ² from slope of C(Δt) vs log(Δt)"""
    cov = log_energy_covariance(L, max_lag)
    lags = np.arange(1, max_lag + 1)
    x = np.log(lags)
    y = cov
    slope, intercept = np.polyfit(x, y, 1)
    lambda2 = -slope
    return lambda2, lags, cov
