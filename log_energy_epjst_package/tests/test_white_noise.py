
import numpy as np
from models.white_noise import white_noise
from log_energy.operators import increment_operator
from log_energy.energy import log_energy_field
from log_energy.intermittency import intermittency_variance
from log_energy.covariance import log_energy_covariance

x = white_noise(5000)
s = 5
coeffs = increment_operator(x, s)
U = log_energy_field(coeffs, s, kappa=10)

lam = intermittency_variance(U)
print("Lambda estimate:", lam)

cov = log_energy_covariance(U, max_lag=50)
print("First 10 covariance values:", cov[:10])
