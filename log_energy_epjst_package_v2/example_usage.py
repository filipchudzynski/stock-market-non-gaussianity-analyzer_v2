
from log_energy_lambda import log_energy, estimate_lambda2_from_covariance
from test_models import white_noise, simple_mrw

scale = 32
max_lag = 200

print("---- WHITE NOISE TEST ----")
r = white_noise(200000)
L = log_energy(r, scale)
lam2, lags, cov = estimate_lambda2_from_covariance(L, max_lag)
print("Estimated lambda^2:", lam2)

print("\n---- MRW TEST ----")
r = simple_mrw(200000, lambda2=0.05)
L = log_energy(r, scale)
lam2, lags, cov = estimate_lambda2_from_covariance(L, max_lag)
print("Estimated lambda^2:", lam2)
