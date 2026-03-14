
import numpy as np
from toy_models import white_noise
from wavelet_transform import compute_cwt
from energy_field import wavelet_energy, log_energy_field
from intermittency import intermittency_variance
from cascade_memory import log_energy_covariance, estimate_lambda2
from scale_coupling import scale_mutual_information

# generate signal
x = white_noise(200000)

# scales
scales = np.arange(2,128)

# wavelet transform
W = compute_cwt(x, scales)

# energy
E = wavelet_energy(W)

# log energy
L = log_energy_field(E)

# intermittency spectrum
I = intermittency_variance(L)
print("Intermittency spectrum:", I[:10])

# cascade memory at one scale
Ls = L[10]
cov = log_energy_covariance(Ls)
lags = np.arange(1,len(cov)+1)
lam2 = estimate_lambda2(cov,lags)
print("Estimated lambda^2:",lam2)

# scale coupling
MI = scale_mutual_information(L)
print("MI matrix shape:",MI.shape)
