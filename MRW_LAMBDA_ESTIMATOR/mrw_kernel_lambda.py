import numpy as np
import matplotlib.pyplot as plt
import os, time
from scipy.signal import fftconvolve
from scipy.linalg import toeplitz

# =========================================
# LOGGER
# =========================================
class Logger:
    def __init__(self, path):
        self.f = open(path, "w")

    def log(self, msg):
        t = time.strftime("[%H:%M:%S] ")
        line = t + msg
        print(line)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()

# =========================================
# MRW GENERATOR
# =========================================
def generate_log_field(N, lam2):
    tau = np.arange(N)
    tau[0] = 1

    cov = lam2 * np.log(N / tau)
    cov[0] = lam2 * np.log(N)

    C = toeplitz(cov)
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals[eigvals < 0] = 0

    z = np.random.randn(N)
    omega = eigvecs @ (np.sqrt(eigvals) * z)
    omega -= np.mean(omega)
    return omega

def generate_mrw(N, lam2):
    omega = generate_log_field(N, lam2)
    noise = np.random.randn(N)
    X = np.cumsum(np.exp(omega) * noise)
    return X, omega

# =========================================
# FILTERS
# =========================================
def moving_average(x, s):
    k = np.ones(s)/s
    return fftconvolve(x, k, mode="same")

def gaussian_filter(x, s):
    L = int(4*s)
    t = np.arange(-L, L+1)
    k = np.exp(-t**2/(2*s**2))
    k /= k.sum()
    return fftconvolve(x, k, mode="same")

# =========================================
# ENERGY → OMEGA
# =========================================
def compute_energy(x, s, method):
    if method == "ma":
        xs = moving_average(x, s)
    else:
        xs = gaussian_filter(x, s)

    fluct = x - xs
    energy = fluct**2

    if method == "ma":
        E = moving_average(energy, s)
    else:
        E = gaussian_filter(energy, s)

    return E

def compute_omega(E):
    return 0.5*np.log(E + 1e-12)

# =========================================
# FAST COVARIANCE
# =========================================
def cov_fft(x, maxlag):
    x = x - np.mean(x)
    N = len(x)

    f = np.fft.fft(x, 2*N)
    acf = np.fft.ifft(f*np.conj(f)).real[:N]
    acf /= np.arange(N,0,-1)

    lags = np.arange(1, maxlag)
    return lags, acf[1:maxlag]

# =========================================
# PHASE SURROGATE
# =========================================
def phase_surrogate(x):
    f = np.fft.rfft(x)
    phases = np.exp(1j * np.random.uniform(0, 2*np.pi, len(f)))
    f_new = np.abs(f) * phases
    return np.fft.irfft(f_new, len(x))

# =========================================
# FIT λ²
# =========================================
def fit_lambda(lags, cov, tau_min, tau_max):
    idx = (lags >= tau_min) & (lags <= tau_max)

    x = np.log(lags[idx])
    y = cov[idx]

    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    lam2 = -slope
    return lam2, slope, intercept, idx

# =========================================
# MAIN PIPELINE
# =========================================
def run():

    N = 2**14
    scales = [10, 20, 50, 100]
    lam_values = [0.01, 0.02, 0.05, 0.1]
    n_surrogates = 20

    base = "FINAL_RESULTS_KERNEL"
    os.makedirs(base, exist_ok=True)

    logger = Logger(os.path.join(base, "log.txt"))
    logger.log("=== FINAL λ² ESTIMATION WITH KERNEL VISUALIZATION ===")

    for lam2 in lam_values:

        logger.log(f"\n=== λ²_true = {lam2} ===")

        X, omega_true = generate_mrw(N, lam2)

        # ===== omega sanity =====
        lags, cov_true = cov_fft(omega_true, 500)

        plt.figure()
        plt.plot(np.log(lags), cov_true, label="ω true")
        plt.plot(np.log(lags), -lam2*np.log(lags),'--',label="theory")
        plt.legend()
        plt.xlabel("log lag")
        plt.ylabel("Covariance")
        plt.title(f"Omega sanity λ²={lam2}")
        plt.savefig(f"{base}/omega_{lam2}.png")
        plt.close()

        for method in ["ma","gaussian"]:
            for s in scales:

                logger.log(f"\n--- {method}, scale={s} ---")

                # ===== DATA =====
                E = compute_energy(X, s, method)
                omega = compute_omega(E)
                lags, C_data = cov_fft(omega, 500)

                # ===== SURROGATE KERNEL =====
                C_surr_all = []

                for i in range(n_surrogates):
                    if i % 5 == 0:
                        logger.log(f"  surrogate {i}/{n_surrogates}")

                    Xs = phase_surrogate(X)
                    E_s = compute_energy(Xs, s, method)
                    omega_s = compute_omega(E_s)

                    _, C_s = cov_fft(omega_s, 500)
                    C_surr_all.append(C_s)

                C_kernel = np.mean(C_surr_all, axis=0)

                # ===== INTRINSIC =====
                C_intrinsic = C_data - C_kernel

                # ===== SCALE-DEPENDENT FIT RANGE =====
                tau_min = int(3 * s)
                tau_max = int(len(lags) // 5)

                lam_est, slope, intercept, idx = fit_lambda(
                    lags, C_intrinsic, tau_min, tau_max)

                logger.log(f"λ²_est ≈ {lam_est:.4f}")
                logger.log(f"fit range: [{tau_min}, {tau_max}]")

                # ===== PLOT =====
                plt.figure()
                logl = np.log(lags)

                plt.plot(logl, C_data, label="raw")
                plt.plot(logl, C_kernel, label="kernel (surrogate)")
                plt.plot(logl, C_intrinsic, label="intrinsic")

                # fit
                xfit = logl[idx]
                yfit = slope*xfit + intercept
                plt.plot(xfit, yfit,'r',label="fit")

                # theory
                plt.plot(logl, -lam2*logl,'--',label="theory")

                # cutoff
                plt.axvline(np.log(tau_min), color='r',
                            linestyle='--', label='kernel cutoff')

                plt.legend()
                plt.xlabel("log lag")
                plt.ylabel("Covariance")
                plt.title(f"{method}, s={s}, λ²_est={lam_est:.4f}")

                plt.savefig(f"{base}/{method}_s{s}_lam{lam2}.png")
                plt.close()

    logger.log("DONE")
    logger.close()


if __name__ == "__main__":
    run()