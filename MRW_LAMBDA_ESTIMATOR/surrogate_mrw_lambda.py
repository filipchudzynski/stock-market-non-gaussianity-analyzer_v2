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
# MRW GENERATOR (CORRECT & STABLE)
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
    k = np.ones(s) / s
    return fftconvolve(x, k, mode="same")

def gaussian_filter(x, s):
    L = int(4 * s)
    t = np.arange(-L, L + 1)
    k = np.exp(-t**2 / (2 * s**2))
    k /= k.sum()
    return fftconvolve(x, k, mode="same")

# =========================================
# ENERGY / OMEGA
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

    return E, fluct

def compute_omega(E):
    return 0.5 * np.log(E + 1e-12)

# =========================================
# FAST COVARIANCE (FFT)
# =========================================
def cov_fft(x, maxlag):
    x = x - np.mean(x)
    N = len(x)

    f = np.fft.fft(x, 2*N)
    acf = np.fft.ifft(f * np.conj(f)).real[:N]
    acf /= np.arange(N, 0, -1)

    lags = np.arange(1, maxlag)
    return lags, acf[1:maxlag]

# =========================================
# FIT λ²
# =========================================
def fit_lambda(lags, cov, fit_range):
    idx = (lags >= fit_range[0]) & (lags <= fit_range[1])

    x = np.log(lags[idx])
    y = cov[idx]

    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    lam2 = -slope
    return lam2, slope, intercept, idx

# =========================================
# KURTOSIS
# =========================================
def kurtosis(fluct):
    m2 = np.mean(fluct**2)
    m4 = np.mean(fluct**4)
    return m4 / (m2**2 + 1e-12)

# =========================================
# MAIN PIPELINE
# =========================================
def run():

    N = 2**14
    scales = [10, 20, 50, 100]
    lam_values = [0.01, 0.02, 0.05, 0.1]
    fit_range = (5, 100)

    base = "FINAL_RESULTS"
    os.makedirs(base, exist_ok=True)

    logger = Logger(os.path.join(base, "log.txt"))
    logger.log("=== FINAL MULTISCALE INTERMITTENCY PIPELINE ===")

    bias = {s: [] for s in scales}

    for lam2 in lam_values:

        logger.log(f"\n=== λ²_true = {lam2} ===")

        X, omega_true = generate_mrw(N, lam2)

        # ===== SANITY CHECK =====
        lags, cov_true = cov_fft(omega_true, 500)

        plt.figure()
        plt.plot(np.log(lags), cov_true, label="ω true")
        plt.plot(np.log(lags), -lam2*np.log(lags), '--', label="theory")
        plt.legend()
        plt.xlabel("log lag")
        plt.ylabel("Covariance")
        plt.title(f"Omega sanity λ²={lam2}")
        plt.savefig(f"{base}/omega_{lam2}.png")
        plt.close()

        for method in ["ma", "gaussian"]:
            for s in scales:

                logger.log(f"{method} scale={s}")

                E, fluct = compute_energy(X, s, method)
                omega_est = compute_omega(E)

                lags, cov = cov_fft(omega_est, 500)

                lam_est, slope, intercept, idx = fit_lambda(
                    lags, cov, fit_range)

                logger.log(f"λ²_est ≈ {lam_est:.4f}")

                bias[s].append((lam2, lam_est))

                # ===== PLOT =====
                plt.figure()
                logl = np.log(lags)

                plt.plot(logl, cov, label="estimate")

                xfit = logl[idx]
                yfit = slope * xfit + intercept
                plt.plot(xfit, yfit, 'r', label="fit")

                plt.plot(logl, -lam2*logl, '--', label="theory")

                plt.axvspan(np.log(fit_range[0]),
                            np.log(fit_range[1]),
                            alpha=0.2)

                plt.legend()
                plt.xlabel("log lag")
                plt.ylabel("Covariance")
                plt.title(f"{method}, scale={s}, λ²_est={lam_est:.4f}")
                plt.savefig(f"{base}/{method}_s{s}_lam{lam2}.png")
                plt.close()

                # ===== KURTOSIS =====
                K = kurtosis(fluct)
                logger.log(f"kurtosis ≈ {K:.3f}")

    # =========================================
    # BIAS CORRECTION
    # =========================================
    logger.log("\n=== BIAS CORRECTION ===")

    correction = {}

    for s in scales:
        data = np.array(bias[s])
        true = data[:,0]
        est  = data[:,1]

        a = np.sum(true * est) / np.sum(est**2)
        correction[s] = a

        logger.log(f"scale {s}: correction a = {a:.3f}")

        plt.figure()
        plt.scatter(est, true, label="data")

        x = np.linspace(0, max(est)*1.2, 100)
        plt.plot(x, a*x, 'r', label=f"fit")

        plt.xlabel("λ²_est")
        plt.ylabel("λ²_true")
        plt.title(f"Bias calibration scale={s}")
        plt.legend()
        plt.savefig(f"{base}/bias_scale_{s}.png")
        plt.close()

    # =========================================
    # SCALE COLLAPSE
    # =========================================
    logger.log("\n=== SCALE COLLAPSE ===")

    for method in ["ma", "gaussian"]:

        plt.figure()

        for lam2 in lam_values:

            X, _ = generate_mrw(N, lam2)

            for s in scales:

                E, _ = compute_energy(X, s, method)
                omega_est = compute_omega(E)

                lags, cov = cov_fft(omega_est, 300)

                cov_corr = cov * correction[s]
                scaled = cov_corr / lam2

                plt.plot(np.log(lags), scaled,
                         label=f"s={s}, λ²={lam2}")

        plt.plot(np.log(lags), -np.log(lags),
                 'k--', label="theory")

        plt.xlabel("log lag")
        plt.ylabel("Cov / λ²")
        plt.title(f"Scale collapse ({method})")
        plt.legend(fontsize=8)
        plt.savefig(f"{base}/collapse_{method}.png")
        plt.close()

    logger.log("DONE")
    logger.close()

if __name__ == "__main__":
    run()