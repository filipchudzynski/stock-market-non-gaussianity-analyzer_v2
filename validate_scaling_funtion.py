import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import fftconvolve
    from scipy.stats import linregress

    plt.style.use("seaborn-v0_8")
    return fftconvolve, linregress, np, pd, plt


@app.cell
def _(np):
    def white_noise(n=50000, seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, n)

    def fractional_bm(n=50000, H=0.7, seed=0):
        rng = np.random.default_rng(seed)
        noise = rng.normal(size=n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = x[i - 1] + noise[i] + H * noise[i - 1]
        return x

    #copilot
    def fbm_daviesharte(n=50000, H=0.7, seed=0):
        rng = np.random.default_rng(seed)

        # autocovariance of fGn
        k = np.arange(0, n)
        r = 0.5 * ((k+1)**(2*H) - 2*k**(2*H) + np.abs(k-1)**(2*H))
        r[0] = 1  # fix NaN at k=0

        # embed in circulant matrix
        r_ext = np.concatenate([r, r[-2:0:-1]])
        lam = np.real(np.fft.fft(r_ext))

        if np.any(lam < 0):
            raise ValueError("Covariance is not positive definite")

        # generate fGn
        W = rng.normal(size=2*n-2) + 1j*rng.normal(size=2*n-2)
        fgn = np.fft.ifft(np.sqrt(lam) * W).real[:n]

        # integrate to get fBM
        fbm = np.cumsum(fgn)
        return fbm

    #claude
    def fbm_daviesharte_2(n=50000, H=0.7, seed=0):
        rng = np.random.default_rng(seed)
        k = np.arange(n)
        cov = 0.5 * (np.abs(k+1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k-1)**(2*H))
        row = np.concatenate([cov, cov[-1:0:-1]])  # length 2n-1
        m = len(row)                               # ← derive size here
        eigenvalues = np.real(np.fft.fft(row))
        eigenvalues = np.maximum(eigenvalues, 0)
        z = rng.normal(size=m) + 1j * rng.normal(size=m)  # ← was hardcoded 2*n
        fgn = np.real(np.fft.ifft(np.sqrt(eigenvalues) * z))[:n]
        return np.cumsum(fgn)

    return fbm_daviesharte, fbm_daviesharte_2, white_noise


@app.cell
def _(np):
    def smooth_step(u, u0, eps):
        return 0.5 * (1 + np.tanh((u - u0) / eps))

    def psi_haar_smooth(u, eps=0.02):
        return (
            smooth_step(u, -0.5, eps)
            - 2 * smooth_step(u, 0.0, eps)
            + smooth_step(u, 0.5, eps)
        )

    def psi_mexh(u):
        return (1 - u**2) * np.exp(-0.5 * u**2)

    return psi_haar_smooth, psi_mexh


@app.cell
def _(fftconvolve, np):


    def scale_wavelet(psi, a, dt):
        L = int(10 * a / dt)
        if L < 2:
            L = 2
        u = np.linspace(-L / 2, L / 2, L) / a
        psi_scaled = psi(u) / a
        return u * a, psi_scaled

    def cwt_fft(f, dt, scales, psi):
        W = []
        for a in scales:
            _, psi_a = scale_wavelet(psi, a, dt)
            conv = fftconvolve(f, psi_a[::-1], mode="same") * dt
            W.append(conv)
        return np.array(W)

    return (cwt_fft,)


@app.cell
def _(np):

    dt = 1.0
    scales = np.logspace(np.log10(4), np.log10(2000), 25)
    scales
    return dt, scales


@app.cell
def _(cwt_fft, linregress, np, plt):


    def analyze_signal(x, scales, dt, psi, wavelet_name="", process_name=""):
        from math import log

        W = cwt_fft(x, dt, scales, psi)

        S1 = np.sum(np.abs(W), axis=1)
        S2 = np.sum(W**2, axis=1)

        log_scales = np.log(scales)
        slope1, intercept1, r1, _, _ = linregress(log_scales, np.log(S1))
        slope2, intercept2, r2, _, _ = linregress(log_scales, np.log(S2))

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        ax[0].plot(scales, S1, "o-")
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_title(
            f"{process_name} — {wavelet_name}\nSum |W|, slope={slope1:.3f}, R={r1:.3f}"
        )
        ax[0].set_xlabel("Scale")
        ax[0].set_ylabel("Sum |W|")

        ax[1].plot(scales, S2, "o-")
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        ax[1].set_title(
            f"{process_name} — {wavelet_name}\nSum W², slope={slope2:.3f}, R={r2:.3f}"
        )
        ax[1].set_xlabel("Scale")
        ax[1].set_ylabel("Sum W²")

        plt.tight_layout()
        plt.show()

        return {
            "slope_sum_abs": slope1,
            "R_sum_abs": r1,
            "slope_sum_sq": slope2,
            "R_sum_sq": r2,
        }

    return (analyze_signal,)


@app.cell
def _(
    fbm_daviesharte,
    fbm_daviesharte_2,
    psi_haar_smooth,
    psi_mexh,
    white_noise,
):
    results = {}

    processes = {
        "White noise": white_noise(),
        "FBM H=0.2 (copilot)": fbm_daviesharte(H=0.2),
        "FBM H=0.5 (copilot)": fbm_daviesharte(H=0.5),
        "FBM H=0.8 (copilot)": fbm_daviesharte(H=0.8),
        "FBM H=0.2 (claude)": fbm_daviesharte_2(H=0.2),
        "FBM H=0.5 (claude)": fbm_daviesharte_2(H=0.5),
        "FBM H=0.8 (claude)": fbm_daviesharte_2(H=0.8),
    }

    wavelets = {
        "Mexican hat": psi_mexh,
        "Haar (smooth)": psi_haar_smooth,
    }
    return processes, results, wavelets


@app.cell
def _(cwt_fft, dt, np, plt, processes, scales, wavelets):
    def plot_cwts():
    # processes dict already defined earlier:
    # processes = { "White noise": ..., "FBM H=0.2": ..., ... }
        def mask_edges_with_nan(W, scales, dt):
            """
            Replace invalid edge regions with NaN so imshow can still plot
            a rectangular matrix without ValueErrors.
            """
            n_scales, n_time = W.shape
            W_masked = np.full_like(W, np.nan)

            for i, a in enumerate(scales):
                L = int(5 * a / dt)  # half-support
                if L < n_time // 2:
                    W_masked[i, L:n_time-L] = W[i, L:n_time-L]

            return W_masked

        for pname, x in processes.items():
            fig, ax = plt.subplots(1, 1, figsize=(12, 3))
            ax.plot(x, linewidth=0.7)
            ax.set_title(f"Generated Signal — {pname}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            plt.tight_layout()
            plt.show()

            # Heatmaps for both wavelets
            for wname, psi in wavelets.items():
                W = cwt_fft(x, dt, scales, psi)

                plt.figure(figsize=(12, 5))
                plt.imshow(
                    np.abs(W),
                    extent=[0, len(x), scales[-1], scales[0]],
                    aspect="auto",
                    cmap="viridis"
                )
                plt.colorbar(label="|W(a,t)|")
                plt.title(f"CWT Heatmap — {pname} — {wname}")
                plt.xlabel("Time")
                plt.ylabel("Scale")
                plt.yscale("log")
                plt.tight_layout()
                plt.show()


                W_masked = mask_edges_with_nan(W, scales, dt)

                plt.imshow(
                    np.abs(W_masked),
                    extent=[0, len(x), scales[-1], scales[0]],
                    aspect="auto",
                    cmap="viridis"
                )
                plt.colorbar(label="|W(a,t)|")
                plt.xlabel("Time")
                plt.ylabel("Scale")
                plt.yscale("log")
                plt.title("CWT Heatmap (edge-corrected)")
                plt.show()
    plot_cwts()
    return


@app.cell
def _(analyze_signal, dt, processes, results, scales, wavelets):
    def plot_scaling_functions():
        for pname, x in processes.items():
            for wname, psi in wavelets.items():
                key = f"{pname} — {wname}"
                stats = analyze_signal(
                    x, scales, dt, psi, wavelet_name=wname, process_name=pname
                )
                results[key] = stats

        results
    plot_scaling_functions()
    return


@app.cell
def _(pd, results):

    rows = []
    for key_1, stats_1 in results.items():
        process, wavelet = key_1.split(" — ")
        rows.append(
            {
                "Process": process,
                "Wavelet": wavelet,
                "Slope Σ|W|": stats_1["slope_sum_abs"],
                "R Σ|W|": stats_1["R_sum_abs"],
                "Slope ΣW²": stats_1["slope_sum_sq"],
                "R ΣW²": stats_1["R_sum_sq"],
            }
        )

    df_results = pd.DataFrame(rows)
    df_results
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
