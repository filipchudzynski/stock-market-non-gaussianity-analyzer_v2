"""
Wavelet Diagnostic: Haar → Gaussian convergence study
Reproducing Arnéodo, Muzy & Sornette (1998) — Figures 1–3

Diagnostic goal: understand why DWT/Haar reproduces the paper's results
but CWT does not, by interpolating between Haar and Gaussian via iterated
self-convolutions of the Haar filter (B-spline wavelets of increasing order).

Run with:
    marimo edit wavelet_diagnostic.py
or:
    marimo run wavelet_diagnostic.py
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full", app_title="Wavelet Diagnostic: Haar → Gaussian")


@app.cell
def _():
    # ---------------------------------------------------------
    # Continuous wavelet transform using numerical integration
    # W_f(a, t) = (1/a) * ∫ f(y) psi((y - t)/a) dy
    # Numerically: W_f(a, t_i) ≈ (1/a) * Σ_j f(t_j) psi((t_j - t_i)/a) * dt
    # ---------------------------------------------------------

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import fftconvolve
    from ssqueezepy import cwt


    return cwt, fftconvolve, np, plt


@app.cell
def _(np, plt):

    def cwt_basic_implementation():

        # ---------------------------------------------------------
        # Parameters and example signal
        # ---------------------------------------------------------

        # Time step: 2 minutes in arbitrary units (e.g., dt = 2)
        dt = 2.0

        # Example time axis and signal f(t)
        T = 400.0  # total duration in same units as dt
        t = np.arange(0, T, dt)
        # Example signal: sum of two sinusoids + noise
        f = np.sin(2 * np.pi * t / 80.0) + 0.5 * np.sin(2 * np.pi * t / 40.0) + 0.2 * np.random.randn(len(t))

        # ---------------------------------------------------------
        # Haar analysing wavelet (mother wavelet)
        # Standard definition on [0,1): +1 on [0,0.5), -1 on [0.5,1), 0 otherwise
        # ---------------------------------------------------------

        def haar_psi(u):
            """
            Haar mother wavelet psi(u).
            u can be a scalar or numpy array.
            """
            u = np.asarray(u)
            psi_vals = np.zeros_like(u, dtype=float)
            psi_vals[(u >= 0.0) & (u < 0.5)] = 1.0
            psi_vals[(u >= 0.5) & (u < 1.0)] = -1.0
            return psi_vals

        # ---------------------------------------------------------
        # Continuous wavelet transform using numerical integration
        # W_f(a, t) = (1/a) * ∫ f(y) psi((y - t)/a) dy
        # Numerically: W_f(a, t_i) ≈ (1/a) * Σ_j f(t_j) psi((t_j - t_i)/a) * dt
        # ---------------------------------------------------------

        def cwt_haar(f, t, scales):
            """
            Compute CWT of signal f(t) using Haar wavelet and numerical integration.

            Parameters
            ----------
            f : array_like
                Signal values at times t.
            t : array_like
                Time samples (uniformly spaced).
            scales : array_like
                Array of scales 'a' at which to compute the transform.

            Returns
            -------
            W : 2D ndarray
                Wavelet coefficients with shape (len(scales), len(t)).
                W[k, i] = W_f(a_k, t_i)
            """
            f = np.asarray(f)
            t = np.asarray(t)
            dt = t[1] - t[0]
            n = len(t)
            scales = np.asarray(scales)

            W = np.zeros((len(scales), n))

            # For each scale a, compute W_f(a, t_i) for all t_i
            for k, a in enumerate(scales):
                # For each time t_i, we need psi((t_j - t_i)/a) for all j
                # We can vectorize this by building a matrix of (t_j - t_i)/a
                # t_j: row, t_i: column
                tj = t.reshape(-1, 1)          # shape (n, 1)
                ti = t.reshape(1, -1)          # shape (1, n)
                u = (tj - ti) / a              # shape (n, n)
                psi_vals = haar_psi(u)         # shape (n, n)

                # Numerical integration: (1/a) * Σ_j f(t_j) psi((t_j - t_i)/a) * dt
                # f(t_j) is shape (n,), so broadcast to (n, 1)
                integrand = f.reshape(-1, 1) * psi_vals  # shape (n, n)
                W[k, :] = (1.0 / a) * np.sum(integrand, axis=0) * dt

            return W

        # ---------------------------------------------------------
        # Example usage
        # ---------------------------------------------------------

        # Choose a set of scales (in same units as t)
        # For Haar, scales roughly correspond to window lengths.
        scales = np.linspace(4, 80, 30)  # adjust as needed

        W = cwt_haar(f, t, scales)

        # ---------------------------------------------------------
        # Plot result
        # ---------------------------------------------------------

        plt.figure(figsize=(10, 6))

        # Signal
        plt.subplot(2, 1, 1)
        plt.plot(t, f, color='k')
        plt.title("Signal f(t)")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Amplitude")

        # Scalogram (magnitude of wavelet coefficients)
        plt.subplot(2, 1, 2)
        plt.imshow(
            np.abs(W),
            extent=[t[0], t[-1], scales[-1], scales[0]],
            aspect='auto',
            cmap='viridis'
        )
        plt.colorbar(label="|W_f(a, t)|")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Scale a")
        plt.title("Haar CWT (numerical integral)")

        plt.tight_layout()
        plt.show()
    cwt_basic_implementation()
    return


@app.cell
def _(fftconvolve, np, plt):
    def cwt_fft_implementation():

        # ---------------------------------------------------------
        # Parameters and example signal
        # ---------------------------------------------------------

        dt = 2.0
        T = 400.0
        t = np.arange(0, T, dt)

        f = (
            np.sin(2 * np.pi * t / 80.0)
            + 0.5 * np.sin(2 * np.pi * t / 40.0)
            + 0.2 * np.random.randn(len(t))
        )

        # ---------------------------------------------------------
        # Haar wavelet scaled for FFT convolution
        # ---------------------------------------------------------

        def haar_wavelet_scaled(a, dt):
            """
            Construct discrete Haar wavelet at scale a.
            """
            N = max(int(a / dt), 2)
            half = N // 2

            psi = np.zeros(N)
            psi[:half] = 1.0
            psi[half:] = -1.0

            return psi / a   # normalization

        # ---------------------------------------------------------
        # FFT‑based Haar CWT
        # ---------------------------------------------------------

        def cwt_haar_fft(f, dt, scales):
            W = []
            for a in scales:
                psi = haar_wavelet_scaled(a, dt)

                # correlation = convolution with reversed kernel
                conv = fftconvolve(f, psi[::-1], mode='same') * dt

                W.append(conv)
            return np.array(W)

        # ---------------------------------------------------------
        # Example usage
        # ---------------------------------------------------------

        scales = np.linspace(4, 80, 30)
        W = cwt_haar_fft(f, dt, scales)

        # ---------------------------------------------------------
        # Plot result
        # ---------------------------------------------------------

        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(t, f, color='k')
        plt.title("Signal f(t)")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        plt.imshow(
            np.abs(W),
            extent=[t[0], t[-1], scales[-1], scales[0]],
            aspect='auto',
            cmap='viridis'
        )
        plt.colorbar(label="|W_f(a, t)|")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Scale a")
        plt.title("Haar CWT (FFT‑based convolution)")

        plt.tight_layout()
        plt.show()
    cwt_fft_implementation()
    return


@app.cell
def _(cwt, fftconvolve, np, plt):
    def cwt_pywt_implementation():
        import pywt

    # ---------------------------------------------------------
    # Custom continuous Haar wavelet for PyWavelets
    # ---------------------------------------------------------

        # class HaarCWT(pywt.ContinuousWavelet):
        #     def __init__(self):
        #         super().__init__("custom_haar")

        #         # Required attributes
        #         self.support_width = 1.0
        #         self.complex_cwt = False
        #         self.center_frequency = 1.0   # arbitrary but required

        #     def wavefun(self, level=10):
        #         """
        #         PyWavelets requires (phi, psi, x).
        #         For CWT, phi is unused.
        #         """
        #         x = np.linspace(0, 1, 2048)
        #         psi = np.zeros_like(x)

        #         psi[(x >= 0) & (x < 0.5)] = 1.0
        #         psi[(x >= 0.5) & (x < 1.0)] = -1.0

        #         phi = np.zeros_like(x)
        #         return phi, psi, x

    # ---------------------------------------------------------
    # Smooth Haar wavelet (time domain)
    # ---------------------------------------------------------

        def smooth_step(t, t0, eps):
            return 0.5 * (1 + np.tanh((t - t0) / eps))
    
        def smooth_haar_time(t, eps=0.01):
            return smooth_step(t, 0.0, eps) - 2*smooth_step(t, 0.5, eps) + smooth_step(t, 1.0, eps)


        # ---------------------------------------------------------
        # Fourier transform of smooth Haar
        # ---------------------------------------------------------

        def smooth_haar_fourier(xi, eps=0.01):
            # Time grid for FFT
            t = np.linspace(-1, 2, 4096)   # wide support to avoid wrap-around
            psi = smooth_haar_time(t, eps)
            dt = t[1] - t[0]
    
            # FFT
            Psi = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi)))
            freqs = np.fft.fftshift(np.fft.fftfreq(len(t), d=dt))
    
            # Interpolate onto requested frequencies
            return np.interp(xi, freqs, Psi)


        # ---------------------------------------------------------
        # Haar CWT using ssqueezepy
        # ---------------------------------------------------------

        def cwt_ssq_haar(f, scales):
            W, _ = cwt(f, wavelet=smooth_haar_fourier, scales=scales)
            return W

        # ---------------------------------------------------------
        # FFT‑based Haar CWT
        # ---------------------------------------------------------

        def haar_wavelet_scaled(a, dt):
            N = max(int(a / dt), 2)
            half = N // 2
            psi = np.zeros(N)
            psi[:half] = 1.0
            psi[half:] = -1.0
            return psi / a

        def cwt_haar_fft(f, dt, scales):
            W = []
            for a in scales:
                psi = haar_wavelet_scaled(a, dt)
                conv = fftconvolve(f, psi[::-1], mode='same') * dt
                W.append(conv)
            return np.array(W)

        # ---------------------------------------------------------
        # Direct numerical Haar CWT (your original)
        # ---------------------------------------------------------

        def haar_psi(u):
            u = np.asarray(u)
            psi_vals = np.zeros_like(u)
            psi_vals[(u >= 0) & (u < 0.5)] = 1.0
            psi_vals[(u >= 0.5) & (u < 1.0)] = -1.0
            return psi_vals

        def cwt_haar_direct(f, t, scales):
            dt = t[1] - t[0]
            n = len(t)
            W = np.zeros((len(scales), n))

            for k, a in enumerate(scales):
                tj = t.reshape(-1, 1)
                ti = t.reshape(1, -1)
                u = (tj - ti) / a
                psi_vals = haar_psi(u)
                integrand = f.reshape(-1, 1) * psi_vals
                W[k, :] = (1.0 / a) * np.sum(integrand, axis=0) * dt

            return W

    # ---------------------------------------------------------
    # Full comparison function
    # ---------------------------------------------------------


        dt = 2.0
        T = 400.0
        t = np.arange(0, T, dt)

        f = (
            np.sin(2 * np.pi * t / 80.0)
            + 0.5 * np.sin(2 * np.pi * t / 40.0)
            + 0.2 * np.random.randn(len(t))
        )

        scales = np.linspace(4, 80, 30)

        # Compute all three transforms
        W_direct = cwt_haar_direct(f, t, scales)
        W_fft = cwt_haar_fft(f, dt, scales)
        W_pywt = cwt_ssq_haar(f, scales)

        # Plot
        plt.figure(figsize=(12, 12))

        plt.subplot(4, 1, 1)
        plt.plot(t, f, color='k')
        plt.title("Signal f(t)")

        plt.subplot(4, 1, 2)
        plt.imshow(np.abs(W_direct),
                   extent=[t[0], t[-1], scales[-1], scales[0]],
                   aspect='auto', cmap='viridis')
        plt.title("Haar CWT — Direct Numerical Integration")

        plt.subplot(4, 1, 3)
        plt.imshow(np.abs(W_fft),
                   extent=[t[0], t[-1], scales[-1], scales[0]],
                   aspect='auto', cmap='viridis')
        plt.title("Haar CWT — FFT Convolution")

        plt.subplot(4, 1, 4)
        plt.imshow(np.abs(W_pywt),
                   extent=[t[0], t[-1], scales[-1], scales[0]],
                   aspect='auto', cmap='viridis')
        plt.title("Haar CWT — PyWavelets (Custom Continuous Haar)")

        plt.tight_layout()
        plt.show()


    cwt_pywt_implementation()
    return


@app.cell
def _(np, plt):
    def test_smooth_haar():
    
        # def smooth_haar_time(t, eps=0.01):
        #     return np.tanh((0.25 - t)/eps) - np.tanh((0.75 - t)/eps)

        def smooth_step(t, t0, eps):
            return 0.5 * (1 + np.tanh((t - t0) / eps))
    
        def smooth_haar_time(t, eps=0.01):
            return smooth_step(t, 0.0, eps) - 2*smooth_step(t, 0.5, eps) + smooth_step(t, 1.0, eps)

    
        t=np.linspace(-1,1,200)
        haar=smooth_haar_time(t)
        plt.plot(t,haar)
        plt.show()
    test_smooth_haar()
    return


@app.cell
def _(np, plt):

    t = np.linspace(-5, 15, 2000)   # time axis for visualization
    # Example signal: sum of two sinusoids + noise
    f = np.sin(2 * np.pi * t / 80.0) + 0.5 * np.sin(2 * np.pi * t / 40.0) + 0.2 * np.random.randn(len(t))



    def haar_psi(u):
        u = np.asarray(u)
        psi_vals = np.zeros_like(u, dtype=float)
        psi_vals[(u >= 0.0) & (u < 0.5)] = 1.0
        psi_vals[(u >= 0.5) & (u < 1.0)] = -1.0
        return psi_vals

    def haar_wavelet_scaled(a, dt):
        """
        Construct discrete Haar wavelet at scale a.
        dt = sampling interval.
        """
        N = int(a / dt)
        if N < 2:
            N = 2  # minimum length

        half = N // 2

        psi = np.zeros(N)
        psi[:half] = 1.0
        psi[half:] = -1.0

        # normalization 1/a
        psi = psi / a

        return psi

    # ---------------------------------------------------------
    # Visualize Haar wavelet at different scales and shifts
    # ---------------------------------------------------------
    dt=1

    scales = np.linspace(1,1000,20)              # different scales a

    plt.figure(figsize=(12, 8))

    plot_index = 1
    for i, a in enumerate(scales, 1):
        psi = haar_wavelet_scaled(a, dt)
        tt = np.arange(len(psi)) * dt

        plt.plot(tt, psi, color='black')
        plt.title(f"Haar wavelet at scale a={a}")
        plt.xlabel("time")
        plt.ylabel("amplitude")
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    return dt, f, haar_wavelet_scaled, scales, t


@app.cell
def _(dt, f, fftconvolve, haar_wavelet_scaled, np, plt, scales, t):
    def cwt_haar_fft(f, dt, scales):
        """
        Compute Haar CWT using FFT convolution.
        f: signal
        dt: sampling interval
        scales: list of scales a
        """
        W = []

        for a in scales:
            psi = haar_wavelet_scaled(a, dt)

            # reverse psi for correlation
            conv = fftconvolve(f, psi[::-1], mode='same') * dt

            W.append(conv)

        return np.array(W)

    # ---------------------------------------------------------
    # Example usage
    # ---------------------------------------------------------

    # Choose a set of scales (in same units as t)
    # For Haar, scales roughly correspond to window lengths.

    W = cwt_haar_fft(f, dt, scales)

    # ---------------------------------------------------------
    # Plot result
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))

    # Signal
    plt.subplot(2, 1, 1)
    plt.plot(t, f, color='k')
    plt.title("Signal f(t)")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Amplitude")

    # Scalogram (magnitude of wavelet coefficients)
    plt.subplot(2, 1, 2)
    plt.imshow(
        np.abs(W),
        extent=[t[0], t[-1], scales[-1], scales[0]],
        aspect='auto',
        cmap='viridis'
    )
    plt.colorbar(label="|W_f(a, t)|")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Scale a")
    plt.title("Haar CWT (numerical integral)")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(fftconvolve, np, plt):
    def compare_direct_and_convolution():
        # ---------------------------------------------------------
        # Haar mother wavelet
        # ---------------------------------------------------------

        def haar_psi(u):
            u = np.asarray(u)
            psi = np.zeros_like(u)
            psi[(u >= 0) & (u < 0.5)] = 1
            psi[(u >= 0.5) & (u < 1)] = -1
            return psi

        # ---------------------------------------------------------
        # Direct numerical CWT (slow, O(N^2))
        # ---------------------------------------------------------

        def cwt_haar_direct(f, t, scales):
            dt = t[1] - t[0]
            n = len(t)
            W = np.zeros((len(scales), n))

            for k, a in enumerate(scales):
                for i, ti in enumerate(t):
                    u = (t - ti) / a
                    psi_vals = haar_psi(u)
                    W[k, i] = (1/a) * np.sum(f * psi_vals) * dt

            return W

        # ---------------------------------------------------------
        # FFT‑based Haar wavelet (fast, O(N log N))
        # ---------------------------------------------------------

        def haar_wavelet_scaled(a, dt):
            N = max(int(a / dt), 2)
            half = N // 2
            psi = np.zeros(N)
            psi[:half] = 1
            psi[half:] = -1
            return psi / a

        def cwt_haar_fft(f, dt, scales):
            W = []
            for a in scales:
                psi = haar_wavelet_scaled(a, dt)
                conv = fftconvolve(f, psi[::-1], mode='same') * dt
                W.append(conv)
            return np.array(W)

        # ---------------------------------------------------------
        # Example signal
        # ---------------------------------------------------------

        dt = 2.0
        t = np.arange(0, 400, dt)
        f = np.sin(2*np.pi*t/80) + 0.5*np.sin(2*np.pi*t/40)

        # ---------------------------------------------------------
        # Compute both transforms
        # ---------------------------------------------------------

        scales = [8, 16, 32, 64]

        W_direct = cwt_haar_direct(f, t, scales)
        W_fft = cwt_haar_fft(f, dt, scales)

        # ---------------------------------------------------------
        # Compare results
        # ---------------------------------------------------------

        plt.figure(figsize=(12, 10))

        for i, a in enumerate(scales):
            plt.subplot(len(scales), 2, 2*i+1)
            plt.plot(t, W_direct[i], label="Direct")
            plt.title(f"Direct CWT (a={a})")
            plt.grid(True)

            plt.subplot(len(scales), 2, 2*i+2)
            plt.plot(t, W_fft[i], label="FFT", color='orange')
            plt.title(f"FFT CWT (a={a})")
            plt.grid(True)

        plt.tight_layout()
        plt.show()

        # ---------------------------------------------------------
        # Numerical error check
        # ---------------------------------------------------------

        error = np.linalg.norm(W_direct - W_fft) / np.linalg.norm(W_direct)
        print("Relative error between methods:", error)
    compare_direct_and_convolution()
    return


@app.cell
def _imports():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import pywt
    from scipy.signal import fftconvolve, correlate
    from scipy.stats import pearsonr
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import functools
    import marimo as mo

    return correlate, fftconvolve, mo, np, pd, pearsonr, plt, pywt


@app.cell
def _controls(mo):
    data_path_ui = mo.ui.text(
        value="SandP_log2min.dat",
        label="Data file path",
        full_width=True,
    )
    n_conv_slider = mo.ui.slider(
        start=1, stop=12, step=1, value=4,
        label="Wavelet order n  (1 = Haar, 12 ≈ Gaussian)",
        show_value=True,
    )
    max_scale_slider = mo.ui.slider(
        start=4, stop=12, step=1, value=10,
        label="Max DWT level (2^level time steps)",
        show_value=True,
    )
    max_lag_slider = mo.ui.slider(
        start=100, stop=5000, step=100, value=2000,
        label="Max autocorrelation lag",
        show_value=True,
    )
    mi_max_lag_slider = mo.ui.slider(
        start=20, stop=300, step=10, value=100,
        label="MI time lag range ± steps",
        show_value=True,
    )
    cwt_norm_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.5,
        label="CWT normalisation exponent  (0.5 = energy, 1.0 = amplitude)",
        show_value=True,
    )
    overlay_orders_ui = mo.ui.multiselect(
        options=list(range(1, 13)),
        value=[1, 2, 4, 8],
        label="Wavelet orders to overlay in Fig-2 & MI plots",
    )
    scale_idx_ref_ui = mo.ui.slider(
        start=0, stop=9, step=1, value=3,
        label="Reference scale index (for MI cross-scale)",
        show_value=True,
    )

    controls = mo.vstack([
        mo.md("## ⚙️ Parameters"),
        data_path_ui,
        mo.hstack([n_conv_slider, max_scale_slider]),
        mo.hstack([max_lag_slider, mi_max_lag_slider]),
        mo.hstack([cwt_norm_slider, scale_idx_ref_ui]),
        overlay_orders_ui,
    ])
    controls
    return (
        cwt_norm_slider,
        data_path_ui,
        max_lag_slider,
        max_scale_slider,
        mi_max_lag_slider,
        n_conv_slider,
        overlay_orders_ui,
        scale_idx_ref_ui,
    )


@app.cell
def _load_data(data_path_ui, mo, np, pd):
    _path = data_path_ui.value
    mo.stop(not _path, mo.md("⚠️ Please enter a data file path above."))

    try:
        _df = pd.read_csv(_path, header=None, names=["price"])
        _df = _df.dropna().reset_index(drop=True)
        _df["log_price"] = np.log(_df["price"])
        log_price = _df["log_price"].values.copy()
        data_ok = True
        data_info = mo.callout(
            mo.md(f"**Loaded** `{_path}` — {len(log_price):,} samples"),
            kind="success",
        )
    except FileNotFoundError:
        log_price = np.array([])
        data_ok = False
        data_info = mo.callout(
            mo.md(
                f"⚠️ File `{_path}` not found. "
                "Generating synthetic log-price for demonstration."
            ),
            kind="warn",
        )
        # Synthetic: multifractal random walk proxy (GARCH-like)
        _rng = np.random.default_rng(42)
        _n = 50_000
        _vol = np.ones(_n)
        _ret = np.zeros(_n)
        for _i in range(1, _n):
            _vol[_i] = 0.99 * _vol[_i - 1] + 0.1 * abs(_ret[_i - 1])
            _ret[_i] = _vol[_i] * _rng.standard_normal()
        log_price = np.cumsum(_ret)
        data_ok = True

    data_info
    return data_ok, log_price


@app.cell
def _(pywt):
    discrete_wavelet_types = list(filter(lambda x: not 'bio' in x,pywt.wavelist(kind='discrete')))
    print(discrete_wavelet_types)
    return (discrete_wavelet_types,)


@app.cell
def _(pywt):
    import math
    c = math.sqrt(2)/2
    dec_lo, dec_hi, rec_lo, rec_hi = [c, c], [-c, c], [c, c], [c, -c]
    filter_bank = [dec_lo, dec_hi, rec_lo, rec_hi]
    myWavelet = pywt.Wavelet(name="myHaarWavelet", filter_bank=filter_bank)

    class HaarFilterBank(object):
         @property
         def filter_bank(self):
             c = math.sqrt(2)/2
             dec_lo, dec_hi, rec_lo, rec_hi = [c, c], [-c, c], [c, c], [c, -c]
             return [dec_lo, dec_hi, rec_lo, rec_hi]
    filter_bank = HaarFilterBank()
    myOtherWavelet = pywt.Wavelet(name="myHaarWavelet", filter_bank=filter_bank)
    return (myOtherWavelet,)


@app.cell
def _(myOtherWavelet):
    myOtherWavelet.wavefun(level=1)
    return


@app.cell
def _(pywt):
    pywt.Wavelet(name='haar').filter_bank()
    return


@app.cell
def _(discrete_wavelet_types, mo):
    wavelet_type_to_plot = mo.ui.dropdown(
        options=discrete_wavelet_types,
        label="Wavelet to plot(note: only orthogonal)",
    )
    wavelet_type_to_plot
    return (wavelet_type_to_plot,)


@app.cell
def _(plt, pywt, wavelet_type_to_plot):
    def plot_wavelet(type='haar',maxlevel=10):
        wavelet = pywt.Wavelet(type)
        plt.figure(figsize=(8, 4))

        for i in range(1,maxlevel):
            phi,psi,x = wavelet.wavefun(level=i)  # scaling function, wavelet function, grid

            plt.plot(x, psi, label=f'{type} wavelet ψ(x) level={i}')
            # plt.plot(x, phi, label='Scaling function φ(x)', linestyle='--')
        plt.title(f"{type} Wavelet and Scaling Function")
        plt.legend()
        plt.grid(True)
        plt.show()

    plot_wavelet(type=wavelet_type_to_plot.value)
    return


@app.cell
def _kernel_utils(correlate, fftconvolve, np, pearsonr, pywt):
    """
    Build the iterated Haar kernel family and the à trous filtering pipeline.
    All functions are pure (no side-effects) so Marimo can cache them.
    """

    def build_haar_kernel(n: int,level=1) -> np.ndarray:
        """
        Return the n-th iterated convolution of the Haar detail filter h=[1,-1].
        n=1 → Haar; n→∞ → DOG (difference of Gaussians).
        Normalised to unit L2 norm.
        """
        wavelet = pywt.Wavelet(name='haar')
        phi,h,x = wavelet.wavefun(level)
        kernel = h.copy()
        for _ in range(n - 1):
            kernel = fftconvolve(kernel, h)
        kernel /= np.linalg.norm(kernel)
        return kernel

    def dilate_kernel(kernel: np.ndarray, scale: int) -> np.ndarray:
        """
        Insert (scale-1) zeros between each sample → dilation by `scale`.
        This is the à trous (holes) trick: no downsampling needed.
        """
        if scale == 1:
            return kernel
        dilated = np.zeros(len(kernel) + (len(kernel) - 1) * (scale - 1))
        dilated[::scale] = kernel
        return dilated

    def atrous_filter(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Apply a single FIR kernel to the signal via FFT convolution.
        Output is trimmed to the same length as the input (centered).
        """
        pad = len(kernel) // 2
        out = fftconvolve(signal, kernel, mode="full")
        return out[pad: pad + len(signal)]

    def build_detail_series(
        log_price: np.ndarray,
        n_conv: int,
        max_level: int,
    ) -> tuple[np.ndarray, list[int]]:
        """
        Build detail_series[scale_idx, time] using the à trous pipeline.

        Returns:
            detail_series : shape (max_level, N)
            scales        : list of scale values in samples
        """
        base_kernel = build_haar_kernel(n_conv)
        N = len(log_price)
        scales = [2 ** j for j in range(1, max_level + 1)]
        detail_series = np.zeros((max_level, N))
        for j, scale in enumerate(scales):
            dk = dilate_kernel(base_kernel, scale)
            detail_series[j] = atrous_filter(log_price, dk)
        return detail_series, scales

    def log_vol_from_detail(
        detail_series: np.ndarray,
        window: int = 50,
    ) -> np.ndarray:
        """
        Local volatility = rolling RMS of the detail series.
        Returns log_vol_series of the same shape.
        """
        import pandas as pd_local
        n_scales, N = detail_series.shape
        log_vol = np.zeros_like(detail_series)
        for j in range(n_scales):
            sq = detail_series[j] ** 2
            vol = np.sqrt(
                pd_local.Series(sq).rolling(window, min_periods=1).mean().values
            )
            log_vol[j] = np.log(vol + 1e-12)
        return log_vol

    def corr_function(x: np.ndarray, max_lag: int) -> np.ndarray:
        """
        Normalised autocorrelation via FFT for lags 0..max_lag.
        Uses unbiased normalization (divides by n - lag).
        """
        x = np.asarray(x, float)
        x = x - x.mean()
        n = len(x)
        c = fftconvolve(x, x[::-1], mode="full")
        c = c[c.size // 2:]          # keep lags 0, 1, 2, ...
        norm = np.arange(n, 0, -1)
        c = c / norm
        c = c / c[0]
        return c[: max_lag + 1]

    def gaussian_mi(x: np.ndarray, y: np.ndarray) -> float:
        """
        Mutual information under Gaussian assumption:
            I = -0.5 * log(1 - rho^2)
        Returns 0.0 if the series are too short or degenerate.
        """
        if len(x) < 10:
            return 0.0
        rho, _ = pearsonr(x, y)
        rho = np.clip(rho, -1 + 1e-9, 1 - 1e-9)
        return -0.5 * np.log(1 - rho ** 2)

    def compute_mi_map(
        log_vol_series: np.ndarray,
        ref_idx: int,
        max_time_lag: int,
    ) -> np.ndarray:
        """
        Compute MI map[scale_idx, lag_idx] using vectorised cross-correlation.

        For the Gaussian MI approximation, MI(x,y) only needs rho = E[xy]/(std_x*std_y).
        We compute the full cross-correlation vector via scipy.signal.correlate,
        then map to MI pointwise — O(N log N) rather than O(N * n_lags).
        """
        n_scales, N = log_vol_series.shape
        n_lags = 2 * max_time_lag + 1
        I_map = np.zeros((n_scales, n_lags))

        w_ref = log_vol_series[ref_idx]
        w_ref_c = w_ref - w_ref.mean()
        std_ref = w_ref_c.std() + 1e-12

        for j in range(n_scales):
            w = log_vol_series[j]
            w_c = w - w.mean()
            std_w = w_c.std() + 1e-12

            # Full cross-correlation: shape (2N-1,)
            cc = correlate(w_c, w_ref_c, mode="full")
            # Normalise by overlap length and stds to get rho(dt)
            lags_full = np.arange(-(N - 1), N)
            overlap = N - np.abs(lags_full)
            overlap = np.maximum(overlap, 1)
            rho_full = cc / (overlap * std_w * std_ref)

            # Extract the window we care about
            center = N - 1  # index of lag=0 in correlate output
            rho_window = rho_full[center - max_time_lag: center + max_time_lag + 1]
            rho_window = np.clip(rho_window, -1 + 1e-9, 1 - 1e-9)
            I_map[j] = -0.5 * np.log(1 - rho_window ** 2)

        return I_map

    return (
        build_detail_series,
        build_haar_kernel,
        compute_mi_map,
        corr_function,
        log_vol_from_detail,
    )


@app.cell
def _phase1_kernel_zoo(build_haar_kernel, mo, n_conv_slider, np, plt):
    mo.stop(build_haar_kernel is None)

    _n_max = n_conv_slider.value

    fig_kernels, axes_k = plt.subplots(
        2, _n_max, figsize=(max(12, 3 * _n_max), 5), squeeze=False
    )
    fig_kernels.suptitle(
        "Phase 1 — Kernel zoo: Haar iterated convolutions (time & frequency)",
        fontsize=13,
    )

    _cmap = plt.cm.viridis(np.linspace(0.15, 0.9, _n_max))

    for _i, _n in enumerate(range(1, _n_max + 1)):
        _k = build_haar_kernel(_n,level=2)
        _t = np.arange(len(_k))

        # Time domain
        _ax_t = axes_k[0, _i]
        _ax_t.plot(_t, _k, color=_cmap[_i], linewidth=1.5)
        _ax_t.axhline(0, color="gray", linewidth=0.5)
        _ax_t.set_title(f"n={_n}", fontsize=9)
        _ax_t.set_xticks([])
        if _i == 0:
            _ax_t.set_ylabel("Amplitude")

        # Frequency domain (|FFT|, one-sided)
        _ax_f = axes_k[1, _i]
        _N_fft = max(512, 2 * len(_k))
        _K_fft = np.abs(np.fft.rfft(_k, n=_N_fft))
        _freqs = np.fft.rfftfreq(_N_fft)
        _ax_f.plot(_freqs, _K_fft, color=_cmap[_i], linewidth=1.5)
        _ax_f.set_xlim(0, 0.5)
        if _i == 0:
            _ax_f.set_ylabel("|FFT|")
        _ax_f.set_xlabel("f", fontsize=8)

    plt.tight_layout()
    plt.show()
    mo.md("### Phase 1 — Kernel zoo")
    return


@app.cell
def _compute_pipeline(
    build_detail_series,
    data_ok,
    log_price,
    log_vol_from_detail,
    max_scale_slider,
    mo,
    n_conv_slider,
):
    mo.stop(not data_ok or len(log_price) == 0)

    _n = n_conv_slider.value
    _max_level = max_scale_slider.value

    with mo.status.spinner(title=f"Running à trous pipeline (n={_n}, levels={_max_level})…"):
        detail_series, scales = build_detail_series(log_price, _n, _max_level)
        log_vol_series = log_vol_from_detail(detail_series, window=50)

    pipeline_info = mo.callout(
        mo.md(
            f"à trous pipeline ready — "
            f"n={_n}, levels={_max_level}, "
            f"detail shape={detail_series.shape}"
        ),
        kind="info",
    )
    pipeline_info
    return detail_series, log_vol_series, scales


@app.cell
def _(mo):
    sanity_check_start_of_data_slider = mo.ui.slider(
        start=0, stop=100, step=1, value=0,
        label="Where to start the plot",
        show_value=True,
    )
    sanity_check_start_of_data_slider
    return (sanity_check_start_of_data_slider,)


@app.cell
def _sanity_check(
    build_haar_kernel,
    data_ok,
    detail_series,
    log_price,
    mo,
    np,
    plt,
    pywt,
    sanity_check_start_of_data_slider,
    scales,
):
    mo.stop(not data_ok or len(log_price) == 0)


    start_value = sanity_check_start_of_data_slider.value
    _wavelet = "haar"
    _max_level = len(scales)

    # Reproduce your original pywt pipeline
    _coeffs = pywt.wavedec(log_price, _wavelet, level=_max_level)

    def _upsample(cD, level, n):
        return pywt.upcoef("d", cD, _wavelet, level=level, take=n)

    _detail_pywt = np.array([
        _upsample(_coeffs[-lv], lv, len(log_price))
        for lv in range(1, _max_level + 1)
    ])

    # Only valid if current n_conv == 1
    _n_kernel = len(build_haar_kernel(1))  # proxy for order check

    fig_sanity, axes_s = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    fig_sanity.suptitle(
        "Sanity check: à trous (n=1, scale=2) vs pywt upcoef (scale=2)",
        fontsize=12,
    )
    _T = min(5000, len(log_price))
    axes_s[0].plot(detail_series[0, start_value:_T], lw=0.8, label="à trous n=1")
    axes_s[0].plot(_detail_pywt[0, start_value:_T], lw=0.8, alpha=0.7, linestyle="--", label="pywt upcoef")
    axes_s[0].legend(fontsize=8)
    axes_s[0].set_ylabel("Detail coeff")

    _diff = detail_series[0, start_value:_T] - _detail_pywt[0, start_value:_T]
    axes_s[1].plot(_diff, lw=0.8, color="red")
    axes_s[1].axhline(0, color="gray", lw=0.5)
    axes_s[1].set_ylabel("Residual")
    axes_s[1].set_xlabel("Time step")
    plt.tight_layout()
    plt.show()
    mo.md("### Sanity check — à trous (n=1) vs pywt")
    return


@app.cell
def _phase3_autocorr(
    build_detail_series,
    corr_function,
    data_ok,
    log_price,
    log_vol_from_detail,
    max_lag_slider,
    max_scale_slider,
    mo,
    np,
    overlay_orders_ui,
    plt,
    scale_idx_ref_ui,
):
    mo.stop(not data_ok or len(log_price) == 0)

    _orders = sorted(overlay_orders_ui.value) or [1, 2, 4, 8]
    _max_level = max_scale_slider.value
    _max_lag = max_lag_slider.value
    _ref_idx = min(scale_idx_ref_ui.value, _max_level - 1)

    with mo.status.spinner(title="Computing autocorrelations for all wavelet orders…"):
        _lags = np.arange(1, _max_lag + 1)
        _cmap_o = plt.cm.plasma(np.linspace(0.1, 0.9, len(_orders)))

        fig_ac, axes_ac = plt.subplots(1, 2, figsize=(14, 5))
        fig_ac.suptitle(
            "Phase 3 — Log-volatility autocorrelation vs ln(Δt)  [Figure 2 style]",
            fontsize=13,
        )

        # Left: correlation vs ln(lag), all scales overlaid for each order
        # Right: correlation at the reference scale only, all orders overlaid
        for _oi, (_n, _col) in enumerate(zip(_orders, _cmap_o)):
            _ds, _sc = build_detail_series(log_price, _n, _max_level)
            _lv = log_vol_from_detail(_ds, window=50)

            # Left panel: reference scale only, colour = order
            _w = _lv[_ref_idx]
            _C = corr_function(_w, _max_lag)
            axes_ac[0].scatter(
                np.log(_lags), _C[1:],
                s=3, color=_col, alpha=0.6,
                label=f"n={_n}",
            )

            # Right panel: all scales, colour = order (lighter = larger scale)
            _scale_alphas = np.linspace(0.3, 1.0, _max_level)
            for _si in range(_max_level):
                _w_s = _lv[_si]
                _C_s = corr_function(_w_s, _max_lag)
                axes_ac[1].scatter(
                    np.log(_lags), _C_s[1:],
                    s=2, color=_col, alpha=_scale_alphas[_si],
                )

        axes_ac[0].set_xlabel("ln(Δt)")
        axes_ac[0].set_ylabel("Autocorrelation")
        axes_ac[0].set_title(f"Reference scale index = {_ref_idx}")
        axes_ac[0].legend(markerscale=4, fontsize=8)
        axes_ac[0].axhline(0, color="gray", lw=0.5)

        axes_ac[1].set_xlabel("ln(Δt)")
        axes_ac[1].set_title("All scales (opacity ∝ scale size)")
        axes_ac[1].axhline(0, color="gray", lw=0.5)

    plt.tight_layout()
    plt.show()
    mo.md("### Phase 3 — Log-volatility autocorrelation")
    return


@app.cell
def _phase4_mi_single(
    compute_mi_map,
    data_ok,
    log_price,
    log_vol_series,
    mi_max_lag_slider,
    mo,
    plt,
    scale_idx_ref_ui,
    scales,
):
    mo.stop(not data_ok or len(log_price) == 0)

    _max_time_lag = mi_max_lag_slider.value
    _ref_idx = min(scale_idx_ref_ui.value, len(scales) - 1)

    with mo.status.spinner(title="Computing MI map…"):
        I_map = compute_mi_map(log_vol_series, _ref_idx, _max_time_lag)

    fig_mi, axes_mi = plt.subplots(1, 2, figsize=(14, 5))
    fig_mi.suptitle("Phase 4 — Mutual information cone (current wavelet order)", fontsize=13)

    _extent = [-_max_time_lag, _max_time_lag, len(scales) - 0.5, -0.5]

    _im = axes_mi[0].imshow(
        I_map, aspect="auto", origin="upper",
        cmap="hot", extent=_extent,
    )
    plt.colorbar(_im, ax=axes_mi[0], label="MI (nats)")
    axes_mi[0].set_xlabel("Time lag (steps)")
    axes_mi[0].set_yticks(range(len(scales)))
    axes_mi[0].set_yticklabels([f"2^{j+1}" for j in range(len(scales))], fontsize=7)
    axes_mi[0].set_ylabel("Scale (samples)")
    axes_mi[0].set_title("MI map")

    # Normalised by row max (shows cone shape independent of amplitude)
    _row_max = I_map.max(axis=1, keepdims=True)
    _row_max[_row_max == 0] = 1.0
    _I_norm = I_map / _row_max
    _im2 = axes_mi[1].imshow(
        _I_norm, aspect="auto", origin="upper",
        cmap="hot", extent=_extent, vmin=0, vmax=1,
    )
    plt.colorbar(_im2, ax=axes_mi[1], label="MI / max(MI per scale)")
    axes_mi[1].set_xlabel("Time lag (steps)")
    axes_mi[1].set_yticks(range(len(scales)))
    axes_mi[1].set_yticklabels([f"2^{j+1}" for j in range(len(scales))], fontsize=7)
    axes_mi[1].set_title("Row-normalised MI map")

    plt.tight_layout()
    mo.md("### Phase 4 — MI cone (single order)")
    return


@app.cell
def _phase4_mi_grid(
    build_detail_series,
    compute_mi_map,
    data_ok,
    log_price,
    log_vol_from_detail,
    max_scale_slider,
    mi_max_lag_slider,
    mo,
    overlay_orders_ui,
    plt,
    scale_idx_ref_ui,
):
    mo.stop(not data_ok or len(log_price) == 0)

    _orders = sorted(overlay_orders_ui.value) or [1, 2, 4, 8]
    _max_level = max_scale_slider.value
    _max_time_lag = mi_max_lag_slider.value
    _ref_idx = min(scale_idx_ref_ui.value, _max_level - 1)
    _n_orders = len(_orders)

    with mo.status.spinner(title=f"Computing MI maps for {_n_orders} wavelet orders…"):
        _maps = []
        for _n in _orders:
            _ds, _sc = build_detail_series(log_price, _n, _max_level)
            _lv = log_vol_from_detail(_ds, window=50)
            _maps.append(compute_mi_map(_lv, _ref_idx, _max_time_lag))

    fig_grid, axes_g = plt.subplots(
        1, _n_orders,
        figsize=(4 * _n_orders, 4),
        squeeze=False,
    )
    fig_grid.suptitle(
        "MI cone comparison across wavelet orders  (row-normalised)",
        fontsize=13,
    )
    _extent = [-_max_time_lag, _max_time_lag, _max_level - 0.5, -0.5]

    for _oi, (_n, _I) in enumerate(zip(_orders, _maps)):
        _row_max = _I.max(axis=1, keepdims=True)
        _row_max[_row_max == 0] = 1.0
        _I_norm = _I / _row_max
        _ax = axes_g[0, _oi]
        _im = _ax.imshow(
            _I_norm, aspect="auto", origin="upper",
            cmap="hot", extent=_extent, vmin=0, vmax=1,
        )
        _ax.set_title(f"n={_n}", fontsize=11)
        _ax.set_xlabel("Lag")
        if _oi == 0:
            _ax.set_ylabel("Scale index")
        else:
            _ax.set_yticks([])

    plt.colorbar(_im, ax=axes_g[0, -1], label="MI / max", shrink=0.8)
    plt.tight_layout()
    mo.md("### Phase 4b — MI cone grid across wavelet orders")
    return


@app.cell
def _phase5_cwt(
    compute_mi_map,
    corr_function,
    cwt_norm_slider,
    data_ok,
    log_price,
    log_vol_from_detail,
    max_scale_slider,
    mi_max_lag_slider,
    mo,
    np,
    plt,
    scale_idx_ref_ui,
):
    mo.stop(not data_ok or len(log_price) == 0)

    import pywt as _pywt

    _max_level = max_scale_slider.value
    _max_time_lag = mi_max_lag_slider.value
    _ref_idx = min(scale_idx_ref_ui.value, _max_level - 1)
    _norm_exp = cwt_norm_slider.value

    # CWT scales: match the DWT dyadic grid in samples
    _cwt_scales = np.array([2 ** j for j in range(1, _max_level + 1)], dtype=float)

    with mo.status.spinner(title="Running pywt CWT (Morlet)…"):
        # pywt.cwt returns (coefs, freqs); coefs shape = (n_scales, N)
        _coefs, _freqs = _pywt.cwt(
            log_price, _cwt_scales, "morl", sampling_period=1.0
        )
        # Normalise by scale^norm_exp (paper uses 1/sqrt(a) = 0.5)
        for _j, _a in enumerate(_cwt_scales):
            _coefs[_j] = _coefs[_j] / (_a ** _norm_exp)

        # Build log-vol from CWT modulus (use as the "detail series")
        _detail_cwt = np.abs(_coefs)   # shape (n_scales, N)
        _log_vol_cwt = log_vol_from_detail(_detail_cwt, window=50)
        _I_map_cwt = compute_mi_map(_log_vol_cwt, _ref_idx, _max_time_lag)

    # ---- Plot: autocorrelation at reference scale ----
    _max_lag_plot = min(2000, len(log_price) // 4)
    _lags = np.arange(1, _max_lag_plot + 1)
    _C_cwt = corr_function(_log_vol_cwt[_ref_idx], _max_lag_plot)

    fig_cwt, axes_cwt = plt.subplots(1, 2, figsize=(14, 5))
    fig_cwt.suptitle(
        f"Phase 5 — CWT (Morlet, norm_exp={_norm_exp:.2f})", fontsize=13
    )

    axes_cwt[0].scatter(np.log(_lags), _C_cwt[1:], s=3, color="steelblue")
    axes_cwt[0].axhline(0, color="gray", lw=0.5)
    axes_cwt[0].set_xlabel("ln(Δt)")
    axes_cwt[0].set_ylabel("Autocorrelation")
    axes_cwt[0].set_title(f"CWT log-vol autocorr  (scale idx={_ref_idx})")

    _row_max_cwt = _I_map_cwt.max(axis=1, keepdims=True)
    _row_max_cwt[_row_max_cwt == 0] = 1.0
    _I_norm_cwt = _I_map_cwt / _row_max_cwt
    _extent_cwt = [-_max_time_lag, _max_time_lag, _max_level - 0.5, -0.5]
    _im_cwt = axes_cwt[1].imshow(
        _I_norm_cwt, aspect="auto", origin="upper",
        cmap="hot", extent=_extent_cwt, vmin=0, vmax=1,
    )
    plt.colorbar(_im_cwt, ax=axes_cwt[1], label="MI / max")
    axes_cwt[1].set_xlabel("Time lag")
    axes_cwt[1].set_ylabel("Scale index")
    axes_cwt[1].set_title("CWT MI cone (row-normalised)")

    plt.tight_layout()
    mo.md("### Phase 5 — CWT (Morlet) comparison")
    return


@app.cell
def _summary_table(
    build_detail_series,
    corr_function,
    data_ok,
    log_price,
    log_vol_from_detail,
    max_lag_slider,
    max_scale_slider,
    mo,
    np,
    overlay_orders_ui,
    scale_idx_ref_ui,
):
    mo.stop(not data_ok or len(log_price) == 0)

    _orders = sorted(overlay_orders_ui.value) or [1, 2, 4, 8]
    _max_level = max_scale_slider.value
    _ref_idx = min(scale_idx_ref_ui.value, _max_level - 1)
    _max_lag = max_lag_slider.value

    def _slope_loglog(C, lags, fit_range=(5, 200)):
        """Fit ln(C) ~ beta * ln(lag) in the power-law range."""
        mask = (lags >= fit_range[0]) & (lags <= fit_range[1]) & (C > 0)
        if mask.sum() < 5:
            return float("nan")
        lx, ly = np.log(lags[mask]), np.log(C[mask])
        beta = np.polyfit(lx, ly, 1)[0]
        return beta

    _rows = []
    _lags = np.arange(1, _max_lag + 1)

    for _n in _orders:
        _ds, _ = build_detail_series(log_price, _n, _max_level)
        _lv = log_vol_from_detail(_ds, window=50)
        _C = corr_function(_lv[_ref_idx], _max_lag)
        _beta = _slope_loglog(_C[1:], _lags)
        _C50 = float(_C[min(50, _max_lag)])
        _C500 = float(_C[min(500, _max_lag)])
        _rows.append({
            "Wavelet order n": _n,
            "Type": "Haar" if _n == 1 else f"B-spline n={_n}",
            "C(50)": f"{_C50:.3f}",
            "C(500)": f"{_C500:.3f}",
            "Power-law slope β": f"{_beta:.3f}" if not np.isnan(_beta) else "—",
        })

    # Add CWT row
    try:
        _C_cwt = corr_function(_log_vol_cwt[_ref_idx], _max_lag)
        _beta_cwt = _slope_loglog(_C_cwt[1:], _lags)
        _rows.append({
            "Wavelet order n": "∞",
            "Type": "CWT Morlet",
            "C(50)": f"{float(_C_cwt[min(50,_max_lag)]):.3f}",
            "C(500)": f"{float(_C_cwt[min(500,_max_lag)]):.3f}",
            "Power-law slope β": f"{_beta_cwt:.3f}" if not np.isnan(_beta_cwt) else "—",
        })
    except Exception:
        pass

    mo.md("### Diagnostic summary — log-vol autocorrelation metrics")
    return


@app.cell
def _shuffled_control(
    build_detail_series,
    compute_mi_map,
    data_ok,
    log_price,
    log_vol_from_detail,
    max_scale_slider,
    mi_max_lag_slider,
    mo,
    n_conv_slider,
    np,
    plt,
    scale_idx_ref_ui,
):
    mo.stop(not data_ok or len(log_price) == 0)

    _n = n_conv_slider.value
    _max_level = max_scale_slider.value
    _max_time_lag = mi_max_lag_slider.value
    _ref_idx = min(scale_idx_ref_ui.value, _max_level - 1)

    with mo.status.spinner(title="Computing shuffled control MI map…"):
        _rng = np.random.default_rng(0)
        _increments = np.diff(log_price)
        _shuffled = _rng.permutation(_increments)
        _log_price_sh = np.concatenate([[log_price[0]], log_price[0] + np.cumsum(_shuffled)])

        _ds_sh, _sc_sh = build_detail_series(_log_price_sh, _n, _max_level)
        _lv_sh = log_vol_from_detail(_ds_sh, window=50)
        _I_sh = compute_mi_map(_lv_sh, _ref_idx, _max_time_lag)

    fig_sh, axes_sh = plt.subplots(1, 2, figsize=(12, 4))
    fig_sh.suptitle(
        f"Shuffled control — MI cone  (n={_n}, should be noisy/flat)",
        fontsize=12,
    )
    _extent_sh = [-_max_time_lag, _max_time_lag, _max_level - 0.5, -0.5]

    _im_sh = axes_sh[0].imshow(
        _I_sh, aspect="auto", origin="upper",
        cmap="hot", extent=_extent_sh,
    )
    plt.colorbar(_im_sh, ax=axes_sh[0], label="MI (nats)")
    axes_sh[0].set_title("Raw MI")
    axes_sh[0].set_xlabel("Lag")
    axes_sh[0].set_ylabel("Scale index")

    _row_max_sh = _I_sh.max(axis=1, keepdims=True)
    _row_max_sh[_row_max_sh == 0] = 1.0
    _im_sh2 = axes_sh[1].imshow(
        _I_sh / _row_max_sh, aspect="auto", origin="upper",
        cmap="hot", extent=_extent_sh, vmin=0, vmax=1,
    )
    plt.colorbar(_im_sh2, ax=axes_sh[1], label="MI / max")
    axes_sh[1].set_title("Row-normalised MI")
    axes_sh[1].set_xlabel("Lag")

    plt.tight_layout()
    mo.md("### Shuffled control")
    return


if __name__ == "__main__":
    app.run()
