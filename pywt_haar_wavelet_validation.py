import marimo

__generated_with = "0.23.5"
app = marimo.App(
    width="full",
    app_title="Haar-convolved Wavelets: Construction & Validation",
)


@app.cell
def _imports():
    import numpy as np
    import pandas as pd
    import pywt
    import math
    import marimo as mo
    import matplotlib.pyplot as plt

    return mo, np, pd, plt, pywt


@app.cell
def _params():
    conv_orders = [1, 2, 3, 4]
    signal_length = 512
    random_seed = 123
    return conv_orders, random_seed, signal_length


@app.cell
def _builder(np, pywt):
    def build_haar_convolved_wavelet(order: int) -> pywt.Wavelet:
        haar = pywt.Wavelet("haar")
        base = np.array(haar.dec_lo)   # this is the Haar scaling filter
        filt = base.copy()
        for _ in range(order - 1):
            filt = np.convolve(filt, base)

        filt = filt / np.linalg.norm(filt)
        qmf = ((-1)**np.arange(len(filt))) * filt[::-1]

        return pywt.Wavelet(
            name=f"haar_conv_{order}",
            filter_bank=[filt, qmf, filt, qmf]
        )

    return (build_haar_convolved_wavelet,)


@app.cell
def _wavelets(build_haar_convolved_wavelet, conv_orders):
    wavelet_dict = {order: build_haar_convolved_wavelet(order) for order in conv_orders}
    return (wavelet_dict,)


@app.cell
def _filter_checks(np, pd, wavelet_dict):
    results_basic = []

    for order, wave in wavelet_dict.items():
        h = np.array(wave.dec_lo)
        g = np.array(wave.dec_hi)
        N = len(h)

        energy = np.sum(h*h)
        sum_h = np.sum(h)
        sum_g = np.sum(g)
        qmf_expected = ((-1)**np.arange(N)) * h[::-1]
        qmf_error = np.linalg.norm(g - qmf_expected)

        results_basic.append({
            "order": order,
            "filter_len": N,
            "energy(h)": energy,
            "sum(h)": sum_h,
            "sum(g)": sum_g,
            "QMF error": qmf_error,
        })

    df_basic_checks = pd.DataFrame(results_basic).sort_values("order")
    df_basic_checks
    return


@app.cell
def _reconstruction(np, pd, pywt, random_seed, signal_length, wavelet_dict):
    def reconstruction():
        rng = np.random.default_rng(random_seed)
        signal = rng.standard_normal(signal_length)
    
        results_recon = []
    
        for order, wave in wavelet_dict.items():
            # single-level
            cA, cD = pywt.dwt(signal, wave)
            rec1 = pywt.idwt(cA, cD, wave)
            err1 = np.linalg.norm(signal - rec1) / np.linalg.norm(signal)
    
            # multi-level
            max_level = pywt.dwt_max_level(len(signal), len(wave.dec_lo))
            level = min(4, max_level)
            coeffs = pywt.wavedec(signal, wave, level=level)
            rec2 = pywt.waverec(coeffs, wave)[:len(signal)]
            err2 = np.linalg.norm(signal - rec2) / np.linalg.norm(signal)
    
            results_recon.append({
                "order": order,
                "levels": level,
                "1-level error": err1,
                "multi-level error": err2,
            })
    
        df_reconstruction = pd.DataFrame(results_recon).sort_values("order")
        return df_reconstruction
    df_reconstruction = reconstruction()
    df_reconstruction
    return


@app.cell
def _(mo, wavelet_dict):
    # UI controls
    order_dropdown = mo.ui.dropdown(
        options=list(wavelet_dict.keys()),
        value=list(wavelet_dict.keys())[0],
        label="Select Haar-convolution order"
    )

    plot_button = mo.ui.button("Plot wavefun")
    order_dropdown


    return (order_dropdown,)


@app.cell
def _(order_dropdown, plt, wavelet_dict):
    def plot_wfs():
        selected_order = order_dropdown.value

        # Reactive plotting
        for i in range(1,selected_order+1):
            wave = wavelet_dict[i]
    
            phi, psi,_,_, x = wave.wavefun(level=1)
    
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
            axes[0].plot(x, phi, color="blue")
            axes[0].set_title(f"Scaling function φ(x) — order {i}")
            axes[0].grid(True)
    
            axes[1].plot(x, psi, color="red")
            axes[1].set_title(f"Wavelet ψ(x) — order {i}")
            axes[1].grid(True)
    
            plt.tight_layout()
            plt.show()
    plot_wfs()
    return


if __name__ == "__main__":
    app.run()
