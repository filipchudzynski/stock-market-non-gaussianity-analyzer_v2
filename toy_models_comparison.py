import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import fftconvolve
    from log_energy_epjst_package.models.white_noise import white_noise
    from log_energy_epjst_package.models.brownian_motion import brownian_motion
    from MRW_LAMBDA_ESTIMATOR.surrogate_mrw_lambda import generate_mrw
    from signal_analysis import analyze_signal, psi_haar_smooth
    from toy_models.model5_lognormal_cascade import generate as generate_lognormal_cascade
    from toy_models.model8_fbm import fbm_daviesharte as generate_fbm

    def covariance_fft(x):
        x = x - np.mean(x)
        c = fftconvolve(x, x[::-1], mode='full')
        c = c[c.size // 2:]
        return c / c[0]

    def compare_signals(signals, **analysis_kwargs):
        results = {}
        for name, signal in signals.items():
            results[name] = analyze_signal(signal, **analysis_kwargs)
        return results

    def plot_signals(results):
        """Plot the original signals (treated as log-price)."""
        fig, axes = plt.subplots(len(results), 1, figsize=(10, 4 * len(results)), sharex=True)
        if len(results) == 1:
            axes = [axes]
        for ax, (name, output) in zip(axes, results.items()):
            ax.plot(output['signal'])
            ax.set_title(f'{name}: Signal')
            ax.set_ylabel('Value')
        plt.xlabel('Time')
        plt.tight_layout()
        plt.show()

    def plot_volatility_walks(results, scale_idx=3):
        """Plot centered log-volatility walks for each signal."""
        fig, axes = plt.subplots(len(results), 1, figsize=(10, 4 * len(results)), sharex=True)
        if len(results) == 1:
            axes = [axes]
        for ax, (name, output) in zip(axes, results.items()):
            w_a = output['log_vol_series'][scale_idx]
            w_a_centered = w_a - np.mean(w_a)
            v_a = np.cumsum(w_a_centered)
            ax.plot(v_a)
            ax.set_title(f"{name}: Centered Log-Volatility Walk (Scale {output['scales'][scale_idx]})")
            ax.set_ylabel('Walk Value')
        plt.xlabel('Time')
        plt.tight_layout()
        plt.show()

    def plot_autocorrelations(results, scale_idx=3, max_lag=500):
        """Plot autocorrelations of returns and log-volatility."""
        fig, axes = plt.subplots(len(results), 2, figsize=(12, 4 * len(results)))
        if len(results) == 1:
            axes = axes.reshape(1, -1)
        for i, (name, output) in enumerate(results.items()):
            signal = output['signal']
            ret = np.diff(signal)
            ret_ac = covariance_fft(ret)
            w_a = output['log_vol_series'][scale_idx]
            w_ac = covariance_fft(w_a)
            axes[i, 0].plot(ret_ac[:max_lag + 1])
            axes[i, 0].set_title(f'{name}: Return Autocorrelation')
            axes[i, 0].set_xlabel('Lag')
            axes[i, 0].set_ylabel('Autocorrelation')
            axes[i, 1].plot(w_ac[:max_lag + 1])
            axes[i, 1].set_title(f"{name}: Log-Volatility Autocorrelation (Scale {output['scales'][scale_idx]})")
            axes[i, 1].set_xlabel('Lag')
            axes[i, 1].set_ylabel('Autocorrelation')
        plt.tight_layout()
        plt.show()

    def plot_mi_maps(results, max_time_lag):
        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), sharey=True)
        if len(results) == 1:
            axes = [axes]
        for ax, (name, output) in zip(axes, results.items()):
            im = ax.imshow(output['mi_map'], aspect='auto', origin='upper', cmap='hot', extent=[-max_time_lag, max_time_lag, output['scales'][-1], output['scales'][0]])
            ax.set_title(name)
            ax.set_xlabel('Time lag')
            ax.set_ylabel('Scale')
            fig.colorbar(im, ax=ax, label='Mutual Information')
        plt.tight_layout()
        plt.show()
        for key in results.keys():
            mi_normalized = []
            for i, I in enumerate(results[key]['mi_map']):
                mi_normalized.append(I / np.max(I))
            results[key]['mi_map_normalized'] = mi_normalized
        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), sharey=True)
        if len(results) == 1:
            axes = [axes]
        for ax, (name, output) in zip(axes, results.items()):
            im = ax.imshow(output['mi_map_normalized'], aspect='auto', origin='upper', cmap='hot', extent=[-max_time_lag, max_time_lag, output['scales'][-1], output['scales'][0]])
            ax.set_title(name)
            ax.set_xlabel('Time lag')
            ax.set_ylabel('Scale')
            fig.colorbar(im, ax=ax, label='Mutual Information normalized')
        plt.tight_layout()
        plt.show()

    return (
        brownian_motion,
        compare_signals,
        generate_fbm,
        generate_lognormal_cascade,
        generate_mrw,
        np,
        plot_autocorrelations,
        plot_mi_maps,
        plot_signals,
        plot_volatility_walks,
        psi_haar_smooth,
        white_noise,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # old attempts using pywt
    """)
    return


@app.cell
def _(
    brownian_motion,
    compare_signals,
    plot_autocorrelations,
    plot_mi_maps,
    plot_signals,
    plot_volatility_walks,
    white_noise,
):
    if __name__ == '__main__':
        _n_samples = 50000
        _signals = {'white_noise': white_noise(_n_samples), 'brownian_motion': brownian_motion(_n_samples)}
        _results = compare_signals(_signals, wavelet='haar', max_level=8, window=50, ref_idx=0, max_time_lag=10, use_parallel=True, n_jobs=8)
        for _name, _output in _results.items():
            print(f"{_name}: log_vol_series shape = {_output['log_vol_series'].shape}, mi_map shape = {_output['mi_map'].shape}")
        plot_signals(_results)
        plot_volatility_walks(_results, scale_idx=3)
        plot_autocorrelations(_results, scale_idx=3, max_lag=500)
        plot_mi_maps(_results, max_time_lag=10)
    return


@app.cell
def _(
    compare_signals,
    generate_lognormal_cascade,
    plot_autocorrelations,
    plot_mi_maps,
    plot_signals,
    plot_volatility_walks,
):
    if __name__ == '__main__':
        _n_samples = 2 ** 15
        _signals = {'lognormal_cascade_0.05': generate_lognormal_cascade(_n_samples, 0.05), 'lognormal_cascade_0.1': generate_lognormal_cascade(_n_samples, 0.1), 'lognormal_cascade_0.5': generate_lognormal_cascade(_n_samples, 0.5)}
        _results = compare_signals(_signals, wavelet='haar', max_level=8, window=50, ref_idx=0, max_time_lag=10, use_parallel=True, n_jobs=8)
        for _name, _output in _results.items():
            print(f"{_name}: log_vol_series shape = {_output['log_vol_series'].shape}, mi_map shape = {_output['mi_map'].shape}")
        plot_signals(_results)
        plot_volatility_walks(_results, scale_idx=3)
        plot_autocorrelations(_results, scale_idx=3, max_lag=500)
        plot_mi_maps(_results, max_time_lag=10)
    return


@app.cell
def _(
    compare_signals,
    generate_mrw,
    plot_autocorrelations,
    plot_mi_maps,
    plot_signals,
    plot_volatility_walks,
):
    if __name__ == '__main__':
        _n_samples = 2 ** 10
        _signals = {'mrw_0.05': generate_mrw(_n_samples, 0.05)[0], 'mrw_0.1': generate_mrw(_n_samples, 0.1)[0], 'mrw_0.5': generate_mrw(_n_samples, 0.5)[0]}
        _results = compare_signals(_signals, wavelet='haar', max_level=8, window=50, ref_idx=0, max_time_lag=10, use_parallel=True, n_jobs=8)
        for _name, _output in _results.items():
            print(f"{_name}: log_vol_series shape = {_output['log_vol_series'].shape}, mi_map shape = {_output['mi_map'].shape}")
        plot_signals(_results)
        plot_volatility_walks(_results, scale_idx=3)
        plot_autocorrelations(_results, scale_idx=3, max_lag=500)
        plot_mi_maps(_results, max_time_lag=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # new attempts using custom cwt
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## toy models
     signal length 10000
    """)
    return


@app.cell
def _(plot_mi_maps, results):
    plot_mi_maps(results, max_time_lag=100)
    return


@app.cell
def _(
    compare_signals,
    generate_fbm,
    np,
    plot_autocorrelations,
    plot_mi_maps,
    plot_signals,
    plot_volatility_walks,
    psi_haar_smooth,
    white_noise,
):
    n_samples = 10000
    scales = np.logspace(np.log10(4), np.log10(2000), 25)
    signals = {
        'white_noise': white_noise(n_samples),
        'fbm_0.2': generate_fbm(n_samples, 0.2), 
        'fbm_0.5': generate_fbm(n_samples, 0.5), 
        'fbm_0.8': generate_fbm(n_samples, 0.8)
    }
    results = compare_signals(signals, scales=scales, wavelet=psi_haar_smooth, max_level=8, window=50, ref_idx=0, max_time_lag=400, use_parallel=True, n_jobs=8)

    for name, output in results.items():
        print(f"{name}: log_vol_series shape = {output['log_vol_series'].shape}, mi_map shape = {output['mi_map'].shape}")
    plot_signals(results)
    plot_volatility_walks(results, scale_idx=3)
    plot_autocorrelations(results, scale_idx=3, max_lag=500)
    plot_mi_maps(results, max_time_lag=400)
    return (results,)


@app.cell
def _(plot_mi_maps, results):
    fbm_only = {"fbm_0.8":results["fbm_0.8"]}
    plot_mi_maps(fbm_only, max_time_lag=400)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### fbm 0.8
    signal lenghth 30000
    """)
    return


@app.cell
def _(plot_mi_maps, results_fbm_2):
    plot_mi_maps(results_fbm_2, max_time_lag=600)

    return


@app.cell
def _(
    compare_signals,
    generate_fbm,
    np,
    plot_autocorrelations,
    plot_mi_maps,
    plot_signals,
    plot_volatility_walks,
    psi_haar_smooth,
):
    def analyse_fbm():
        n_samples = 30000
        scales = np.logspace(np.log10(4), np.log10(2000), 25)
        signals = {
            'fbm_0.8': generate_fbm(n_samples, 0.8)
        }
        results = compare_signals(signals, scales=scales, wavelet=psi_haar_smooth, max_level=8, window=50, ref_idx=0, max_time_lag=400, use_parallel=True, n_jobs=8)
    
        for name, output in results.items():
            print(f"{name}: log_vol_series shape = {output['log_vol_series'].shape}, mi_map shape = {output['mi_map'].shape}")
        plot_signals(results)
        plot_volatility_walks(results, scale_idx=3)
        plot_autocorrelations(results, scale_idx=3, max_lag=500)
        plot_mi_maps(results, max_time_lag=400)
        return results
    results_fbm = analyse_fbm()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## fbm 0.8
    signal length 15000

    MI map +/-600
    """)
    return


@app.cell
def _(
    compare_signals,
    generate_fbm,
    np,
    plot_autocorrelations,
    plot_mi_maps,
    plot_signals,
    plot_volatility_walks,
    psi_haar_smooth,
):
    def analyse_fbm_2():
        n_samples = 15000
        scales = np.logspace(np.log10(4), np.log10(2000), 25)
        signals = {
            'fbm_0.8': generate_fbm(n_samples, 0.8)
        }
        results = compare_signals(signals, scales=scales, wavelet=psi_haar_smooth, max_level=8, window=50, ref_idx=0, max_time_lag=600, use_parallel=True, n_jobs=8)
    
        for name, output in results.items():
            print(f"{name}: log_vol_series shape = {output['log_vol_series'].shape}, mi_map shape = {output['mi_map'].shape}")
        plot_signals(results)
        plot_volatility_walks(results, scale_idx=3)
        plot_autocorrelations(results, scale_idx=3, max_lag=500)
        plot_mi_maps(results, max_time_lag=400)
        return results
    results_fbm_2 = analyse_fbm_2()
    return (results_fbm_2,)


@app.cell
def _(plot_mi_maps, results_fbm_2):
    plot_mi_maps(results_fbm_2,max_time_lag=600)
    return


@app.cell
def _(
    compare_signals,
    generate_fbm,
    np,
    plot_autocorrelations,
    plot_mi_maps,
    plot_signals,
    plot_volatility_walks,
    psi_haar_smooth,
):
    def analyse_fbm_3():
        n_samples = 15000
        mi_map_width = 2000
        scales = np.logspace(np.log10(4), np.log10(2000), 25)
        signals = {
            'fbm_0.8': generate_fbm(n_samples, 0.8)
        }
        results = compare_signals(signals, scales=scales, wavelet=psi_haar_smooth, max_level=8, window=50, ref_idx=0, max_time_lag=mi_map_width, use_parallel=True, n_jobs=8)
    
        for name, output in results.items():
            print(f"{name}: log_vol_series shape = {output['log_vol_series'].shape}, mi_map shape = {output['mi_map'].shape}")
        plot_signals(results)
        plot_volatility_walks(results, scale_idx=3)
        plot_autocorrelations(results, scale_idx=3, max_lag=500)
        plot_mi_maps(results, max_time_lag=mi_map_width)
        return results
    results_fbm_3 = analyse_fbm_3()
    return


@app.cell
def _(np):
    import pandas as pd

    DATA_PATH = "SandP_log2min.dat"   # one value per row

    # Try reading with header; if no header, fallback
    try:
        df = pd.read_csv(DATA_PATH)
        if df.shape[1] == 1:
            df.columns = ['price']
    except:
        df = pd.read_csv(DATA_PATH, header=None)
        df.columns = ['price']

    df = df.dropna().reset_index(drop=True)

    # Log-price
    df['log_price'] = np.log(df['price'])

    # Since there is no intraday pattern, we skip deseasonalization
    df['log_price_deseason'] = df['log_price']

    df.head()


    return df, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## S&P
    signal length 30000
    """)
    return


@app.cell
def _(
    compare_signals,
    df,
    np,
    plot_autocorrelations,
    plot_mi_maps,
    plot_signals,
    plot_volatility_walks,
    psi_haar_smooth,
):
    def analyse_SandP():
        log_price = np.array(df['log_price_deseason'].values)
    
        n_samples = 30000
        scales = np.logspace(np.log10(4), np.log10(2000), 25)
        signals = {
            'S&P500': log_price[:n_samples]
        }
        results = compare_signals(signals, scales=scales, wavelet=psi_haar_smooth, max_level=8, window=50, ref_idx=0, max_time_lag=400, use_parallel=True, n_jobs=8)
    
        for name, output in results.items():
            print(f"{name}: log_vol_series shape = {output['log_vol_series'].shape}, mi_map shape = {output['mi_map'].shape}")
        plot_signals(results)
        plot_volatility_walks(results, scale_idx=3)
        plot_autocorrelations(results, scale_idx=3, max_lag=500)
        plot_mi_maps(results, max_time_lag=400)

        return results
    results_snp = analyse_SandP()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## S&P
    signal length 60000
    """)
    return


@app.cell
def _(
    compare_signals,
    df,
    np,
    plot_autocorrelations,
    plot_mi_maps,
    plot_signals,
    plot_volatility_walks,
    psi_haar_smooth,
):
    def analyse_SandP_2():
        log_price = np.array(df['log_price_deseason'].values)
    
        n_samples = 60000
        scales = np.logspace(np.log10(4), np.log10(2000), 25)
        signals = {
            'S&P500': log_price[:n_samples]
        }
        results = compare_signals(signals, scales=scales, wavelet=psi_haar_smooth, max_level=8, window=50, ref_idx=0, max_time_lag=400, use_parallel=True, n_jobs=8)
    
        for name, output in results.items():
            print(f"{name}: log_vol_series shape = {output['log_vol_series'].shape}, mi_map shape = {output['mi_map'].shape}")
        plot_signals(results)
        plot_volatility_walks(results, scale_idx=3)
        plot_autocorrelations(results, scale_idx=3, max_lag=500)
        plot_mi_maps(results, max_time_lag=400)

        return results
    results_snp_2 = analyse_SandP_2()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## BTC
    data length 30000
    """)
    return


@app.cell
def _(pd):
    DATA_DIR_BTC = 'bitstamp-btcusd-minute-data/data'
    df_hist_btc = pd.read_csv(
        f'{DATA_DIR_BTC}/historical/btcusd_bitstamp_1min_2012-2025.csv',
    )
    return (df_hist_btc,)


@app.cell
def _(df_hist_btc):
    df_hist_btc.head()
    return


@app.cell
def _(
    compare_signals,
    df_hist_btc,
    np,
    plot_autocorrelations,
    plot_mi_maps,
    plot_signals,
    plot_volatility_walks,
    psi_haar_smooth,
):
    def analyze_btc():
        n_samples = 30000

        log_price = np.array(df_hist_btc["open"].values)
        scales = np.logspace(np.log10(4), np.log10(2000), 25)
        signals = {
            # I put -n_samples: instead of :n_samples as before because the beginning of the btc signal might be not very representative
            'BTC': log_price[-n_samples:]
        }
        results = compare_signals(signals, scales=scales, wavelet=psi_haar_smooth, max_level=8, window=50, ref_idx=0, max_time_lag=400, use_parallel=True, n_jobs=8)
    
        for name, output in results.items():
            print(f"{name}: log_vol_series shape = {output['log_vol_series'].shape}, mi_map shape = {output['mi_map'].shape}")
        plot_signals(results)
        plot_volatility_walks(results, scale_idx=3)
        plot_autocorrelations(results, scale_idx=3, max_lag=500)
        plot_mi_maps(results, max_time_lag=400)

        return results
    results_btc = analyze_btc()
    return


if __name__ == "__main__":
    app.run()
