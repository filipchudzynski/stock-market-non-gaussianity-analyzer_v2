import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full", auto_download=["html", "ipynb"])


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

    def plot_mi_maps(results, max_time_lag,negative=False):
        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), sharey=True)
        if len(results) == 1:
            axes = [axes]
        for ax, (name, output) in zip(axes, results.items()):
            im = ax.imshow(output['mi_map'], aspect='auto', origin='upper', extent=[-max_time_lag, max_time_lag, output['scales'][-1], output['scales'][0]])
            ax.set_title(name)
            ax.set_xlabel('Time lag')
            ax.set_ylabel('Scale')
            fig.colorbar(im, ax=ax, label='Mutual Information')
        plt.tight_layout()
        plt.show()
        if not negative:
            for key in results.keys():
                mi_normalized = []
                for i, I in enumerate(results[key]['mi_map']):
                    mi_normalized.append(I / np.max(I))
                results[key]['mi_map_normalized'] = mi_normalized
        else:
            for key in results.keys():
                mi_normalized = []
                for i, I in enumerate(results[key]['mi_map']):
                    mi_normalized.append((I+np.abs(np.min(I))) / np.abs(np.max(I)-np.min(I)))
                results[key]['mi_map_normalized'] = mi_normalized
        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), sharey=True)
        if len(results) == 1:
            axes = [axes]
        for ax, (name, output) in zip(axes, results.items()):
            im = ax.imshow(output['mi_map_normalized'], aspect='auto', origin='upper' , extent=[-max_time_lag, max_time_lag, output['scales'][-1], output['scales'][0]])
            ax.set_title(name)
            ax.set_xlabel('Time lag')
            ax.set_ylabel('Scale')
            fig.colorbar(im, ax=ax, label='Mutual Information normalized')
        plt.tight_layout()
        plt.show()

    return (
        compare_signals,
        np,
        plot_autocorrelations,
        plot_mi_maps,
        plot_signals,
        plot_volatility_walks,
        plt,
        psi_haar_smooth,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # performance comparison
    multiprocessing of a single dt,s instance vs computing single dt instance and MI for all s is processed internally by scikit
    """)
    return


@app.cell
def _(np, plt):
    import pandas as pd

    DATA_PATH = "SandP_log2min.dat"   # one value per row

    # Try reading with header; if no header, fallback
    try:
        df_snp = pd.read_csv(DATA_PATH)
        if df_snp.shape[1] == 1:
            df_snp.columns = ['log_price']
    except:
        df_snp = pd.read_csv(DATA_PATH, header=None)
        df_snp.columns = ['log_price']

    df_snp = df_snp.dropna().reset_index(drop=True)

    # Log-price
    df_snp['price'] = 0.01*np.exp(df_snp['log_price'])

    print(len(df_snp))
    df_snp.head()

    plt.plot(0.01*np.exp(df_snp["log_price"]))
    plt.title("84-94")
    plt.show()
    return df_snp, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    30000 sample size
    400 lag length
    14 min

    30000 sample length
    250 lag length
    8 min

    20000 samle lenth
    250 lag length
    5 min

    64054/20000 = 32
    32*5=160 min
    64*5=320~5h
    """)
    return


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
    np,
    plot_autocorrelations,
    plot_mi_maps,
    plot_signals,
    plot_volatility_walks,
    psi_haar_smooth,
):
    def analyse_SandP(df,batched,method,index):
        log_price = np.array(df['log_price'].values)
        max_lag=250
        n_samples = 20000
        scales = np.logspace(np.log10(4), np.log10(2000), 25)
        signals = {
            'S&P500': log_price[:n_samples]
        }
        results = compare_signals(signals, scales=scales, wavelet=psi_haar_smooth, max_level=8, window=50, ref_idx=0, max_time_lag=max_lag, use_parallel=True, n_jobs=8,batched=batched,method=method)

        for name, output in results.items():
            print(f"{name}: log_vol_series shape = {output['log_vol_series'].shape}, mi_map shape = {output['mi_map'].shape}")
        if index%10==0:
            plot_signals(results)
            plot_volatility_walks(results, scale_idx=3)
            plot_autocorrelations(results, scale_idx=3, max_lag=500)
            plot_mi_maps(results, max_time_lag=max_lag)

        return results


    return (analyse_SandP,)


@app.cell
def _(analyse_SandP, df_snp):
    results_snp = analyse_SandP(df_snp,batched=False,method="binsearch")
    return (results_snp,)


@app.cell
def _(np):
    #results_snp_legacy = analyse_SandP(df_snp,batched=False)
    results_snp_legacy=np.load("ksg_optimization_legacy.npy",allow_pickle=True)
    return (results_snp_legacy,)


@app.cell
def _(results_snp_legacy):
    results_snp_legacy.item()["S&P500"]["mi_map"]
    return


@app.cell
def _(np, results_snp_legacy):
    for _i in results_snp_legacy.item()["S&P500"]["log_vol_series"]:
        print(np.any(np.unique(_i, return_counts=True)[1]>1))
    return


@app.cell
def _(plot_mi_maps, results_snp, results_snp_legacy):
    results_method_comp = {}
    results_method_comp["S&P-S&P legacy"] = {}
    results_method_comp["S&P-S&P legacy"]["scales"] = results_snp["S&P500"]["scales"]
    # results_method_comp["S&P-S&P legacy"]["mi_map"] = results_snp["S&P500"]["mi_map"] - results_snp_legacy["S&P500"]["mi_map"]
    results_method_comp["S&P-S&P legacy"]["mi_map"] = results_snp["S&P500"]["mi_map"] - results_snp_legacy.item()["S&P500"]["mi_map"]
    plot_mi_maps(results_method_comp,800)
    return (results_method_comp,)


@app.cell
def _(np, results_method_comp):
    np.average(results_method_comp["S&P-S&P legacy"]["mi_map"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    the speedup is negligible 26 min 50' vs 27 min 30'
    """)
    return


@app.cell
def _(results_snp, results_snp_legacy):
    ((results_snp["S&P500"]["mi_map"] - results_snp_legacy["S&P500"]["mi_map"])>0.00001).any() and not ((results_snp["S&P500"]["mi_map"] - results_snp_legacy["S&P500"]["mi_map"])>0.0001).any()
    return


@app.cell
def _(np):
    results_snp_legacy=np.load("ksg_optimization_legacy.npy")
    return (results_snp_legacy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## S&P
    156 rolling windows of size 20000
    step size around 1 month
    window size around 5 months
    """)
    return


@app.cell
def _(df_snp, np, plt):
    N = len(df_snp)          # should be 640000
    window_size = 20000
    step_size = 4000         # ~1 month
    num_windows = (N - window_size) // step_size + 1  # -> 152

    windows = []
    starts = []

    for k in range(num_windows):
        start = k * step_size
        end = start + window_size
        if end > N:
            break
        w = df_snp.iloc[start:end]
        windows.append(w)
        starts.append(start)

    print(len(windows))  # sanity check: 152

    plt.figure(figsize=(16, 6))

    # Full series
    plt.plot(df_snp["price"].values, color="lightgray", linewidth=1.0, label="Full series")

    cmap = plt.cm.tab20
    colors = plt.cm.tab20(np.arange(60))   # first 5 distinct colors


    for k, w in enumerate(windows):
        start = starts[k]
        end = start + window_size

        x = np.arange(start, end)          # length 20000
        y = w["price"].values              # length 20000

        plt.plot(x, y+50*(k%5), color=cmap.colors[(k//5)%20], linewidth=1.2, alpha=1)

    plt.title("Rolling windows (20,000 points) with monthly step (~4095 points)")
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.show()
    return step_size, window_size, windows


@app.cell
def _(analyse_SandP, np, step_size, window_size, windows):
    SnP_results_batch_custom = []
    for _i,_window in enumerate(windows[:78]):
        print(f"{_i}-batch")
        _result_batch = analyse_SandP(df=_window,batched=False,method="binsearch",index=_i)
        np.save(f"mi_map_snp_rolling_{_i*step_size}-{_i*step_size + window_size}.npy",_result_batch)
        SnP_results_batch_custom.append(_result_batch)
    return (SnP_results_batch_custom,)


@app.cell
def _(
    SnP_results_batch_custom,
    analyse_SandP,
    np,
    step_size,
    window_size,
    windows,
):
    for _i,_window in enumerate(windows[78:]):
        _base=78
        print(f"{_base+_i}-batch")
        _result_batch = analyse_SandP(df=_window,batched=False,method="binsearch",index=_i)
        np.save(f"mi_map_snp_rolling_2_{(_base+_i)*step_size}-{(_base+_i)*step_size + window_size}.npy",_result_batch)
        SnP_results_batch_custom.append(_result_batch)
    return


@app.cell
def _(SnP_results_batch_custom):
    for ind,result in enumerate(SnP_results_batch_custom):
        if "mi_map_normalized" not in result["S&P500"]:
            print(ind)
    return


@app.cell
def _(SnP_results_batch_custom, np):
    np.save("mi_map_snp_rolling_all.npy",SnP_results_batch_custom)
    return


@app.cell
def _(np, plt):
    from lmfit import Model
    from scipy.ndimage import gaussian_filter
    def smooth_mi_map(mi_map, sigma_scale=1.0, sigma_lag=2.0, alpha=0.0):
        """    alpha = regularisation strength (0 = none)
        """
        smoothed = gaussian_filter(mi_map, sigma=[sigma_scale, sigma_lag])

        if alpha > 0:
            # Tikhonov regularisation: (I + αL)^(-1) M
            # Here: simple shrinkage toward smoothness
            smoothed = (1 - alpha) * smoothed + alpha * gaussian_filter(smoothed, sigma=3)

        return smoothed
    def split_gaussian(x, amplitude, center, sigma_left, sigma_right):
        sigma = np.where(x < center, sigma_left, sigma_right)
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)

    def fit_single_ridge(row, lags, prev_params=None,
                         center_init=0.0, center_range=150.0,
                         sigma_init=30.0, sigma_min=2.0, sigma_max=400.0,
                         amplitude_thresh=0.05):
        """
        Fit a single asymmetric Gaussian to one scale row.

        Key fix: subtract row baseline before fitting so the peak
        is defined relative to background, not absolute MI level.
        """
        model = Model(split_gaussian)

        # ── baseline subtraction ──────────────────────────────────────────
        # Use a percentile rather than min to be robust to outliers
        baseline = np.percentile(row, 10)
        row_centered = row - baseline
        row_range = row_centered.max()

        # Reject flat rows early — nothing to fit
        if row_range < amplitude_thresh:
            return None, False

        # Normalise to [0,1] for stable fitting, rescale result after
        row_norm = row_centered / row_range

        # ── params ───────────────────────────────────────────────────────
        if prev_params is not None:
            params = prev_params.copy()
            params['center'].set(
                min=params['center'].value - center_range,
                max=params['center'].value + center_range,
            )
            # Reset amplitude to current row's scale
            params['amplitude'].set(value=1.0, min=amplitude_thresh, max=1.5)
        else:
            params = model.make_params(
                amplitude=dict(value=1.0, min=amplitude_thresh, max=1.5),
                center=dict(value=center_init,
                            min=center_init - center_range,
                            max=center_init + center_range),
                sigma_left=dict(value=sigma_init, min=sigma_min, max=sigma_max),
                sigma_right=dict(value=sigma_init, min=sigma_min, max=sigma_max),
            )

        try:
            result = model.fit(row_norm, params, x=lags)
        except Exception:
            return None, False

        # ── quality checks ────────────────────────────────────────────────
        p = result.params
        amp    = p['amplitude'].value
        sl     = p['sigma_left'].value
        sr     = p['sigma_right'].value
        center = p['center'].value

        failed = (
            not result.success
            or amp < amplitude_thresh           # too weak
            or sl >= sigma_max * 0.95           # sigma hit bound → degenerate
            or sr >= sigma_max * 0.95
            or not (lags[0] < center < lags[-1])# center outside data range
            or result.redchi > 1.0              # poor fit quality
        )

        if failed:
            return None, False

        # Store row_range so caller can rescale amplitude if needed
        result.row_range = row_range
        result.baseline  = baseline
        return result, True


    def extract_single_ridge_lmfit(mi_map, scales, lags,
                                    smooth_sigma_scale=2.0, smooth_sigma_lag=2.0,
                                    center_init=0.0, center_range=150.0,
                                    sigma_init=30.0, sigma_max=400.0,
                                    amplitude_thresh=0.05,
                                    fit_window=None):
        """
        fit_window : (lag_min, lag_max) or None.
                     Restrict fitting to a lag sub-window around the expected ridge.
                     Dramatically helps when large-scale rows are flat outside the peak.
        """
        sm = smooth_mi_map(mi_map,
                           sigma_scale=smooth_sigma_scale,
                           sigma_lag=smooth_sigma_lag)

        # Optionally restrict to a lag window
        if fit_window is not None:
            lo, hi = fit_window
            mask = (lags >= lo) & (lags <= hi)
            lags_fit = lags[mask]
            sm_fit   = sm[:, mask]
        else:
            lags_fit = lags
            sm_fit   = sm

        n_scales    = len(scales)
        centers     = np.full(n_scales, np.nan)
        sigma_left  = np.full(n_scales, np.nan)
        sigma_right = np.full(n_scales, np.nan)
        amplitudes  = np.full(n_scales, np.nan)
        success     = np.zeros(n_scales, dtype=bool)

        prev_params  = None
        fail_streak  = 0
        MAX_FAILS    = 5   # reset warm-start after this many consecutive failures

        # coarse → fine
        for i in range(n_scales - 1, -1, -1):
            row = sm_fit[i]

            result, ok = fit_single_ridge(
                row, lags_fit,
                prev_params=prev_params,
                center_init=center_init,
                center_range=center_range,
                sigma_init=sigma_init,
                sigma_max=sigma_max,
                amplitude_thresh=amplitude_thresh,
            )

            if ok:
                p = result.params
                centers[i]     = p['center'].value
                sigma_left[i]  = p['sigma_left'].value
                sigma_right[i] = p['sigma_right'].value
                amplitudes[i]  = p['amplitude'].value * result.row_range
                success[i]     = True
                prev_params    = result.params
                fail_streak    = 0
            else:
                fail_streak += 1
                if fail_streak >= MAX_FAILS:
                    prev_params = None   # full reset — don't keep dragging a bad estimate
                    fail_streak = 0

        return centers, sigma_left, sigma_right, amplitudes, success

    def plot_fit_at_scale(ax, sm, lags, scales, centers, sigma_left, sigma_right, success,
                          target_scale=100.0):
        """
        On ax: scatter of raw smoothed row + fitted split-Gaussian overlay at the
        scale closest to target_scale.
        """
        # find closest scale index
        i = np.argmin(np.abs(scales - target_scale))
        actual_scale = scales[i]
        row = sm[i]

        # baseline-subtract same way as during fitting
        baseline = np.percentile(row, 10)
        row_display = row - baseline

        ax.scatter(lags, row_display, s=6, color='steelblue', alpha=0.6, label='data (baseline sub.)')

        if success[i]:
            # reconstruct fitted curve
            fitted = split_gaussian(lags,
                                    amplitude=(row_display.max()),  # rescale to data
                                    center=centers[i],
                                    sigma_left=sigma_left[i],
                                    sigma_right=sigma_right[i])
            ax.plot(lags, fitted, 'r-', lw=1.8, label='fit')

            # mark center and sigmas
            ax.axvline(centers[i],                  color='white',  lw=1.0, ls='--')
            ax.axvline(centers[i] - sigma_left[i],  color='cyan',   lw=0.8, ls=':')
            ax.axvline(centers[i] + sigma_right[i], color='orange', lw=0.8, ls=':')

            ax.set_title(f"Fit at s={actual_scale:.0f}  "
                         f"μ={centers[i]:.1f}  "
                         f"σL={sigma_left[i]:.1f}  σR={sigma_right[i]:.1f}")
        else:
            ax.set_title(f"Fit at s={actual_scale:.0f} — FAILED")

        ax.set_xlabel('Lag τ')
        ax.set_ylabel('MI (baseline sub.)')
        ax.legend(fontsize=8)
        ax.set_facecolor('#1a1a2e')


    def plot_single_ridge_lmfit(mi_map, scales, lags,
                                 centers, sigma_left, sigma_right,
                                 amplitudes, success, title="",
                                 smooth_sigma_scale=2.0, smooth_sigma_lag=2.0,
                                 diagnostic_scale=100.0):
        sm = smooth_mi_map(mi_map, smooth_sigma_scale, smooth_sigma_lag)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                                 gridspec_kw={'width_ratios': [3, 1],
                                              'height_ratios': [2, 1]})

        # ── top-left: heatmap ─────────────────────────────────────────────
        ax = axes[0, 0]
        pcm = ax.pcolormesh(lags, scales, sm, cmap='viridis', shading='auto')
        ax.set_yscale('log')
        ax.set_xlabel('Lag τ')
        ax.set_ylabel('Scale s')
        ax.set_title(f"{title} — asymmetric ridge (lmfit)")
        plt.colorbar(pcm, ax=ax, label='MI')

        s_ok  = scales[success]
        c_ok  = centers[success]
        sl_ok = sigma_left[success]
        sr_ok = sigma_right[success]

        ax.plot(c_ok,         s_ok, 'w--',                lw=1.5, label='center')
        ax.plot(c_ok - sl_ok, s_ok, color='cyan',  lw=1, ls=':', label='−σ_left')
        ax.plot(c_ok + sr_ok, s_ok, color='orange', lw=1, ls=':', label='+σ_right')
        if diagnostic_scale is not None:
            # mark the diagnostic scale with a horizontal line
            ax.axhline(diagnostic_scale, color='red', lw=0.8, ls='--', alpha=0.6)
        ax.legend(fontsize=8)

        s_fail = scales[~success]
        if len(s_fail):
            ax.scatter(np.zeros(len(s_fail)), s_fail, c='red', s=4, zorder=5)

        # ── top-right: asymmetry ratio ────────────────────────────────────
        ax2 = axes[0, 1]
        ratio = sr_ok / sl_ok
        ax2.plot(ratio, s_ok, 'k-o', ms=2)
        ax2.axvline(1.0, color='gray', lw=0.8, ls='--')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('σ_right / σ_left')
        ax2.set_ylabel('Scale s')
        ax2.set_title('Asymmetry ratio')

        # ── bottom-left: single scale fit diagnostic ──────────────────────
        if diagnostic_scale is not None:
            ax3 = axes[1, 0]
            ax3.set_facecolor('#1a1a2e')
            plot_fit_at_scale(ax3, sm, lags, scales,
                              centers, sigma_left, sigma_right, success,
                              target_scale=diagnostic_scale)
        else:
            axes[1, 0].axis('off')

        # ── bottom-right: unused — can show residuals or leave blank ──────
        axes[1, 1].axis('off')

        plt.suptitle(title, fontsize=12, y=1.01)
        plt.tight_layout()
        plt.show()


    # ── entry point ────────────────────────────────────────────────────────────────

    def extract_single_ridge_and_plot(
            mi_map=None, scales=None, lags=None,
            title="",
            center_init=0.0, center_range=150.0,
            sigma_init=30.0, sigma_max=200.0,
            amplitude_thresh=0.05,
            smooth_sigma_scale=2.0, smooth_sigma_lag=2.0,
            diagnostic_scale=100.0,
            fit_window=None,      # e.g. (-200, 200) to restrict lag range
    ):
        mi_map = np.array(mi_map)

        centers, sl, sr, amp, ok = extract_single_ridge_lmfit(
            mi_map, scales, lags,
            smooth_sigma_scale=smooth_sigma_scale,
            smooth_sigma_lag=smooth_sigma_lag,
            center_init=center_init,
            center_range=center_range,
            sigma_init=sigma_init,
            sigma_max=sigma_max,
            amplitude_thresh=amplitude_thresh,
            fit_window=fit_window,
        )

        plot_single_ridge_lmfit(
            mi_map, scales, lags,
            centers, sl, sr, amp, ok,
            title=title,
            smooth_sigma_scale=smooth_sigma_scale,
            smooth_sigma_lag=smooth_sigma_lag,
            diagnostic_scale=diagnostic_scale
        )

        print(f"Successful fits: {ok.sum()} / {len(ok)}")
        return centers, sl, sr, amp, ok

    return (extract_single_ridge_and_plot,)


@app.cell
def _(SnP_results_batch_custom, extract_single_ridge_and_plot, np):
    snp_fits=[]
    scales=SnP_results_batch_custom[0]["S&P500"]["scales"]
    for _i,_ in enumerate(SnP_results_batch_custom):
        snp_fits.append(extract_single_ridge_and_plot(
            mi_map          = np.array(SnP_results_batch_custom[_i]["S&P500"]["mi_map_normalized"]),
            scales          = scales,
            lags            = np.linspace(-250, 250, 501),
            title           = "S&P500",
            center_init     = 0.0,
            center_range    = 50.0,
            sigma_init      = 20.0,
            sigma_max       = 100.0,       # tighter than before
            amplitude_thresh= 0.05,
            diagnostic_scale=None,
            fit_window      = (-100, 100), # ignore flat tails entirely
            smooth_sigma_lag= 0,smooth_sigma_scale=0
        ))
    return scales, snp_fits


@app.cell
def _(np, snp_fits):
    np.save("snp_rolling_descriptor_fits.npy",snp_fits)
    return


@app.cell
def _(np, plt, scales, snp_fits):
    def plot_average_stats():
        def compute_stats(fits):
            arr = np.array(fits)              # shape: (n_fits, n_metrics, n_scales)
            mean = arr.mean(axis=0)           # shape: (n_metrics, n_scales)
            var  = arr.var(axis=0)            # shape: (n_metrics, n_scales)
            return mean, var

        snp_mean, snp_var = compute_stats(snp_fits)

        metrics = ["ridge center", "σ_left ", "σ_right", "amplitude"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # SNP
            ax.plot(scales, snp_mean[i], label="SNP mean", color="blue")
            ax.fill_between(scales,
                            snp_mean[i] - np.sqrt(snp_var[i]),
                            snp_mean[i] + np.sqrt(snp_var[i]),
                            color="blue", alpha=0.2)

            ax.set_xscale("log")
            ax.set_title(f"{metric} comparison")
            if i == 3:
                ax.set_ylabel('amplitude')
            else:    
                ax.set_ylabel('Lag τ')
            ax.set_xlabel('scale s')
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plt.show()
        return snp_mean,snp_var
    snp_mean,snp_var=plot_average_stats()
    return


@app.cell
def _(plt, scales, snp_fits):

    components = ["ridge center", "σ_left ", "σ_right", "amplitude"]
    datasets = {
        "S&P": snp_fits
    }

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 14), sharex=True)

    for comp_idx, comp_name in enumerate(components):
        for col_idx, (label, fits) in enumerate(datasets.items()):
            ax = axes[comp_idx, col_idx]

            for _i, _result in enumerate(fits):
                ax.plot(scales, _result[comp_idx], label=f"{label} {_i}")

            ax.set_xscale("log")
            ax.set_title(f"{label} – {comp_name}")
            ax.grid(True, alpha=0.3)
            if comp_idx == 3:
                ax.set_ylabel('amplitude')
            else:    
                ax.set_ylabel('Lag τ')
            ax.set_xlabel('scale s')
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
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
    print(len(df_hist_btc))
    df_hist_btc.tail(100)
    return


@app.cell
def _(df_hist_btc):
    len(df_hist_btc[::2])
    return


@app.cell
def _(
    compare_signals,
    np,
    plot_autocorrelations,
    plot_mi_maps,
    plot_signals,
    plot_volatility_walks,
    psi_haar_smooth,
):
    def analyze_btc(df):
        n_samples = 30000
        #take every 2nd sample to have a same sampling as sandp
        log_price = np.array(df["open"].values)
        scales = np.logspace(np.log10(4), np.log10(2000), 25)
        signals = {
            # I put -n_samples: instead of :n_samples as before because the beginning of the btc signal might be not very representative
            'BTC': np.log(log_price[:n_samples])
        }
        results = compare_signals(signals, scales=scales, wavelet=psi_haar_smooth, max_level=8, window=50, ref_idx=0, max_time_lag=400, use_parallel=True, n_jobs=8)

        for name, output in results.items():
            print(f"{name}: log_vol_series shape = {output['log_vol_series'].shape}, mi_map shape = {output['mi_map'].shape}")
        plot_signals(results)
        plot_volatility_walks(results, scale_idx=3)
        plot_autocorrelations(results, scale_idx=3, max_lag=500)
        plot_mi_maps(results, max_time_lag=400)

        return results


    return (analyze_btc,)


@app.cell
def _(analyze_btc):
    results_btc = analyze_btc()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    BTC batches
    10 batches 30000 samples
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # cumulative plots
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## diff btc
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## diff S&P
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## diff btc-s&p
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
