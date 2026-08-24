import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium", auto_download=["ipynb", "html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    return gaussian_filter, np, plt


@app.cell
def _(np):
    results_snp=np.load("mi_map_snp_all.npy",allow_pickle=True)
    results_btc=np.load("mi_map_btc_all.npy",allow_pickle=True)
    results_snp_2=np.load("mi_map_snp_all_2nd.npy",allow_pickle=True)
    results_btc_2=np.load("mi_map_btc_all_2.npy",allow_pickle=True)
    return results_btc, results_btc_2, results_snp, results_snp_2


@app.cell
def _(results_snp):
    results_snp[0]["S&P500"]["mi_map"]
    return


@app.cell
def _(plt, results_snp):
    plt.imshow(results_snp[0]["S&P500"]["mi_map_normalized"],aspect="auto",extent=[-800, 800, results_snp[0]["S&P500"]["scales"][-1], results_snp[0]["S&P500"]["scales"][0]])
    plt.title("S&P500")
    plt.xlabel('Time lag')
    plt.ylabel('Scale')
    plt.yscale('log')
    plt.colorbar(label='MI')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # gaussian smoothing
    \( G(s,\tau) = \exp\left(-\frac{s^2}{2\sigma_{\text{scale}}^2} - \frac{\tau^2}{2\sigma_{\text{lag}}^2}\right) \)
    """)
    return


@app.cell
def _(gaussian_filter):
    def smooth_mi_map(mi_map, sigma_scale=1.0, sigma_lag=2.0, alpha=0.0):
        """    alpha = regularisation strength (0 = none)
        """
        smoothed = gaussian_filter(mi_map, sigma=[sigma_scale, sigma_lag])

        if alpha > 0:
            # Tikhonov regularisation: (I + αL)^(-1) M
            # Here: simple shrinkage toward smoothness
            smoothed = (1 - alpha) * smoothed + alpha * gaussian_filter(smoothed, sigma=3)

        return smoothed

    return (smooth_mi_map,)


@app.cell
def _(gaussian_filter, np, plt):
    def example_smoothing(sigmas = [
            (0, 0),
            (1.0, 0),
            (0, 1.0),
            (1, 1.0),
        ]):
        # Create synthetic 2D field
        scales = np.linspace(1, 20, 100)
        lags = np.linspace(-200, 200, 300)

        S, L = np.meshgrid(scales, lags, indexing="ij")

        # True ridge: a simple curve
        true_ridge = 0.5 * (S - 10)

        # Base field: Gaussian bump around the ridge
        field = np.exp(-0.5 * ((L - true_ridge) / 10)**2)

        # Add noise
        np.random.seed(0)
        noisy_field = field + 0.3 * np.random.randn(*field.shape)


        plt.imshow(noisy_field,aspect='auto',extent=[lags[0], lags[-1], scales[-1], scales[0]])
        plt.show()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        for ax, (s_scale, s_lag) in zip(axes.flat, sigmas):
            sm = gaussian_filter(noisy_field, sigma=[s_scale, s_lag])
            sm = ax.imshow(sm, aspect="auto", extent=[lags[0], lags[-1], scales[-1], scales[0]], cmap="viridis")
            ax.set_title(f"σ_scale (y axis)={s_scale}, σ_lag (x axis)={s_lag}")
            ax.set_xlabel("Lag")
            ax.set_ylabel("Scale")
            fig.colorbar(sm,label="Z")

        plt.tight_layout()
        plt.show()
    example_smoothing()
    return (example_smoothing,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## selective smoothing
    notice how signal amplitude is affected as well
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    smoothing only x
    """)
    return


@app.cell
def _(example_smoothing):
    example_smoothing(sigmas = [(0,0.2),(0,0.4),(0,0.6),(0,0.8)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    smoothing only y
    """)
    return


@app.cell
def _(example_smoothing):
    example_smoothing(sigmas = [(0.2,0),(0.4,0),(0.6,0),(0.8,0)])
    return


@app.cell
def _():
    # input_map = results_snp[0]["S&P500"]["mi_map_normalized"]
    # sigmas = [(0.8, 1.5), (1.0, 2.0), (1.2, 2.5)]
    # ridges = []
    # plt.imshow(input_map ,aspect="auto",extent=[-800, 800, results_snp[0]["S&P500"]["scales"][-1], results_snp[0]["S&P500"]["scales"][0]])
    # plt.title("S&P500 input")
    # plt.xlabel('Time lag')
    # plt.ylabel('Scale')
    # plt.colorbar(label='MI')
    # plt.show()
    # for s_scale, s_lag in sigmas:
    #     sm = smooth_mi_map(input_map, s_scale, s_lag)
    #     plt.imshow(sm ,aspect="auto",extent=[-800, 800, results_snp[0]["S&P500"]["scales"][-1], results_snp[0]["S&P500"]["scales"][0]])
    #     plt.title("S&P500 smoothed")
    #     plt.xlabel('Time lag')
    #     plt.ylabel('Scale')
    #     plt.colorbar(label='MI')
    #     plt.show()
    return


@app.cell
def _(np, plt):
    def plot_raw_mi_map(input_map, scales, lags,title="MI map"):
        # --- 1. Plot raw input ---
        plt.figure(figsize=(12, 5))
        plt.imshow(
            input_map,
            interpolation=None,
            aspect="auto",
            extent=[lags[0], lags[-1], scales[-1], scales[0]],

        )
        plt.title(title)
        plt.xlabel("Time lag")
        plt.ylabel("Scale")
        plt.yscale('log')
        plt.colorbar(label="MI")
        plt.tight_layout()
        plt.show()

    def plot_comparison_with_ridges(input_map, scales, sigmas, lags, smooth_fn):
        """
        input_map : 2D MI map (scales × lags)
        scales    : array of scale values (top→bottom)
        sigmas    : list of (sigma_scale, sigma_lag)
        lags      : array of lag values
        smooth_fn : smoothing function, e.g. smooth_mi_map
        """
        plot_raw_mi_map(input_map, scales,lags,"S&P500 — Raw MI map")


        # --- 2. Plot smoothed versions ---
        for (s_scale, s_lag) in sigmas:
            if s_scale == 0 and s_lag ==0:
                sm = input_map
            else:
                sm = smooth_fn(input_map, s_scale, s_lag)
            ridge = np.argmax(sm, axis=1)  # ridge index per scale
            print(ridge)
            ridge_lags = lags[ridge]

            plt.figure(figsize=(12, 5))
            plt.imshow(
                sm,
                interpolation=None,
                aspect="auto",
                extent=[lags[0], lags[-1], scales[-1], scales[0]],
            )
            plt.plot(ridge_lags, scales, color="red", linewidth=2, label="Ridge")
            if s_scale == 0 and s_lag ==0:
                plt.title(f"S&P500 — no smoothing")

            else:
                plt.title(f"S&P500 — Smoothed (σ_scale={s_scale}, σ_lag={s_lag})")

            plt.xlabel("Time lag")
            plt.ylabel("Scale")
            plt.colorbar(label="MI")
            plt.yscale('log')
            plt.legend()
            plt.tight_layout()
            plt.show()


    return (plot_raw_mi_map,)


@app.cell
def _(np, results_snp):
    scales = results_snp[0]["S&P500"]["scales"]
    lags = np.linspace(-800, 800, 801)  # or your actual lag array
    return lags, scales


@app.cell
def _():
    # input_map = results_snp[0]["S&P500"]["mi_map"]
    # scales = results_snp[0]["S&P500"]["scales"]
    # lags = np.linspace(-800, 800, 801)  # or your actual lag array

    # sigmas = [(0,0),(0.8, 1.5), (1.0, 2.0), (1.2, 2.5)]

    # plot_comparison_with_ridges(
    #     input_map=input_map,
    #     scales=scales,
    #     sigmas=sigmas,
    #     lags=lags,
    #     smooth_fn=smooth_mi_map
    # )
    return


@app.cell
def _(input_map, np, plt):
    plt.imshow(input_map,aspect='auto')
    plt.colorbar(label="MI")
    plt.show()

    plt.imshow(input_map,aspect='auto',vmin=0.8,vmax=1.2)
    plt.colorbar(label="MI")
    plt.show()
    for _i in [10,21,23]:
        print(f"ind {_i}, max val {np.max(input_map[_i])} max ind {np.argmax(input_map[_i])}")
        plt.plot(input_map[_i])
        plt.show()

    for _i in range(1,25):
        plt.plot(input_map[_i])
    plt.yscale('log')
    plt.show()
    return


@app.cell
def _(input_map, np, plt, smooth_mi_map):
    plt.imshow(smooth_mi_map(input_map,0.8,1.5),aspect='auto')
    plt.colorbar(label="MI")
    plt.show()
    plt.imshow(smooth_mi_map(input_map,0.8,1.5),aspect='auto',vmin=0.8,vmax=1.2)
    plt.colorbar(label="MI")
    plt.show()
    for i in [10,21,23]:
        print(f"ind {i}, max val {np.max(smooth_mi_map(input_map,0.8,1.5)[i])} max ind {np.argmax(smooth_mi_map(input_map,0.8,1.5)[i])}")
        plt.plot(smooth_mi_map(input_map,0.8,1.5)[i])
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # smoothing vs ridge
    """)
    return


@app.cell
def _(np, plt):
    def plot_sigma_grid_with_ridges(input_map, scales, lags, sigma_pairs, smooth_fn):
        """
        input_map   : 2D MI map (scales × lags)
        scales      : array of scale values (top→bottom)
        lags        : array of lag values
        sigma_pairs : list of 16 (σ_scale, σ_lag) tuples
        smooth_fn   : smoothing function, e.g. smooth_mi_map
        """

        assert len(sigma_pairs) == 16, "Provide exactly 16 sigma pairs."

        fig, axes = plt.subplots(4, 4, figsize=(18, 18))

        for ax, (s_scale, s_lag) in zip(axes.flat, sigma_pairs):

            # Apply smoothing
            if s_scale == 0 and s_lag == 0:
                sm = input_map
                title = "σ_scale=0, σ_lag=0 (raw)"
            else:
                sm = smooth_fn(input_map, s_scale, s_lag)
                title = f"σ_scale={s_scale}, σ_lag={s_lag}"

            # Ridge extraction
            ridge_idx = np.argmax(sm, axis=1)
            ridge_lags = lags[ridge_idx]

            # Plot MI map
            ax.imshow(
                sm,
                interpolation=None,
                aspect="auto",
                extent=[lags[0], lags[-1], scales[-1], scales[0]],
                cmap="viridis"
            )

            # Plot ridge
            ax.plot(ridge_lags, scales, color="red", linewidth=1.5)

            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Lag")
            ax.set_ylabel("Scale")
            ax.set_yscale('log')


        plt.tight_layout()
        plt.show()


    return (plot_sigma_grid_with_ridges,)


@app.cell
def _(np, plot_sigma_grid_with_ridges, results_snp, smooth_mi_map):
    sigma_pairs = [
        (0,0), (0.2,0.5), (0.2,1.0), (0.2,2.0),
        (0.5,0.2), (0.5,0.5), (0.5,1.0), (0.5,2.0),
        (1.0,0.2), (1.0,0.5), (1.0,1.0), (1.0,2.0),
        (2.0,0.2), (2.0,0.5), (2.0,1.0), (2.0,2.0),
    ]
    plot_sigma_grid_with_ridges(
        input_map=results_snp[0]["S&P500"]["mi_map"],
        scales=results_snp[0]["S&P500"]["scales"],
        lags=np.linspace(-800, 800, 400+400+1),
        sigma_pairs=sigma_pairs,
        smooth_fn=smooth_mi_map
    )
    return (sigma_pairs,)


@app.cell
def _(
    np,
    plot_sigma_grid_with_ridges,
    results_snp,
    sigma_pairs,
    smooth_mi_map,
):
    plot_sigma_grid_with_ridges(
        input_map=results_snp[0]["S&P500"]["mi_map_normalized"],
        scales=results_snp[0]["S&P500"]["scales"],
        lags=np.linspace(-800, 800, 400+400+1),
        sigma_pairs=sigma_pairs,
        smooth_fn=smooth_mi_map
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # smoothing vs ridge width
    """)
    return


@app.cell
def _(np, plt):

    def ridge_widths_curvature(smoothed_mi, lags):
        """
        Curvature-based width estimate per scale.
        Assumes local Gaussian shape around ridge in lag.
        """
        n_scales, n_lags = smoothed_mi.shape
        widths = np.full(n_scales, np.nan)

        # finite difference step in lag (assumes uniform grid)
        dτ = lags[1] - lags[0]

        # ridge index per scale
        ridge_idx = np.argmax(smoothed_mi, axis=1)

        for i in range(n_scales):
            j = ridge_idx[i]

            # need neighbors on both sides
            if j <= 0 or j >= n_lags - 1:
                continue

            row = smoothed_mi[i, :]

            # avoid log(0)
            eps = 1e-12
            f_minus = np.log(row[j - 1] + eps)
            f_0     = np.log(row[j]     + eps)
            f_plus  = np.log(row[j + 1] + eps)

            # second derivative of log M wrt lag (central difference)
            d2_logM = (f_plus - 2 * f_0 + f_minus) / (dτ**2)

            if d2_logM >= 0:  # not a proper maximum / bad curvature
                continue

            sigma = 1.0 / np.sqrt(-d2_logM)
            widths[i] = sigma

        return widths



    def plot_sigma_grid_with_ridges_and_widths(input_map, scales, lags, sigma_pairs, smooth_fn):
        """
        4×4 grid: MI map + ridge + ridge width overlay.
        """
        assert len(sigma_pairs) == 16, "Provide exactly 16 sigma pairs."

        fig, axes = plt.subplots(4, 4, figsize=(18, 18))

        for ax, (s_scale, s_lag) in zip(axes.flat, sigma_pairs):

            # Smoothing
            if s_scale == 0 and s_lag == 0:
                sm = input_map
                title = "raw (σ=0,0)"
            else:
                sm = smooth_fn(input_map, s_scale, s_lag)
                title = f"σ_scale={s_scale}, σ_lag={s_lag}"

            # Ridge
            ridge_idx = np.argmax(sm, axis=1)
            ridge_lags = lags[ridge_idx]

            # Widths
            widths = ridge_widths_curvature(sm, lags)

            # Plot MI map
            ax.imshow(
                sm,
                interpolation=None,
                aspect="auto",
                extent=[lags[0], lags[-1], scales[-1], scales[0]],
                cmap="viridis"
            )

            # Plot ridge
            ax.plot(ridge_lags, scales, color="red", linewidth=1.5)

            # Plot width bars
            for i, (lag_center, w) in enumerate(zip(ridge_lags, widths)):
                if np.isnan(w):
                    continue
                s = scales[i]
                ax.hlines(s, lag_center - w, lag_center + w, colors="white", linewidth=1)

            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Lag")
            ax.set_ylabel("Scale")
            ax.set_yscale('log')
        plt.tight_layout()
        plt.show()


    return (plot_sigma_grid_with_ridges_and_widths,)


@app.cell
def _(
    np,
    plot_sigma_grid_with_ridges_and_widths,
    results_snp,
    sigma_pairs,
    smooth_mi_map,
):
    plot_sigma_grid_with_ridges_and_widths(
        input_map=np.array(results_snp[0]["S&P500"]["mi_map"]),
        scales=results_snp[0]["S&P500"]["scales"],
        lags=np.linspace(-800, 800, 400+400+1),
        sigma_pairs=sigma_pairs,
        smooth_fn=smooth_mi_map
    )
    return


@app.cell
def _(
    np,
    plot_sigma_grid_with_ridges_and_widths,
    results_snp,
    sigma_pairs,
    smooth_mi_map,
):
    plot_sigma_grid_with_ridges_and_widths(
        input_map=np.array(results_snp[0]["S&P500"]["mi_map_normalized"]),
        scales=results_snp[0]["S&P500"]["scales"],
        lags=np.linspace(-800, 800, 400+400+1),
        sigma_pairs=sigma_pairs,
        smooth_fn=smooth_mi_map
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # building spline field
    """)
    return


@app.cell
def _(np, plt):
    from scipy.interpolate import RectBivariateSpline

    def build_field(smoothed, scales, lags):
        return RectBivariateSpline(scales, lags, smoothed,kx=3,ky=3)


    def plot_spline_field(M, scales, lags, upsample_scale=4, upsample_lag=4,ridge=None, title="Spline-interpolated field M(s, τ)"):
        # Build dense grid
        s_dense = np.linspace(scales.min(), scales.max(), len(scales)*upsample_scale)
        t_dense = np.linspace(lags.min(), lags.max(), len(lags)*upsample_lag)

        S, T = np.meshgrid(s_dense, t_dense, indexing="ij")

        # Evaluate spline on dense grid
        Z = M(s_dense, t_dense)

        plt.figure(figsize=(10, 6))
        plt.imshow(
            Z,
            aspect="auto",
            extent=[t_dense[0], t_dense[-1], s_dense[-1], s_dense[0]],
            cmap="viridis"
        )
        if ridge is not None:
            plt.plot(ridge, scales, color="red", linewidth=2, label="Ridge")

        plt.colorbar(label="M(s, τ)")
        plt.xlabel("Lag τ")
        plt.ylabel("Scale s")
        plt.yscale('log')
        plt.title(title)
        plt.tight_layout()
        plt.show()


    return RectBivariateSpline, build_field, plot_spline_field


@app.cell
def _(build_field, np, results_snp, smooth_mi_map):
    sm = smooth_mi_map(np.array(results_snp[0]["S&P500"]["mi_map_normalized"]), 0.5,2)

    M = build_field(sm,
                    scales=results_snp[0]["S&P500"]["scales"],
                    lags=np.linspace(-800, 800, 400+400+1))
    return M, sm


@app.cell
def _(M, np, plot_raw_mi_map, plot_spline_field, results_snp):
    plot_raw_mi_map(np.array(results_snp[0]["S&P500"]["mi_map_normalized"]),
                    scales=results_snp[0]["S&P500"]["scales"],
                    lags=np.linspace(-800, 800, 400+400+1),title="Raw input MI map S&P500")
    plot_spline_field(M,
                    scales=results_snp[0]["S&P500"]["scales"],
                    lags=np.linspace(-800, 800, 400+400+1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    reading specific points
    """)
    return


@app.cell
def _(M):
    print(M(0,0))
    print(M(25,range(0,100)))
    return


@app.cell
def _(M, np, plot_spline_field, results_snp):
    for dx,dy in [(1,0),(2,0),(0,1),(0,2)]:
        dM = lambda x,y: M(x,y,dx=dx,dy=dy)
        plot_spline_field(dM,
                        scales=results_snp[0]["S&P500"]["scales"],
                        lags=np.linspace(-800, 800, 400+400+1),title=f"Spline-interpolated field M(s, τ) ({dx} derivative in x, {dy} derivative in y)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## smoothing impact on spline
    """)
    return


@app.cell
def _(RectBivariateSpline, np, plt):


    def plot_sigma_grid_spline_error(input_map, scales, lags, sigma_pairs, smooth_fn):
        """
        4×4 grid: spline interpolation error for different smoothings.

        input_map   : 2D MI map (scales × lags)
        scales      : 1D array of scales (len = n_scales)
        lags        : 1D array of lags   (len = n_lags)
        sigma_pairs : list of 16 (σ_scale, σ_lag)
        smooth_fn   : function(input_map, σ_scale, σ_lag) -> smoothed_map
        """
        assert len(sigma_pairs) == 16

        fig, axes = plt.subplots(4, 4, figsize=(18, 18),constrained_layout=True)

        for ax, (s_scale, s_lag) in zip(axes.flat, sigma_pairs):

            # 1) Smooth
            if s_scale == 0 and s_lag == 0:
                sm = input_map
                title = "raw (σ=0,0)"
            else:
                sm = smooth_fn(input_map, s_scale, s_lag)
                title = f"σ_scale={s_scale}, σ_lag={s_lag}"

            # 2) Fit spline on smoothed field
            M = RectBivariateSpline(scales, lags, sm,kx=2,ky=2)

            # 3) Evaluate spline back on original grid
            sm_spline = M(scales, lags)  # shape (n_scales, n_lags)

            # 4) Error field
            err = np.abs(sm_spline - sm)

            im = ax.imshow(
                err,
                aspect="auto",
                extent=[lags[0], lags[-1], scales[-1], scales[0]],
                cmap="magma",
                interpolation=None
            )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Lag")
            ax.set_ylabel("Scale")
            ax.set_yscale('log')

        fig.colorbar(im, ax=axes.ravel().tolist(), label="|M_spline - M_smoothed|")
        plt.show()


    return (plot_sigma_grid_spline_error,)


@app.cell
def _(
    np,
    plot_sigma_grid_spline_error,
    results_snp,
    sigma_pairs,
    smooth_mi_map,
):
    plot_sigma_grid_spline_error(
        input_map=np.array(results_snp[0]["S&P500"]["mi_map_normalized"]),
        scales=results_snp[0]["S&P500"]["scales"],
        lags=np.linspace(-800, 800, 801),
        sigma_pairs=sigma_pairs,
        smooth_fn=smooth_mi_map
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # identifying ridge from continous surface
    """)
    return


@app.cell
def _(np):
    from scipy.optimize import brentq

    def find_ridge_lag(M, s, lag_min, lag_max, n_samples=200):
        # sample derivative to find sign changes
        lags = np.linspace(lag_min, lag_max, n_samples)
        dM = np.array([M(s, τ, dx=0, dy=1)[0][0] for τ in lags])

        # find zero crossings
        idx = np.where(np.diff(np.sign(dM)))[0]
        if len(idx) == 0:
            return None

        candidates = []
        for i in idx:
            a, b = lags[i], lags[i+1]
            try:
                τ0 = brentq(lambda τ: M(s, τ, dx=0, dy=1)[0][0], a, b)
                curv = M(s, τ0, dx=0, dy=2)[0][0]
                if curv < 0:  # must be a maximum
                    candidates.append((τ0, curv))
            except ValueError:
                pass

        if not candidates:
            return None

        # choose the strongest maximum (most negative curvature)
        τ_best, _ = min(candidates, key=lambda x: x[1])
        return τ_best

    def extract_ridge(M, scales, lag_min, lag_max):
        ridge = []
        for s in scales:
            τ = find_ridge_lag(M, s, lag_min, lag_max)
            ridge.append(τ)
        return np.array(ridge)


    return (extract_ridge,)


@app.cell
def _(np, plt):
    def plot_ridge(input_map, scales, lags, ridges, title="Ridge",widths=None):
        plt.figure(figsize=(10, 6))
        plt.imshow(
            input_map,
            aspect="auto",
            extent=[lags[0], lags[-1], scales[-1], scales[0]],
            cmap="viridis",
            interpolation=None
        )
        for ind,ridge in enumerate(ridges):
            plt.plot(ridge, scales, lw=2, ls=':', label=f"Ridge {ind}")

            if widths is not None:
            # unpack tuple widths
                left = np.array([w[0] for w in widths[ind]])
                right = np.array([w[1] for w in widths[ind]])

                # left boundary
                plt.plot(ridge - left, scales, lw=1, ls='--', color='white')

                # right boundary
                plt.plot(ridge + right, scales, lw=1, ls='--', color='white')
        plt.xlabel("Lag τ")
        plt.ylabel("Scale s")
        plt.yscale("log")
        plt.title(title)
        plt.colorbar(label="MI")
        plt.legend()
        plt.tight_layout()
        plt.show()


    return (plot_ridge,)


@app.cell
def _(M, extract_ridge, results_snp):
    # 3. Extract ridge
    ridge_lags = extract_ridge(M, results_snp[0]["S&P500"]["scales"], -800, 800)
    return (ridge_lags,)


@app.cell
def _(M, np, plot_ridge, plot_spline_field, results_snp, ridge_lags, sm):
    plot_ridge(sm,
               scales=results_snp[0]["S&P500"]["scales"],
               lags=np.linspace(-800, 800, 400+400+1),
               ridges=[ridge_lags],
               title="Derivative-based ridge")
    plot_spline_field(M,
                    scales=results_snp[0]["S&P500"]["scales"],
                    lags=np.linspace(-800, 800, 400+400+1),ridge=ridge_lags)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## finite difference partial derivatives
    """)
    return


@app.cell
def _(np, plt):
    def finite_diff_derivatives(smoothed, scales, lags):
        """
        Compute first and second derivatives wrt lag using central differences.
        smoothed: 2D array (n_scales × n_lags)
        """
        dτ = lags[1] - lags[0]

        # First derivative wrt lag (axis=1)
        dM_dlag = np.zeros_like(smoothed)
        dM_dlag[:, 1:-1] = (smoothed[:, 2:] - smoothed[:, :-2]) / (2 * dτ)

        # Second derivative wrt lag
        d2M_dlag2 = np.zeros_like(smoothed)
        d2M_dlag2[:, 1:-1] = (smoothed[:, 2:] - 2*smoothed[:, 1:-1] + smoothed[:, :-2]) / (dτ**2)

        return dM_dlag, d2M_dlag2

    def plot_field_array(Z, scales, lags, title="Field"):
        plt.figure(figsize=(10, 6))
        plt.imshow(
            Z,
            aspect="auto",
            extent=[lags[0], lags[-1], scales[-1], scales[0]],
            cmap="viridis"
        )
        plt.colorbar(label=title)
        plt.xlabel("Lag τ")
        plt.ylabel("Scale s")
        plt.yscale('log')
        plt.title(title)
        plt.tight_layout()
        plt.show()



    return finite_diff_derivatives, plot_field_array


@app.cell
def _(
    finite_diff_derivatives,
    lags,
    np,
    plot_field_array,
    results_snp,
    scales,
    sm,
):
    # Compute finite-difference derivatives
    dM_dlag, d2M_dlag2 = finite_diff_derivatives(sm, results_snp[0]["S&P500"]["scales"], np.linspace(-800, 800, 801))

    # Plot raw smoothed field
    plot_field_array(sm, scales, lags, title="Smoothed MI")

    # Plot first derivative wrt lag
    plot_field_array(dM_dlag, scales, lags, title="∂M/∂τ (finite diff)")

    # Plot second derivative wrt lag
    plot_field_array(d2M_dlag2, scales, lags, title="∂²M/∂τ² (finite diff)")
    return


@app.cell
def _(finite_diff_derivatives, plt):
    def plot_field(ax, Z, scales, lags, title, cmap="viridis"):
        ax.imshow(
            Z,
            aspect="auto",
            extent=[lags[0], lags[-1], scales[-1], scales[0]],
            cmap=cmap,
            interpolation=None
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Lag τ")
        ax.set_ylabel("Scale s")
        ax.set_yscale("log")

    def plot_sigma_grid_finite_diff(input_map, scales, lags, sigma_pairs, smooth_fn):
        assert len(sigma_pairs) == 16

        fig, axes = plt.subplots(16, 3, figsize=(18, 48), constrained_layout=True)

        for row, (σs, σt) in enumerate(sigma_pairs):

            # 1) Smooth MI map
            if σs == 0 and σt == 0:
                sm = input_map
                title = "raw (σ=0,0)"
            else:
                sm = smooth_fn(input_map, σs, σt)
                title = f"σ_scale={σs}, σ_lag={σt}"

            # 2) Finite-difference derivatives
            dM, d2M = finite_diff_derivatives(sm, scales, lags)

            # 3) Plot fields
            plot_field(axes[row, 0], sm, scales, lags, title="Smoothed MI\n" + title)
            plot_field(axes[row, 1], dM, scales, lags, title="∂M/∂τ (finite diff)")
            plot_field(axes[row, 2], d2M, scales, lags, title="∂²M/∂τ² (finite diff)", cmap="magma")

        plt.show()


    return (plot_sigma_grid_finite_diff,)


@app.cell
def _(
    np,
    plot_sigma_grid_finite_diff,
    results_btc,
    results_snp,
    sigma_pairs,
    smooth_mi_map,
):
    plot_sigma_grid_finite_diff(
        input_map=np.array(results_btc[0]["BTC"]["mi_map_normalized"]),
        scales=results_snp[0]["S&P500"]["scales"],
        lags=np.linspace(-800, 800, 801),
        sigma_pairs=sigma_pairs,
        smooth_fn=smooth_mi_map
    )
    return


@app.cell
def _(np, plt):
    from scipy.signal import argrelextrema

    def extract_ridge_finite_diff(smoothed, dM, d2M, scales, lags):
        ridge = np.full(len(scales), np.nan)

        for i, s in enumerate(scales):
            dM_row = dM[i]
            d2M_row = d2M[i]

            # zero-crossings of dM
            idx = np.where(np.diff(np.sign(dM_row)))[0]
            if len(idx) == 0:
                continue

            candidates = []
            for j in idx:
                τ = lags[j]
                if d2M_row[j] < 0:   # true maximum
                    candidates.append((τ, d2M_row[j]))

            if not candidates:
                continue

            # pick strongest maximum (most negative curvature)
            τ_best, _ = min(candidates, key=lambda x: x[1])
            ridge[i] = τ_best
            print(len(candidates))
            plt.plot(dM_row)
            plt.plot(d2M_row)
            plt.title(f"{s}")
            plt.show()
        return ridge

    def quad_interp(y, j):
        a, b, c = y[j-1], y[j], y[j+1]
        return j + 0.5 * (a - c) / (a - 2*b + c)

    def extract_topN_ridges(smoothed, d2M, scales, lags, N=3,symetric=None):
        S, T = smoothed.shape
        ridges = np.full((N, S), np.nan)

        # For each scale, collect all candidates
        all_candidates = []
        for i in range(S):
            row = smoothed[i]
            d2 = d2M[i]

            idx = argrelextrema(row, np.greater)[0]
            candidates = []

            for j in idx:
                if j == 0 or j == T-1:
                    continue

                τ_idx = quad_interp(row, j)  # sub-sample index
                τ = np.interp(τ_idx, np.arange(len(lags)), lags)  # convert to lag

                score = -d2[j]  # sharper peak = better

                if score > 0:
                    candidates.append((τ, score))

            # sort by score descending
            candidates.sort(key=lambda x: x[1], reverse=True)
            all_candidates.append(candidates)

        # Track N ridges across scales
        for k in range(N):
            prev_tau = None

            for i in range(S):
                candidates = all_candidates[i]
                if not candidates:
                    continue

                if prev_tau is None:
                    # first scale: pick k-th best if exists
                    if len(candidates) > k:
                        τ = candidates[k][0]
                    else:
                        τ = candidates[-1][0]
                else:
                    # pick candidate closest to previous τ
                    τ = min(candidates, key=lambda x: abs(x[0] - prev_tau))[0]

                ridges[k, i] = τ
                prev_tau = τ

        return ridges,None

    def zero_crossing_pos(x0, x1, y0, y1):
        # linear interpolation of zero crossing between (x0, y0) and (x1, y1)
        if y0 == y1:
            return 0.5 * (x0 + x1)
        return x0 - y0 * (x1 - x0) / (y1 - y0)

    def ridge_width_from_d2(d2_row, lags, j):
        n = len(d2_row)

        # --- left side: find where d2 crosses from <0 to >=0 ---
        jl = j
        while jl > 0 and d2_row[jl] < 0:
            jl -= 1

        if jl == j:  # no negative region to the left
            left_lag = lags[j]
        else:
            # interpolate zero crossing between jl and jl+1
            x0, x1 = jl, jl + 1
            y0, y1 = d2_row[x0], d2_row[x1]
            zc_idx_left = zero_crossing_pos(x0, x1, y0, y1)
            left_lag = np.interp(zc_idx_left, np.arange(n), lags)

        # --- right side: find where d2 crosses from <0 to >=0 ---
        jr = j
        while jr < n - 1 and d2_row[jr] < 0:
            jr += 1

        if jr == j:  # no negative region to the right
            right_lag = lags[j]
        else:
            x0, x1 = jr - 1, jr
            y0, y1 = d2_row[x0], d2_row[x1]
            zc_idx_right = zero_crossing_pos(x0, x1, y0, y1)
            right_lag = np.interp(zc_idx_right, np.arange(n), lags)

        width_total = right_lag - left_lag
        width_left = lags[j] - left_lag
        width_right = right_lag - lags[j]

        return width_total, width_left, width_right

    def extract_topN_ridges_with_width(smoothed, d2M, scales, lags, N=3,symetric=True):
        S, T = smoothed.shape
        ridges = np.full((N, S), np.nan)
        widths = np.empty((N, S), dtype=object)
        widths[:] = None


        # Precompute local dx for lag conversion
        lag_idx = np.arange(len(lags))

        def local_dx(j):
            if j == 0:
                return lags[1] - lags[0]
            elif j == len(lags) - 1:
                return lags[-1] - lags[-2]
            else:
                return 0.5 * ((lags[j+1] - lags[j]) + (lags[j] - lags[j-1]))

        # Collect all candidates per scale
        all_candidates = []
        for i in range(S):
            row = smoothed[i]
            d2 = d2M[i]

            idx = argrelextrema(row, np.greater)[0]
            candidates = []

            for j in idx:
                if j == 0 or j == T - 1:
                    continue

                # sub-sample index
                τ_idx = quad_interp(row, j)

                # convert index → lag
                τ = np.interp(τ_idx, lag_idx, lags)

                score = -d2[j]  # curvature score

                if score > 0:
                    # store τ, score, τ_idx, j
                    candidates.append((τ, score, τ_idx, j))

            candidates.sort(key=lambda x: x[1], reverse=True)
            all_candidates.append(candidates)

        # Track N ridges across scales
        for k in range(N):
            prev_tau = None

            for i in range(S):
                candidates = all_candidates[i]
                if not candidates:
                    continue

                if prev_tau is None:
                    # first scale: pick k-th best if exists
                    if len(candidates) > k:
                        τ, score, τ_idx, j = candidates[k]
                    else:
                        τ, score, τ_idx, j = candidates[-1]
                else:
                    # pick candidate closest to previous τ
                    τ, score, τ_idx, j = min(
                        candidates, key=lambda x: abs(x[0] - prev_tau)
                    )

                ridges[k, i] = τ
                prev_tau = τ

                # compute width from curvature
                if symetric:
                    d2_val = d2M[i, j]
                    if d2_val < 0:
                        dx = local_dx(j)
                        width = np.sqrt(-2.0 / d2_val) * dx
                        widths[k, i] = width
                else:
                    d2_row = d2M[i]
                    width_total, width_left, width_right = ridge_width_from_d2(d2_row, lags, j)
                    widths[k, i] = (width_left, width_right)
                    # optionally store left/right if you want


        return ridges, widths


    return extract_topN_ridges, extract_topN_ridges_with_width


@app.cell
def _(
    extract_topN_ridges,
    finite_diff_derivatives,
    lags,
    np,
    plot_ridge,
    results_btc,
    results_snp,
    scales,
    smooth_mi_map,
):
    # Compute finite-difference derivatives
    def extract_n_ridges_and_plot(input=np.array(results_snp[0]["S&P500"]["mi_map_normalized"]),ridge_algorithm=extract_topN_ridges,n_ridges=3,plot_derrivatives=False,title="",symetric=False):

        sm = smooth_mi_map(input, 2,2)


        dM_dlag, d2M_dlag2 = finite_diff_derivatives(sm, results_snp[0]["S&P500"]["scales"], np.linspace(-800, 800, 801))

        # ridge = extract_ridge_finite_diff(sm, dM_dlag, d2M_dlag2, scales, lags)

        ridges,widths = ridge_algorithm(input, d2M_dlag2, scales, lags,n_ridges,symetric)
        if plot_derrivatives:
            # Plot first derivative wrt lag
            plot_ridge(dM_dlag, scales, lags,ridges, title=f"{title} ∂M/∂τ (finite diff)",widths=widths)

            # Plot second derivative wrt lag
            plot_ridge(d2M_dlag2, scales, lags,ridges, title=f"{title} ∂²M/∂τ² (finite diff)",widths=widths)

        plot_ridge(sm,scales,lags,ridges,f"{title} smoothed map",widths=widths)
    extract_n_ridges_and_plot(np.array(results_snp[0]["S&P500"]["mi_map_normalized"]),extract_topN_ridges,7,plot_derrivatives=True,title="S&P500")
    extract_n_ridges_and_plot(np.array(results_btc[0]["BTC"]["mi_map_normalized"]),extract_topN_ridges,7,plot_derrivatives=True,title="BTC")
    return (extract_n_ridges_and_plot,)


@app.cell
def _(
    extract_n_ridges_and_plot,
    extract_topN_ridges_with_width,
    np,
    results_btc,
    results_snp,
):
    extract_n_ridges_and_plot(np.array(results_snp[0]["S&P500"]["mi_map_normalized"]),extract_topN_ridges_with_width,3,plot_derrivatives=True,title="S&P500",symetric=False)
    extract_n_ridges_and_plot(np.array(results_btc[0]["BTC"]["mi_map_normalized"]),extract_topN_ridges_with_width,3,plot_derrivatives=True,title="BTC",symetric=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # fit gaussians
    """)
    return


@app.cell(hide_code=True)
def _():
    # from lmfit import Model
    # def split_gaussian(x, amplitude, center, sigma_left, sigma_right):
    #     sigma = np.where(x < center, sigma_left, sigma_right)
    #     return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)

    # def fit_single_ridge(row, lags, prev_params=None,
    #                      center_init=0.0, center_range=150.0,
    #                      sigma_init=30.0, sigma_min=2.0, sigma_max=400.0,
    #                      amplitude_thresh=0.05):
    #     """
    #     Fit a single asymmetric Gaussian to one scale row.

    #     Key fix: subtract row baseline before fitting so the peak
    #     is defined relative to background, not absolute MI level.
    #     """
    #     model = Model(split_gaussian)

    #     # ── baseline subtraction ──────────────────────────────────────────
    #     # Use a percentile rather than min to be robust to outliers
    #     baseline = np.percentile(row, 10)
    #     row_centered = row - baseline
    #     row_range = row_centered.max()

    #     # Reject flat rows early — nothing to fit
    #     if row_range < amplitude_thresh:
    #         return None, False

    #     # Normalise to [0,1] for stable fitting, rescale result after
    #     row_norm = row_centered / row_range

    #     # ── params ───────────────────────────────────────────────────────
    #     if prev_params is not None:
    #         params = prev_params.copy()
    #         params['center'].set(
    #             min=params['center'].value - center_range,
    #             max=params['center'].value + center_range,
    #         )
    #         # Reset amplitude to current row's scale
    #         params['amplitude'].set(value=1.0, min=amplitude_thresh, max=1.5)
    #     else:
    #         params = model.make_params(
    #             amplitude=dict(value=1.0, min=amplitude_thresh, max=1.5),
    #             center=dict(value=center_init,
    #                         min=center_init - center_range,
    #                         max=center_init + center_range),
    #             sigma_left=dict(value=sigma_init, min=sigma_min, max=sigma_max),
    #             sigma_right=dict(value=sigma_init, min=sigma_min, max=sigma_max),
    #         )

    #     try:
    #         result = model.fit(row_norm, params, x=lags)
    #     except Exception:
    #         return None, False

    #     # ── quality checks ────────────────────────────────────────────────
    #     p = result.params
    #     amp    = p['amplitude'].value
    #     sl     = p['sigma_left'].value
    #     sr     = p['sigma_right'].value
    #     center = p['center'].value

    #     failed = (
    #         not result.success
    #         or amp < amplitude_thresh           # too weak
    #         or sl >= sigma_max * 0.95           # sigma hit bound → degenerate
    #         or sr >= sigma_max * 0.95
    #         or not (lags[0] < center < lags[-1])# center outside data range
    #         or result.redchi > 1.0              # poor fit quality
    #     )

    #     if failed:
    #         return None, False

    #     # Store row_range so caller can rescale amplitude if needed
    #     result.row_range = row_range
    #     result.baseline  = baseline
    #     return result, True


    # def extract_single_ridge_lmfit(mi_map, scales, lags,
    #                                 smooth_sigma_scale=2.0, smooth_sigma_lag=2.0,
    #                                 center_init=0.0, center_range=150.0,
    #                                 sigma_init=30.0, sigma_max=400.0,
    #                                 amplitude_thresh=0.05,
    #                                 fit_window=None):
    #     """
    #     fit_window : (lag_min, lag_max) or None.
    #                  Restrict fitting to a lag sub-window around the expected ridge.
    #                  Dramatically helps when large-scale rows are flat outside the peak.
    #     """
    #     sm = smooth_mi_map(mi_map,
    #                        sigma_scale=smooth_sigma_scale,
    #                        sigma_lag=smooth_sigma_lag)

    #     # Optionally restrict to a lag window
    #     if fit_window is not None:
    #         lo, hi = fit_window
    #         mask = (lags >= lo) & (lags <= hi)
    #         lags_fit = lags[mask]
    #         sm_fit   = sm[:, mask]
    #     else:
    #         lags_fit = lags
    #         sm_fit   = sm

    #     n_scales    = len(scales)
    #     centers     = np.full(n_scales, np.nan)
    #     sigma_left  = np.full(n_scales, np.nan)
    #     sigma_right = np.full(n_scales, np.nan)
    #     amplitudes  = np.full(n_scales, np.nan)
    #     success     = np.zeros(n_scales, dtype=bool)

    #     prev_params  = None
    #     fail_streak  = 0
    #     MAX_FAILS    = 5   # reset warm-start after this many consecutive failures

    #     # coarse → fine
    #     for i in range(n_scales - 1, -1, -1):
    #         row = sm_fit[i]

    #         result, ok = fit_single_ridge(
    #             row, lags_fit,
    #             prev_params=prev_params,
    #             center_init=center_init,
    #             center_range=center_range,
    #             sigma_init=sigma_init,
    #             sigma_max=sigma_max,
    #             amplitude_thresh=amplitude_thresh,
    #         )

    #         if ok:
    #             p = result.params
    #             centers[i]     = p['center'].value
    #             sigma_left[i]  = p['sigma_left'].value
    #             sigma_right[i] = p['sigma_right'].value
    #             amplitudes[i]  = p['amplitude'].value * result.row_range
    #             success[i]     = True
    #             prev_params    = result.params
    #             fail_streak    = 0
    #         else:
    #             fail_streak += 1
    #             if fail_streak >= MAX_FAILS:
    #                 prev_params = None   # full reset — don't keep dragging a bad estimate
    #                 fail_streak = 0

    #     return centers, sigma_left, sigma_right, amplitudes, success

    # def plot_fit_at_scale(ax, sm, lags, scales, centers, sigma_left, sigma_right, success,
    #                       target_scale=100.0):
    #     """
    #     On ax: scatter of raw smoothed row + fitted split-Gaussian overlay at the
    #     scale closest to target_scale.
    #     """
    #     # find closest scale index
    #     i = np.argmin(np.abs(scales - target_scale))
    #     actual_scale = scales[i]
    #     row = sm[i]

    #     # baseline-subtract same way as during fitting
    #     baseline = np.percentile(row, 10)
    #     row_display = row - baseline

    #     ax.scatter(lags, row_display, s=6, color='steelblue', alpha=0.6, label='data (baseline sub.)')

    #     if success[i]:
    #         # reconstruct fitted curve
    #         fitted = split_gaussian(lags,
    #                                 amplitude=(row_display.max()),  # rescale to data
    #                                 center=centers[i],
    #                                 sigma_left=sigma_left[i],
    #                                 sigma_right=sigma_right[i])
    #         ax.plot(lags, fitted, 'r-', lw=1.8, label='fit')

    #         # mark center and sigmas
    #         ax.axvline(centers[i],                  color='white',  lw=1.0, ls='--')
    #         ax.axvline(centers[i] - sigma_left[i],  color='cyan',   lw=0.8, ls=':')
    #         ax.axvline(centers[i] + sigma_right[i], color='orange', lw=0.8, ls=':')

    #         ax.set_title(f"Fit at s={actual_scale:.0f}  "
    #                      f"μ={centers[i]:.1f}  "
    #                      f"σL={sigma_left[i]:.1f}  σR={sigma_right[i]:.1f}")
    #     else:
    #         ax.set_title(f"Fit at s={actual_scale:.0f} — FAILED")

    #     ax.set_xlabel('Lag τ')
    #     ax.set_ylabel('MI (baseline sub.)')
    #     ax.legend(fontsize=8)
    #     ax.set_facecolor('#1a1a2e')


    # def plot_single_ridge_lmfit(mi_map, scales, lags,
    #                              centers, sigma_left, sigma_right,
    #                              amplitudes, success, title="",
    #                              smooth_sigma_scale=2.0, smooth_sigma_lag=2.0,
    #                              diagnostic_scale=100.0):
    #     sm = smooth_mi_map(mi_map, smooth_sigma_scale, smooth_sigma_lag)

    #     fig, axes = plt.subplots(2, 2, figsize=(14, 10),
    #                              gridspec_kw={'width_ratios': [3, 1],
    #                                           'height_ratios': [2, 1]})

    #     # ── top-left: heatmap ─────────────────────────────────────────────
    #     ax = axes[0, 0]
    #     pcm = ax.pcolormesh(lags, scales, sm, cmap='viridis', shading='auto')
    #     ax.set_yscale('log')
    #     ax.set_xlabel('Lag τ')
    #     ax.set_ylabel('Scale s')
    #     ax.set_title(f"{title} — asymmetric ridge (lmfit)")
    #     plt.colorbar(pcm, ax=ax, label='MI')

    #     s_ok  = scales[success]
    #     c_ok  = centers[success]
    #     sl_ok = sigma_left[success]
    #     sr_ok = sigma_right[success]

    #     ax.plot(c_ok,         s_ok, 'w--',                lw=1.5, label='center')
    #     ax.plot(c_ok - sl_ok, s_ok, color='cyan',  lw=1, ls=':', label='−σ_left')
    #     ax.plot(c_ok + sr_ok, s_ok, color='orange', lw=1, ls=':', label='+σ_right')
    #     if diagnostic_scale is not None:
    #         # mark the diagnostic scale with a horizontal line
    #         ax.axhline(diagnostic_scale, color='red', lw=0.8, ls='--', alpha=0.6)
    #     ax.legend(fontsize=8)

    #     s_fail = scales[~success]
    #     if len(s_fail):
    #         ax.scatter(np.zeros(len(s_fail)), s_fail, c='red', s=4, zorder=5)

    #     # ── top-right: asymmetry ratio ────────────────────────────────────
    #     ax2 = axes[0, 1]
    #     ratio = sr_ok / sl_ok
    #     ax2.plot(ratio, s_ok, 'k-o', ms=2)
    #     ax2.axvline(1.0, color='gray', lw=0.8, ls='--')
    #     ax2.set_xscale('log')
    #     ax2.set_yscale('log')
    #     ax2.set_xlabel('σ_right / σ_left')
    #     ax2.set_ylabel('Scale s')
    #     ax2.set_title('Asymmetry ratio')

    #     # ── bottom-left: single scale fit diagnostic ──────────────────────
    #     if diagnostic_scale is not None:
    #         ax3 = axes[1, 0]
    #         ax3.set_facecolor('#1a1a2e')
    #         plot_fit_at_scale(ax3, sm, lags, scales,
    #                           centers, sigma_left, sigma_right, success,
    #                           target_scale=diagnostic_scale)
    #     else:
    #         axes[1, 0].axis('off')

    #     # ── bottom-right: unused — can show residuals or leave blank ──────
    #     axes[1, 1].axis('off')

    #     plt.suptitle(title, fontsize=12, y=1.01)
    #     plt.tight_layout()
    #     plt.show()


    # # ── entry point ────────────────────────────────────────────────────────────────

    # def extract_single_ridge_and_plot(
    #         mi_map=None, scales=None, lags=None,
    #         title="",
    #         center_init=0.0, center_range=150.0,
    #         sigma_init=30.0, sigma_max=200.0,
    #         amplitude_thresh=0.05,
    #         smooth_sigma_scale=2.0, smooth_sigma_lag=2.0,
    #         diagnostic_scale=100.0,
    #         fit_window=None,      # e.g. (-200, 200) to restrict lag range
    # ):
    #     mi_map = np.array(mi_map)

    #     centers, sl, sr, amp, ok = extract_single_ridge_lmfit(
    #         mi_map, scales, lags,
    #         smooth_sigma_scale=smooth_sigma_scale,
    #         smooth_sigma_lag=smooth_sigma_lag,
    #         center_init=center_init,
    #         center_range=center_range,
    #         sigma_init=sigma_init,
    #         sigma_max=sigma_max,
    #         amplitude_thresh=amplitude_thresh,
    #         fit_window=fit_window,
    #     )

    #     plot_single_ridge_lmfit(
    #         mi_map, scales, lags,
    #         centers, sl, sr, amp, ok,
    #         title=title,
    #         smooth_sigma_scale=smooth_sigma_scale,
    #         smooth_sigma_lag=smooth_sigma_lag,
    #         diagnostic_scale=diagnostic_scale
    #     )

    #     print(f"Successful fits: {ok.sum()} / {len(ok)}")
    #     return centers, sl, sr, amp, ok
    return


@app.cell
def _(np, results_btc, results_btc_2, results_snp, results_snp_2):
    results_snp_merged = np.concatenate([results_snp,results_snp_2])
    results_btc_merged = np.concatenate([results_btc,results_btc_2])
    return results_btc_merged, results_snp_merged


@app.cell
def _(extract_single_ridge_and_plot, np, results_snp_merged, scales):
    snp_fits=[]
    for _i,_ in enumerate(results_snp_merged):
        snp_fits.append(extract_single_ridge_and_plot(
            mi_map          = np.array(results_snp_merged[_i]["S&P500"]["mi_map_normalized"]),
            scales          = scales,
            lags            = np.linspace(-800, 800, 801),
            title           = "S&P500",
            center_init     = 0.0,
            center_range    = 100.0,
            sigma_init      = 20.0,
            sigma_max       = 200.0,       # tighter than before
            amplitude_thresh= 0.05,
            diagnostic_scale=None,
            fit_window      = (-300, 300), # ignore flat tails entirely
            smooth_sigma_lag= 0,smooth_sigma_scale=0
        ))
    return (snp_fits,)


@app.cell
def _(extract_single_ridge_and_plot, np, results_btc_merged, scales):
    btc_fits=[]
    for _i,_ in enumerate(results_btc_merged):
        btc_fits.append(extract_single_ridge_and_plot(
            mi_map          = np.array(results_btc_merged[_i]["BTC"]["mi_map_normalized"]),
            scales          = scales,
            lags            = np.linspace(-800, 800, 801),
            title           = "BTC",
            center_init     = 0.0,
            center_range    = 100.0,
            sigma_init      = 20.0,
            sigma_max       = 200.0,       # tighter than before
            amplitude_thresh= 0.05,
            diagnostic_scale=None,
            fit_window      = (-300, 300), # ignore flat tails entirely
            smooth_sigma_lag= 0,smooth_sigma_scale=0
        ))
    return (btc_fits,)


@app.cell
def _(btc_fits, np, plt, scales, snp_fits):
    def plot_average_stats():
        def compute_stats(fits):
            arr = np.array(fits)              # shape: (n_fits, n_metrics, n_scales)
            mean = arr.mean(axis=0)           # shape: (n_metrics, n_scales)
            var  = arr.var(axis=0)            # shape: (n_metrics, n_scales)
            return mean, var

        snp_mean, snp_var = compute_stats(snp_fits)
        btc_mean, btc_var = compute_stats(btc_fits)

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

            # BTC
            ax.plot(scales, btc_mean[i], label="BTC mean", color="orange")
            ax.fill_between(scales,
                            btc_mean[i] - np.sqrt(btc_var[i]),
                            btc_mean[i] + np.sqrt(btc_var[i]),
                            color="orange", alpha=0.2)

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
        return snp_mean,snp_var, btc_mean,btc_var
    snp_mean,snp_var, btc_mean,btc_var=plot_average_stats()
    return


@app.cell
def _(btc_fits, plt, scales, snp_fits):

    components = ["ridge center", "σ_left ", "σ_right", "amplitude"]
    datasets = {
        "S&P": snp_fits,
        "BTC": btc_fits
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


@app.cell
def _(np, plt, results_snp_merged):
    def plot_signal_grid():
        fig, axes = plt.subplots(4, 5, figsize=(12, 10))
        axes = axes.flatten()

        for i in range(19):
            ax = axes[i]
            signal = results_snp_merged[i]["S&P500"]["signal"]
            ax.plot(0.01 * np.exp(signal))
            ax.set_title(f"S&P500 batch {i}")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    plot_signal_grid()
    return


@app.cell
def _(np, plt, results_btc_merged):
    def plot_signal_grid_btc():
        fig, axes = plt.subplots(5,5, figsize=(12, 10))
        axes = axes.flatten()

        for i in range(21):
            ax = axes[i]
            signal = results_btc_merged[i]["BTC"]["signal"]
            ax.plot(np.exp(signal))
            ax.set_title(f"btc batch {i}")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    plot_signal_grid_btc()
    return


@app.cell
def _(np, plt, smooth_mi_map):
    def plot_Mi_map_publication(mi_map, scales, lags,
                                   centers=None, sigma_left=None, sigma_right=None,
                                   success=None, title="", 
                                   smooth_sigma_scale=0, smooth_sigma_lag=0,
                                   save_as=None):
        """
        Publication‑grade single‑panel ridge fit plot.
        Springer‑compatible: serif fonts, clean layout, single‑column width.
        """

        # --- Style ------------------------------------------------------------
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "figure.dpi": 300,
            "savefig.dpi": 600,
        })

        # --- Smooth MI map ----------------------------------------------------
        sm = smooth_mi_map(mi_map, smooth_sigma_scale, smooth_sigma_lag)

        # --- Prepare figure ---------------------------------------------------
        fig, ax = plt.subplots(figsize=(3.27, 2.8))  # single‑column width

        pcm = ax.pcolormesh(lags, scales, sm, cmap="viridis", shading="auto")
        ax.set_yscale("log")
        ax.set_xlabel("Lag $\\tau$")
        ax.set_ylabel("Scale $s$")
        ax.set_title(title)

        if centers is not None and sigma_left is not None and sigma_right is not None and success is not None:
            # --- Ridge overlay ----------------------------------------------------
            s_ok  = scales[success]
            c_ok  = centers[success]
            sl_ok = sigma_left[success]
            sr_ok = sigma_right[success]

            ax.plot(c_ok,         s_ok, "w--", lw=1.2, label="center")
            ax.plot(c_ok - sl_ok, s_ok, color="cyan",   lw=0.9, ls=":", label="$-\\sigma_L$")
            ax.plot(c_ok + sr_ok, s_ok, color="orange", lw=0.9, ls=":", label="$+\\sigma_R$")

        # --- Colorbar ---------------------------------------------------------
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
        cbar.set_label("normalized MI")

        ax.legend(frameon=False)
        fig.tight_layout()

        # --- Save if requested ------------------------------------------------
        if save_as is not None:
            fig.savefig(save_as, bbox_inches="tight")

        plt.show()
    def plot_MI_map_from_results(asset_name,
                                ridge_results,
                                results_merged,
                                scales,
                                batch_index=0):

        if ridge_results is not None:
            centers, sl, sr, amp, ok = ridge_results[batch_index]

        # Extract MI map for the chosen asset and batch
        mi_map = np.array(results_merged[batch_index][asset_name]["mi_map_normalized"])
        lags   = np.linspace(-800, 800, 801)

        # Produce publication‑grade figure
        if ridge_results is not None:
            plot_Mi_map_publication(
                mi_map=mi_map,
                scales=scales,
                lags=lags,
                centers=centers,
                sigma_left=sl,
                sigma_right=sr,
                success=ok,
                title=f"{asset_name} — MI Ridge Fit",
                save_as=f"ridge_fit_{asset_name.lower()}_batch_{batch_index+1}.pdf"
            )
        else:
            plot_Mi_map_publication(
            mi_map=mi_map,
            scales=scales,
            lags=lags,
            title=f"{asset_name} — MI map",
            save_as=f"MI_map_{asset_name.lower()}_batch_{batch_index+1}.pdf"
        )



    return (plot_MI_map_from_results,)


@app.cell
def _(plot_MI_map_from_results, results_btc_merged, scales):
    plot_MI_map_from_results(
        asset_name="BTC",
        ridge_results=None,#btc_fits,
        results_merged=results_btc_merged,
        scales=scales,
        batch_index=18
    )
    return


@app.cell
def _(plot_MI_map_from_results, results_snp_merged, scales):
    plot_MI_map_from_results(
        asset_name="S&P500",
        ridge_results=None,#snp_fits,
        results_merged=results_snp_merged,
        scales=scales,
        batch_index=5
    )
    return


@app.cell
def _(btc_fits, plot_MI_map_from_results, results_btc_merged, scales):
    plot_MI_map_from_results(
        asset_name="BTC",
        ridge_results=btc_fits,
        results_merged=results_btc_merged,
        scales=scales,
        batch_index=18
    )
    return


@app.cell
def _(np, plot_ridge_fit_publication, results_btc, results_snp, scales):
    def make_publication_ridge_plots(results_snp, results_btc, scales):
        lags = np.linspace(-800, 800, 801)

        # S&P500
        snp = results_snp[0]["S&P500"]
        plot_ridge_fit_publication(
            mi_map=snp["mi_map_normalized"],
            scales=scales,
            lags=lags,
            centers=snp["centers"],
            sigma_left=snp["sigma_left"],
            sigma_right=snp["sigma_right"],
            success=snp["success"],
            title="S&P500 — MI Ridge Fit",
            save_as="ridge_fit_sp500.pdf"
        )

        # BTC
        btc = results_btc[0]["BTC"]
        plot_ridge_fit_publication(
            mi_map=btc["mi_map_normalized"],
            scales=scales,
            lags=lags,
            centers=btc["centers"],
            sigma_left=btc["sigma_left"],
            sigma_right=btc["sigma_right"],
            success=btc["success"],
            title="BTC — MI Ridge Fit",
            save_as="ridge_fit_btc.pdf"
        )
    make_publication_ridge_plots(results_snp, results_btc, scales)
    return


@app.cell
def _(np, plt):
    def plot_average_stats_publication(scales, snp_fits, btc_fits, metrics):
        """
        Publication‑grade comparison of mean ± std envelopes
        for ridge descriptors across scales.
        """

        # --- Global style (Springer‑compatible) -------------------------------
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "figure.dpi": 300,
            "savefig.dpi": 600,
        })

        # --- Extract descriptor arrays from new fits format -------------------
        def extract_descriptor_arrays(fits, metrics):
            """
            fits: list of batches
                  each batch is a tuple: (descriptor_dict, ...)
            metrics: list of descriptor names
            returns: dict metric -> 2D array (n_fits × n_scales)
            """
            out = {m: [] for m in metrics}

            for batch in fits:
                desc = batch[0]   # dictionary of fitted curves

                for m in metrics:
                    out[m].append(np.array(desc[m]))

            # convert lists to arrays
            for m in metrics:
                out[m] = np.vstack(out[m])   # shape: (n_fits, n_scales)

            return out

        snp_desc = extract_descriptor_arrays(snp_fits, metrics)
        btc_desc = extract_descriptor_arrays(btc_fits, metrics)

        # --- Compute mean and std --------------------------------------------
        snp_mean = {m: snp_desc[m].mean(axis=0) for m in metrics}
        snp_std  = {m: snp_desc[m].std(axis=0)  for m in metrics}

        btc_mean = {m: btc_desc[m].mean(axis=0) for m in metrics}
        btc_std  = {m: btc_desc[m].std(axis=0)  for m in metrics}

        # --- Figure layout ----------------------------------------------------
        fig, axes = plt.subplots(
            2, 2,
            figsize=(6.73, 5.2),
            constrained_layout=True
        )
        axes = axes.flatten()

        panel_labels = ["(a)", "(b)", "(c)", "(d)"]

        # --- Plot each descriptor --------------------------------------------
        for i, metric in enumerate(metrics):
            ax = axes[i]

            # S&P500
            ax.plot(scales, snp_mean[metric], color="steelblue", label="S&P500")
            ax.fill_between(
                scales,
                snp_mean[metric] - snp_std[metric],
                snp_mean[metric] + snp_std[metric],
                color="steelblue",
                alpha=0.20
            )

            # BTC
            ax.plot(scales, btc_mean[metric], color="darkorange", label="BTC")
            ax.fill_between(
                scales,
                btc_mean[metric] - btc_std[metric],
                btc_mean[metric] + btc_std[metric],
                color="darkorange",
                alpha=0.20
            )

            ax.set_xscale("log")
            ax.set_xlabel("Scale $s$")
            ax.set_title(metric)

            if metric == "amplitude":
                ax.set_ylabel("Amplitude")
            else:
                ax.set_ylabel("Lag $\\tau$")

            ax.grid(alpha=0.3)
            ax.legend(frameon=False)

            # Panel label
            ax.text(
                0.02, 0.95, panel_labels[i],
                transform=ax.transAxes,
                fontsize=9,
                va="top", ha="left"
            )

        filename = "descriptors_BTC_SnP"
        fig.savefig(f"{filename}.pdf", bbox_inches="tight")
        plt.show()

        return snp_mean, snp_std, btc_mean, btc_std


    return (plot_average_stats_publication,)


@app.function
def plot_average_stats_publication_single(scales, snp_fits, btc_fits):
    """
    Publication‑grade single‑panel figures for each descriptor.
    Springer‑compatible: serif fonts, clean layout, single‑column width.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # --- Global Springer‑style settings -----------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "figure.dpi": 300,
        "savefig.dpi": 600,
    })

    # --- Extract descriptor arrays from new fits format -------------------
    def extract_descriptor_arrays(fits, metrics):
        """
        fits: list of batches
              each batch is a tuple: (descriptor_dict, ...)
        metrics: list of descriptor names
        returns: dict metric -> 2D array (n_fits × n_scales)
        """
        out = {m: [] for m in metrics}

        for batch in fits:
            desc = batch[0]   # dictionary of fitted curves

            for m in metrics:
                out[m].append(np.array(desc[m]))

        # convert lists to arrays
        for m in metrics:
            out[m] = np.vstack(out[m])   # shape: (n_fits, n_scales)

        return out

    metrics = ["center", "sigma_left", "sigma_right", "amplitude"]

    snp_desc = extract_descriptor_arrays(snp_fits, metrics)
    btc_desc = extract_descriptor_arrays(btc_fits, metrics)

    # --- Compute mean and std --------------------------------------------
    snp_mean = {m: snp_desc[m].mean(axis=0) for m in metrics}
    snp_std  = {m: snp_desc[m].std(axis=0)  for m in metrics}

    btc_mean = {m: btc_desc[m].mean(axis=0) for m in metrics}
    btc_std  = {m: btc_desc[m].std(axis=0)  for m in metrics}

    # --- Loop over metrics and produce one figure per metric --------------
    for metric in metrics:

        fig, ax = plt.subplots(figsize=(3.27, 2.6))  # 8.3 cm × 6.6 cm (single‑column)

        # S&P500
        ax.plot(scales, snp_mean[metric], color="steelblue", label="S&P500")
        ax.fill_between(
            scales,
            snp_mean[metric] - snp_std[metric],
            snp_mean[metric] + snp_std[metric],
            color="steelblue",
            alpha=0.20
        )

        # BTC
        ax.plot(scales, btc_mean[metric], color="darkorange", label="BTC")
        ax.fill_between(
            scales,
            btc_mean[metric] - btc_std[metric],
            btc_mean[metric] + btc_std[metric],
            color="darkorange",
            alpha=0.20
        )

        ax.set_xscale("log")
        ax.set_xlabel("Scale $s$")
        ax.set_title(metric)

        if metric == "amplitude":
            ax.set_ylabel("Amplitude")
        else:
            ax.set_ylabel("Lag $\\tau$")

        ax.grid(alpha=0.3)
        ax.legend(frameon=False)

        fig.tight_layout()
        filename = metric.replace(" ", "_")
        fig.savefig(f"{filename}.pdf", bbox_inches="tight")
        plt.show()

    return snp_mean, snp_std, btc_mean, btc_std


@app.cell
def _():
    return


@app.cell
def _():
    # import plotly.graph_objects as go

    # # Convert your list of arrays into a 2D matrix
    # Z = np.array(d2M_dlag2)   # shape: (rows, cols)

    # # Create X and Y coordinate grids
    # Y = np.log10(scales)
    # X = lags
    # X, Y = np.meshgrid(X, Y)

    # fig = go.Figure(data=[
    #     go.Surface(z=Z, x=X, y=Y, colorscale='Viridis',surfacecolor=Z)
    # ])

    # fig.update_layout(
    #     title='Surface Plot of d2M_dlag2',
    #     scene=dict(
    #         xaxis_title='Index in each array',
    #         yaxis_title='Array number',
    #         zaxis_title='Value',
    #         # zaxis=dict(range=[Z.min(), 2])
    #     ),
    #     autosize=True,
    #     width=900,
    #     height=700
    # )

    # fig.show()
    return


@app.cell
def _(mo):

    mo.md(r"""
    # generalized fitting
    """)
    return


@app.cell
def _(np, plt, smooth_mi_map):


    from lmfit import Model
    # ───────────────────────────────────────────────────────────────────────
    #  MODEL SPECIFICATION SYSTEM
    # ───────────────────────────────────────────────────────────────────────

    class BaseModelSpec:
        """Abstract interface for pluggable models."""
        name = "base"

        def model_func(self, x, **params):
            raise NotImplementedError

        def make_params(self, model, amplitude_thresh, center_init, center_range,
                        sigma_init, sigma_min, sigma_max):
            raise NotImplementedError

        def reconstruct(self, x, params, row_display):
            """Return fitted curve in original scale."""
            raise NotImplementedError

        def param_names(self):
            """Return list of parameter names to extract."""
            raise NotImplementedError

    class SplitGaussianSpec(BaseModelSpec):
        name = "split_gaussian"

        def model_func(self, x, amplitude, center, sigma_left, sigma_right):
            sigma = np.where(x < center, sigma_left, sigma_right)
            return amplitude * np.exp(-0.5 * ((x - center) / sigma)**2)

        def make_params(self, model, amplitude_thresh, center_init, center_range,
                        sigma_init, sigma_min, sigma_max):
            return model.make_params(
                amplitude=dict(value=1.0, min=amplitude_thresh, max=1.5),
                center=dict(value=center_init,
                            min=center_init - center_range,
                            max=center_init + center_range),
                sigma_left=dict(value=sigma_init, min=sigma_min, max=sigma_max),
                sigma_right=dict(value=sigma_init, min=sigma_min, max=sigma_max),
            )

        def reconstruct(self, x, params, row_display):
            return self.model_func(
                x,
                amplitude=row_display.max(),
                center=params['center'],
                sigma_left=params['sigma_left'],
                sigma_right=params['sigma_right'],
            )

        def param_names(self):
            return ["center", "sigma_left", "sigma_right", "amplitude"]

    class GaussianSpec(BaseModelSpec):
        name = "gaussian"

        def model_func(self, x, amplitude, center, sigma):
            return amplitude * np.exp(-0.5 * ((x - center) / sigma)**2)

        def make_params(self, model, amplitude_thresh, center_init, center_range,
                        sigma_init, sigma_min, sigma_max):
            return model.make_params(
                amplitude=dict(value=1.0, min=amplitude_thresh, max=1.5),
                center=dict(value=center_init,
                            min=center_init - center_range,
                            max=center_init + center_range),
                sigma=dict(value=sigma_init, min=sigma_min, max=sigma_max),
            )

        def reconstruct(self, x, params, row_display):
            return self.model_func(
                x,
                amplitude=row_display.max(),
                center=params['center'],
                sigma=params['sigma'],
            )

        def param_names(self):
            return ["center", "sigma", "amplitude"]


    class TwoSidedExpSpec(BaseModelSpec):
        name = "two_sided_exp(Laplace)"

        def model_func(self, x, amplitude, center, sigma_left, sigma_right):
            b = np.where(x < center, sigma_left, sigma_right)
            return amplitude * np.exp(-np.abs(x - center) / b)

        def make_params(self, model, amplitude_thresh, center_init, center_range,
                        sigma_init, sigma_min, sigma_max):
            return model.make_params(
                amplitude=dict(value=1.0, min=amplitude_thresh, max=1.5),
                center=dict(value=center_init,
                            min=center_init - center_range,
                            max=center_init + center_range),
                sigma_left=dict(value=sigma_init, min=sigma_min, max=sigma_max),
                sigma_right=dict(value=sigma_init, min=sigma_min, max=sigma_max),
            )

        def reconstruct(self, x, params, row_display):
            return self.model_func(
                x,
                amplitude=row_display.max(),
                center=params['center'],
                sigma_left=params['sigma_left'],
                sigma_right=params['sigma_right'],
            )

        def param_names(self):
            return ["center", "sigma_left", "sigma_right", "amplitude"]

    def fit_single_ridge(row, lags, model_spec,
                         prev_params=None,
                         center_init=0.0, center_range=150.0,
                         sigma_init=30.0, sigma_min=2.0, sigma_max=400.0,
                         amplitude_thresh=0.05):
        model = Model(model_spec.model_func)

        baseline = np.percentile(row, 10)
        row_centered = row - baseline
        row_range = row_centered.max()

        if row_range < amplitude_thresh:
            return None, False

        row_norm = row_centered / row_range

        if prev_params is not None:
            params = prev_params.copy()
            params['center'].set(
                min=params['center'].value - center_range,
                max=params['center'].value + center_range,
            )
            params['amplitude'].set(value=1.0, min=amplitude_thresh, max=1.5)
        else:
            params = model_spec.make_params(
                model,
                amplitude_thresh,
                center_init,
                center_range,
                sigma_init,
                sigma_min,
                sigma_max,
            )

        try:
            result = model.fit(row_norm, params, x=lags)
        except Exception:
            return None, False

        p = result.params
        amp = p['amplitude'].value
        center = p['center'].value

        failed = (
            not result.success
            or amp < amplitude_thresh
            or not (lags[0] < center < lags[-1])
            or result.redchi > 1.0
        )

        if failed:
            return None, False

        result.row_range = row_range
        result.baseline = baseline
        return result, True

    def extract_single_ridge_lmfit(mi_map, scales, lags, model_spec,
                                   smooth_sigma_scale=2.0, smooth_sigma_lag=2.0,
                                   center_init=0.0, center_range=150.0,
                                   sigma_init=30.0, sigma_max=400.0,
                                   amplitude_thresh=0.05,
                                   fit_window=None):
        sm = smooth_mi_map(mi_map,
                           sigma_scale=smooth_sigma_scale,
                           sigma_lag=smooth_sigma_lag)

        if fit_window is not None:
            lo, hi = fit_window
            mask = (lags >= lo) & (lags <= hi)
            lags_fit = lags[mask]
            sm_fit = sm[:, mask]
        else:
            lags_fit = lags
            sm_fit = sm

        param_names = model_spec.param_names()
        param_arrays = {name: np.full(len(scales), np.nan) for name in param_names}
        success = np.zeros(len(scales), dtype=bool)
        #aic,bic
        fit_scores = {"aic":np.full(len(scales), np.nan),"bic":np.full(len(scales), np.nan)}

        prev_params = None
        fail_streak = 0
        MAX_FAILS = 5

        for i in range(len(scales) - 1, -1, -1):
            row = sm_fit[i]

            result, ok = fit_single_ridge(
                row, lags_fit, model_spec,
                prev_params=prev_params,
                center_init=center_init,
                center_range=center_range,
                sigma_init=sigma_init,
                sigma_max=sigma_max,
                amplitude_thresh=amplitude_thresh,
            )

            if ok:
                for name in param_names:
                    param_arrays[name][i] = result.params[name].value
                param_arrays["amplitude"][i] *= result.row_range
                success[i] = True
                fit_scores["aic"][i] = result.aic 
                fit_scores["bic"][i] = result.bic 
                prev_params = result.params
                fail_streak = 0
            else:
                fail_streak += 1
                if fail_streak >= MAX_FAILS:
                    prev_params = None
                    fail_streak = 0

        return param_arrays, success,fit_scores


    def plot_fit_at_scale(ax, sm, lags, scales, param_arrays, success,
                          model_spec, target_scale=100.0):
        i = np.argmin(np.abs(scales - target_scale))
        row = sm[i]

        baseline = np.percentile(row, 10)
        row_display = row - baseline

        ax.scatter(lags, row_display, s=6, color='steelblue', alpha=0.6)

        if success[i]:
            params = {name: param_arrays[name][i] for name in model_spec.param_names()}
            fitted = model_spec.reconstruct(lags, params, row_display)
            ax.plot(lags, fitted, 'r-', lw=1.8)
            ax.axvline(params["center"], color='white', lw=1.0, ls='--')

        ax.set_xlabel("Lag τ")
        ax.set_ylabel("MI (baseline sub.)")

    def plot_single_ridge_lmfit(mi_map, scales, lags,
                                 param_arrays, success,
                                 model_spec,
                                 title="",
                                 smooth_sigma_scale=2.0, smooth_sigma_lag=2.0,
                                 diagnostic_scale=100.0):
        sm = smooth_mi_map(mi_map, smooth_sigma_scale, smooth_sigma_lag)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                                 gridspec_kw={'width_ratios': [3, 1],
                                              'height_ratios': [2, 1]})

        # ── top-left: heatmap + ridge curves ──────────────────────────────
        ax = axes[0, 0]
        pcm = ax.pcolormesh(lags, scales, sm, cmap='viridis', shading='auto')
        ax.set_yscale('log')
        ax.set_xlabel('Lag τ')
        ax.set_ylabel('Scale s')
        ax.set_title(f"{title} — model={model_spec.name}")
        plt.colorbar(pcm, ax=ax, label='MI')

        centers = param_arrays["center"]
        s_ok = scales[success]
        c_ok = centers[success]

        # center curve
        ax.plot(c_ok, s_ok, 'w--', lw=1.5, label='center')

        # ridge edges: depend on model
        if model_spec.name == "split_gaussian" or model_spec.name == "two_sided_exp(Laplace)":
            sl_ok = param_arrays["sigma_left"][success]
            sr_ok = param_arrays["sigma_right"][success]
            ax.plot(c_ok - sl_ok, s_ok, color='cyan', lw=1, ls=':', label='−σ_left')
            ax.plot(c_ok + sr_ok, s_ok, color='orange', lw=1, ls=':', label='+σ_right')
        elif model_spec.name == "gaussian" or model_spec.name == "two_sided_exp(Laplace)":
            sigma_ok = param_arrays["sigma"][success]
            ax.plot(c_ok - sigma_ok, s_ok, color='cyan', lw=1, ls=':', label='−σ')
            ax.plot(c_ok + sigma_ok, s_ok, color='orange', lw=1, ls=':', label='+σ')

        if diagnostic_scale is not None:
            ax.axhline(diagnostic_scale, color='red', lw=0.8, ls='--', alpha=0.6)

        s_fail = scales[~success]
        if len(s_fail):
            ax.scatter(np.zeros(len(s_fail)), s_fail, c='red', s=4, zorder=5)

        ax.legend(fontsize=8)

        # ── top-right: diagnostics ────────────────────────────────────────
        ax2 = axes[0, 1]
        ax2.set_title("Diagnostics")

        if model_spec.name == "split_gaussian" or model_spec.name == "two_sided_exp(Laplace)":
            sl_ok = param_arrays["sigma_left"][success]
            sr_ok = param_arrays["sigma_right"][success]
            ratio = sr_ok / sl_ok
            ax2.plot(ratio, s_ok, 'k-o', ms=2)
            ax2.axvline(1.0, color='gray', lw=0.8, ls='--')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.set_xlabel('σ_right / σ_left')
            ax2.set_ylabel('Scale s')
        else:
            ax2.text(0.1, 0.5, "No asymmetry diagnostics\nfor this model",
                     transform=ax2.transAxes)

        # ── bottom-left: single-scale fit diagnostic ──────────────────────
        if diagnostic_scale is not None:
            ax3 = axes[1, 0]
            ax3.set_facecolor('#1a1a2e')
            plot_fit_at_scale(ax3, sm, lags, scales,
                              param_arrays, success,
                              model_spec,
                              target_scale=diagnostic_scale)
        else:
            axes[1, 0].axis('off')

        # ── bottom-right: unused ──────────────────────────────────────────
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()




    def extract_single_ridge_and_plot(
            mi_map=None, scales=None, lags=None,
            title="",
            center_init=0.0, center_range=150.0,
            sigma_init=30.0, sigma_max=200.0,
            amplitude_thresh=0.05,
            smooth_sigma_scale=2.0, smooth_sigma_lag=2.0,
            diagnostic_scale=100.0,
            fit_window=None,
            model_spec=SplitGaussianSpec(),   # NEW DEFAULT
    ):
        mi_map = np.array(mi_map)

        param_arrays, ok, fit_scores = extract_single_ridge_lmfit(
            mi_map, scales, lags, model_spec,
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
            param_arrays, ok,
            model_spec=model_spec,
            title=title,
            smooth_sigma_scale=smooth_sigma_scale,
            smooth_sigma_lag=smooth_sigma_lag,
            diagnostic_scale=diagnostic_scale
        )

        print(f"Successful fits: {ok.sum()} / {len(ok)}")
        print(f"AICs: {fit_scores["aic"]}")
        print(f"BICs: {fit_scores["bic"]}")

        return param_arrays, ok, fit_scores

    return (
        GaussianSpec,
        SplitGaussianSpec,
        TwoSidedExpSpec,
        extract_single_ridge_and_plot,
    )


@app.cell
def _(
    SplitGaussianSpec,
    extract_single_ridge_and_plot,
    np,
    results_snp_merged,
    scales,
):
    snp_fits_generalized=[]
    for _i,_ in enumerate(results_snp_merged):
        snp_fits_generalized.append(extract_single_ridge_and_plot(
            mi_map          = np.array(results_snp_merged[_i]["S&P500"]["mi_map_normalized"]),
            scales          = scales,
            lags            = np.linspace(-800, 800, 801),
            title           = "S&P500",
            center_init     = 0.0,
            center_range    = 100.0,
            sigma_init      = 20.0,
            sigma_max       = 200.0,       # tighter than before
            amplitude_thresh= 0.05,
            diagnostic_scale=None,
            fit_window      = (-300, 300), # ignore flat tails entirely
            smooth_sigma_lag= 0,smooth_sigma_scale=0,
            model_spec=SplitGaussianSpec(),   # NEW DEFAULT
        ))
    return (snp_fits_generalized,)


@app.cell
def _(
    GaussianSpec,
    extract_single_ridge_and_plot,
    np,
    results_snp_merged,
    scales,
):
    snp_fits_generalized_gauss=[]
    for _i,_ in enumerate(results_snp_merged):
        snp_fits_generalized_gauss.append(extract_single_ridge_and_plot(
            mi_map          = np.array(results_snp_merged[_i]["S&P500"]["mi_map_normalized"]),
            scales          = scales,
            lags            = np.linspace(-800, 800, 801),
            title           = "S&P500",
            center_init     = 0.0,
            center_range    = 100.0,
            sigma_init      = 20.0,
            sigma_max       = 200.0,       # tighter than before
            amplitude_thresh= 0.05,
            diagnostic_scale=None,
            fit_window      = (-300, 300), # ignore flat tails entirely
            smooth_sigma_lag= 0,smooth_sigma_scale=0,
            model_spec=GaussianSpec(),   # NEW DEFAULT
        ))
    return (snp_fits_generalized_gauss,)


@app.cell
def _(
    TwoSidedExpSpec,
    extract_single_ridge_and_plot,
    np,
    results_snp_merged,
    scales,
):
    snp_fits_exp_generalized=[]
    for _i,_ in enumerate(results_snp_merged):
        snp_fits_exp_generalized.append(extract_single_ridge_and_plot(
            mi_map          = np.array(results_snp_merged[_i]["S&P500"]["mi_map_normalized"]),
            scales          = scales,
            lags            = np.linspace(-800, 800, 801),
            title           = "S&P500",
            center_init     = 0.0,
            center_range    = 100.0,
            sigma_init      = 20.0,
            sigma_max       = 200.0,       # tighter than before
            amplitude_thresh= 0.05,
            diagnostic_scale=None,
            fit_window      = (-300, 300), # ignore flat tails entirely
            smooth_sigma_lag= 0,smooth_sigma_scale=0,
            model_spec=TwoSidedExpSpec(),   # NEW DEFAULT
        ))
    return (snp_fits_exp_generalized,)


@app.cell
def _(
    SplitGaussianSpec,
    extract_single_ridge_and_plot,
    np,
    results_btc_merged,
    scales,
):
    btc_fits_generalized=[]
    for _i,_ in enumerate(results_btc_merged):
        btc_fits_generalized.append(extract_single_ridge_and_plot(
            mi_map          = np.array(results_btc_merged[_i]["BTC"]["mi_map_normalized"]),
            scales          = scales,
            lags            = np.linspace(-800, 800, 801),
            title           = "BTC",
            center_init     = 0.0,
            center_range    = 100.0,
            sigma_init      = 20.0,
            sigma_max       = 200.0,       # tighter than before
            amplitude_thresh= 0.05,
            diagnostic_scale=None,
            fit_window      = (-300, 300), # ignore flat tails entirely
            smooth_sigma_lag= 0,smooth_sigma_scale=0,
            model_spec=SplitGaussianSpec(),   # NEW DEFAULT
        ))
    return (btc_fits_generalized,)


@app.cell
def _(
    GaussianSpec,
    extract_single_ridge_and_plot,
    np,
    results_btc_merged,
    scales,
):
    btc_fits_generalized_gauss=[]
    for _i,_ in enumerate(results_btc_merged):
        btc_fits_generalized_gauss.append(extract_single_ridge_and_plot(
            mi_map          = np.array(results_btc_merged[_i]["BTC"]["mi_map_normalized"]),
            scales          = scales,
            lags            = np.linspace(-800, 800, 801),
            title           = "BTC",
            center_init     = 0.0,
            center_range    = 100.0,
            sigma_init      = 20.0,
            sigma_max       = 200.0,       # tighter than before
            amplitude_thresh= 0.05,
            diagnostic_scale=None,
            fit_window      = (-300, 300), # ignore flat tails entirely
            smooth_sigma_lag= 0,smooth_sigma_scale=0,
            model_spec=GaussianSpec(),   # NEW DEFAULT
        ))

    return (btc_fits_generalized_gauss,)


@app.cell
def _(
    TwoSidedExpSpec,
    extract_single_ridge_and_plot,
    np,
    results_btc_merged,
    scales,
):
    btc_fits_exp_generalized=[]
    for _i,_ in enumerate(results_btc_merged):
        btc_fits_exp_generalized.append(extract_single_ridge_and_plot(
            mi_map          = np.array(results_btc_merged[_i]["BTC"]["mi_map_normalized"]),
            scales          = scales,
            lags            = np.linspace(-800, 800, 801),
            title           = "BTC",
            center_init     = 0.0,
            center_range    = 100.0,
            sigma_init      = 20.0,
            sigma_max       = 200.0,       # tighter than before
            amplitude_thresh= 0.05,
            diagnostic_scale=None,
            fit_window      = (-300, 300), # ignore flat tails entirely
            smooth_sigma_lag= 0,smooth_sigma_scale=0,
            model_spec=TwoSidedExpSpec(),   # NEW DEFAULT
        ))

    return (btc_fits_exp_generalized,)


@app.cell
def _(SplitGaussianSpec):
    SplitGaussianSpec().param_names()
    return


@app.cell
def _(
    SplitGaussianSpec,
    btc_fits_generalized,
    plot_average_stats_publication,
    scales,
    snp_fits_generalized,
):
    plot_average_stats_publication(scales, snp_fits_generalized, btc_fits_generalized,metrics=SplitGaussianSpec().param_names())

    return


@app.cell
def _(
    SplitGaussianSpec,
    btc_fits_exp_generalized,
    plot_average_stats_publication,
    scales,
    snp_fits_exp_generalized,
):
    plot_average_stats_publication(scales, snp_fits_exp_generalized, btc_fits_exp_generalized,metrics=SplitGaussianSpec().param_names())
    return


@app.cell
def _():
    # np.save("btc_fits_generalized_score.npy",np.array(btc_fits_generalized, dtype=object))
    # np.save("snp_fits_generalized_score.npy",np.array(snp_fits_generalized, dtype=object))
    # np.save("snp_fits_generalized_score_gauss.npy",np.array(snp_fits_generalized_gauss, dtype=object))
    # np.save("snp_fits_generalized_score_exp.npy",np.array(snp_fits_exp_generalized, dtype=object))
    # np.save("btc_fits_generalized_score_exp.npy",np.array(btc_fits_exp_generalized, dtype=object))
    return


@app.cell
def _():
    # btc_fits_generalized=np.load("btc_fits_generalized_score.npy",allow_pickle=True)
    # btc_fits_generalized_gauss=np.load("btc_fits_generalized_score_gauss.npy",allow_pickle=True)
    # btc_fits_exp_generalized=np.load("btc_fits_generalized_score_exp.npy",allow_pickle=True)

    # snp_fits_generalized=np.load("snp_fits_generalized_score.npy",allow_pickle=True)
    # snp_fits_generalized_gauss=np.load("snp_fits_generalized_score_gauss.npy",allow_pickle=True)
    # snp_fits_exp_generalized=np.load("snp_fits_generalized_score_exp.npy",allow_pickle=True)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # AIC/BIC model validation
    we're fitting a model f to signal y with error ϵ

    $y_i=f(x_i)+ϵ_i$

    log likelihood for signal with gaussian noise model can be defined as

    $ln⁡L=−2*n* ln⁡(2πs^2)−∑_{i}^{n}(y_i−f(x_i))^2/(2s^2)$

    where s^2=var

    and the aic bic takes form

    $AIC = 2*k - 2*logL$

    $BIC = ln(n)*k - 2*logL$
    """)
    return


@app.cell
def _(np):


    def compute_logLikelihood(x,y,model,params):
        residuals = y - model(x=x,**params)
        var = np.var(residuals,ddof=1)
        n=len(y)
        return -0.5*n*np.log(2*np.pi*var) - np.sum(residuals**2)/(2*var)

    def AIC(logl,k):
        return 2*k - 2*logl

    def BIC(logl,k,n):
        return np.log(n)*k - 2*logl


    return AIC, BIC, compute_logLikelihood


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this section there is manual implementation of aic/bic calculation, which was finally not used for comparison of model, because lmfit provide those kind of scores for its fits
    """)
    return


@app.cell
def _(AIC, BIC, SplitGaussianSpec, compute_logLikelihood, np, plt):
    def compute_aic_bic(fits, result, asset_type="S&P500",model_spec=SplitGaussianSpec()):
        fits_with_score = []

        model = model_spec.model_func
        metrics = model_spec.param_names()

        for _batch, batch in enumerate(fits):

            # ───────────────────────────────────────────────
            # Detect actual structure
            # batch = (param_dict, success_array)
            # ───────────────────────────────────────────────
            if (isinstance(batch, tuple) or isinstance(batch,type(np.array([])))) and len(batch) == 3:
                param_arrays = batch[0]
                success = batch[1]
            elif isinstance(batch, dict):
                # already new format with success inside
                param_arrays = batch
                success = batch["success"]
            else:
                raise ValueError("Unrecognized fits format")

            logls = []
            aics = []
            bics = []

            n_scales = len(param_arrays["center"])
            for _i in range(n_scales):
                if not success[_i]:
                    logls.append(np.nan)
                    aics.append(np.nan)
                    bics.append(np.nan)
                    continue

                # extract parameters

                _params = {metric: param_arrays[metric][_i] for metric in metrics}

                # x and y
                _x = np.arange(-400, 401, 1)
                _y = result[_batch][asset_type]["mi_map_normalized"][_i]

                # baseline subtraction (same as fitter)
                baseline = np.percentile(_y, 10)
                y_centered = _y - baseline


                # model evaluation
                _fit = model(_x, **_params)

                # likelihood
                _logl = compute_logLikelihood(_x, y_centered, model, _params)

                # AIC / BIC
                k = len(_params)
                _aic = AIC(_logl, k)
                _bic = BIC(_logl, k, len(_x))

                if(_batch%5==0):
                    plt.plot(_x,y_centered)
                    plt.plot(_x,_fit)
                    plt.title(f"batch {_batch} AIC:{batch[2]["aic"][_i]} BIC:{batch[2]["bic"][_i]}")
                    plt.show()

                logls.append(_logl)
                aics.append(_aic)
                bics.append(_bic)

            # build unified output structure
            batch_with_scores = {
                **param_arrays,
                "success": success,
                "logl": np.array(logls),
                "aic": np.array(aics),
                "bic": np.array(bics)
            }

            fits_with_score.append(batch_with_scores)

        return fits_with_score


    def compute_average_scores(fits_with_score):
        aics = []
        bics = []

        for batch in fits_with_score:
            aics.extend(batch[2]["aic"])
            bics.extend(batch[2]["bic"])

        aics_nnan = np.nan_to_num(aics)
        bics_nnan = np.nan_to_num(bics)

        return (
            np.sum(aics_nnan),
            np.average(aics_nnan),
            np.sum(bics_nnan),
            np.average(bics_nnan)
        )


    return compute_aic_bic, compute_average_scores


@app.cell
def _(
    SplitGaussianSpec,
    compute_aic_bic,
    results_snp_merged,
    snp_fits_generalized,
):
    snp_fits_with_score=compute_aic_bic(snp_fits_generalized,results_snp_merged,asset_type="S&P500",model_spec=SplitGaussianSpec())
    return


@app.cell
def _(
    GaussianSpec,
    compute_aic_bic,
    results_snp_merged,
    snp_fits_generalized_gauss,
):
    snp_fits_gauss_with_score=compute_aic_bic(snp_fits_generalized_gauss,results_snp_merged,asset_type="S&P500",model_spec=GaussianSpec())
    return


@app.cell
def _(
    TwoSidedExpSpec,
    compute_aic_bic,
    results_snp_merged,
    snp_fits_exp_generalized,
):
    snp_fits_exp_with_score=compute_aic_bic(snp_fits_exp_generalized,results_snp_merged,asset_type="S&P500",model_spec=TwoSidedExpSpec())
    return


@app.cell
def _(
    SplitGaussianSpec,
    btc_fits_generalized,
    compute_aic_bic,
    results_btc_merged,
):
    btc_fits_with_score=compute_aic_bic(btc_fits_generalized,results_btc_merged,asset_type="BTC",model_spec=SplitGaussianSpec())
    return


@app.cell
def _(
    GaussianSpec,
    btc_fits_generalized_gauss,
    compute_aic_bic,
    results_btc_merged,
):
    btc_fits_gauss_with_score=compute_aic_bic(btc_fits_generalized_gauss,results_btc_merged,asset_type="BTC",model_spec=GaussianSpec())
    return


@app.cell
def _(
    TwoSidedExpSpec,
    btc_fits_exp_generalized,
    compute_aic_bic,
    results_btc_merged,
):
    btc_fits_exp_with_score=compute_aic_bic(btc_fits_exp_generalized,results_btc_merged,asset_type="BTC",model_spec=TwoSidedExpSpec())
    return


@app.cell
def _(
    btc_fits_exp_generalized,
    btc_fits_generalized,
    btc_fits_generalized_gauss,
    compute_average_scores,
    snp_fits_exp_generalized,
    snp_fits_generalized,
    snp_fits_generalized_gauss,
):
    import pandas as pd

    # results from your compute_average_scores(...)
    # each is: (sum_aic, avg_aic, sum_bic, avg_bic)

    sp_asym_sum_aic, sp_asym_avg_aic, sp_asym_sum_bic, sp_asym_avg_bic = compute_average_scores(snp_fits_generalized)
    sp_sym_sum_aic,  sp_sym_avg_aic,  sp_sym_sum_bic,  sp_sym_avg_bic  = compute_average_scores(snp_fits_generalized_gauss)
    sp_exp_sum_aic,  sp_exp_avg_aic,  sp_exp_sum_bic,  sp_exp_avg_bic  = compute_average_scores(snp_fits_exp_generalized)

    btc_asym_sum_aic, btc_asym_avg_aic, btc_asym_sum_bic, btc_asym_avg_bic = compute_average_scores(btc_fits_generalized)
    btc_sym_sum_aic,  btc_sym_avg_aic,  btc_sym_sum_bic,  btc_sym_avg_bic =compute_average_scores(btc_fits_generalized_gauss)
    btc_exp_sum_aic,  btc_exp_avg_aic,  btc_exp_sum_bic,  btc_exp_avg_bic = compute_average_scores(btc_fits_exp_generalized)


    df = pd.DataFrame([
        {
            "Asset": "S&P500",
            "Model": "Asymmetric Gaussian",
            "AIC Sum": sp_asym_sum_aic,
            "AIC Avg": sp_asym_avg_aic,
            "BIC Sum": sp_asym_sum_bic,
            "BIC Avg": sp_asym_avg_bic,
        },
        {
            "Asset": "S&P500",
            "Model": "Gaussian",
            "AIC Sum": sp_sym_sum_aic,
            "AIC Avg": sp_sym_avg_aic,
            "BIC Sum": sp_sym_sum_bic,
            "BIC Avg": sp_sym_avg_bic,
        },
        {
            "Asset": "S&P500",
            "Model": "two_sided_exp(Laplace)",
            "AIC Sum": sp_exp_sum_aic,
            "AIC Avg": sp_exp_avg_aic,
            "BIC Sum": sp_exp_sum_bic,
            "BIC Avg": sp_exp_avg_bic,
        },
        {
            "Asset": "BTC",
            "Model": "Asymmetric Gaussian",
            "AIC Sum": btc_asym_sum_aic,
            "AIC Avg": btc_asym_avg_aic,
            "BIC Sum": btc_asym_sum_bic,
            "BIC Avg": btc_asym_avg_bic,
        },
        {
            "Asset": "BTC",
            "Model": "Gaussian",
            "AIC Sum": btc_sym_sum_aic,
            "AIC Avg": btc_sym_avg_aic,
            "BIC Sum": btc_sym_sum_bic,
            "BIC Avg": btc_sym_avg_bic,
        },
        {
            "Asset": "BTC",
            "Model": "two_sided_exp(Laplace)",
            "AIC Sum": btc_exp_sum_aic,
            "AIC Avg": btc_exp_avg_aic,
            "BIC Sum": btc_exp_sum_bic,
            "BIC Avg": btc_exp_avg_bic,
        }
    ])

    df
    return (pd,)


@app.cell
def _(plt):

    def plot_aic_bic_compare_with_batch_labels(models_dict, score_type="aic",title="symetric/asymetric comparison"):
        """
        models_dict = {
            "S&P asym": fits_with_score,
            "S&P sym": fits_with_score,
            "BTC asym": fits_with_score,
            "BTC sym": fits_with_score,
        }

        Each fits_with_score is a list of batches:
            batch["aic"]  -> array of AIC values per scale
            batch["bic"]  -> array of BIC values per scale
        """

        plt.figure(figsize=(16, 8))


        # Plot each model
        for label, fits in models_dict.items():
            aic_seq = []
            for batch in fits:
                aic_seq.extend(batch[2][score_type])
            plt.plot(aic_seq, label=f"{label} {score_type.upper()}")

        # Build x-axis tick positions for batch boundaries
        tick_positions = []
        tick_labels = []

        offset = 0
        for batch_idx, fits in enumerate(models_dict[list(models_dict.keys())[0]]):
            n_scales = len(fits[2]["aic"])
            # first scale of batch
            tick_positions.append(offset)
            tick_labels.append(f"B{batch_idx}-S0")

            offset += n_scales

        plt.xticks(tick_positions, tick_labels, rotation=45, ha="right")

        plt.xlabel("Batch–Scale index")
        plt.ylabel(f"{score_type.upper()} score")
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


    return (plot_aic_bic_compare_with_batch_labels,)


@app.cell
def _(
    plot_aic_bic_compare_with_batch_labels,
    snp_fits_exp_generalized,
    snp_fits_generalized,
    snp_fits_generalized_gauss,
):
    _snp_models_dict = {
        "S&P asym": snp_fits_generalized,
        "S&P sym": snp_fits_generalized_gauss,
        "S&P exp": snp_fits_exp_generalized
    }
    plot_aic_bic_compare_with_batch_labels(_snp_models_dict)
    return


@app.cell
def _(
    plot_aic_bic_compare_with_batch_labels,
    snp_fits_exp_generalized,
    snp_fits_generalized,
    snp_fits_generalized_gauss,
):
    _snp_models_dict = {
        "S&P asym": snp_fits_generalized,
        "S&P sym": snp_fits_generalized_gauss,
        "S&P exp": snp_fits_exp_generalized
    }
    plot_aic_bic_compare_with_batch_labels(_snp_models_dict,"bic")
    return


@app.cell
def _(
    btc_fits_exp_generalized,
    btc_fits_generalized,
    btc_fits_generalized_gauss,
    plot_aic_bic_compare_with_batch_labels,
):
    _btc_models_dict = {
        "BTC asym": btc_fits_generalized,
        "BTC sym": btc_fits_generalized_gauss,
        "BTC exp": btc_fits_exp_generalized
    }
    plot_aic_bic_compare_with_batch_labels(_btc_models_dict)
    return


@app.cell
def _(
    btc_fits_exp_generalized,
    btc_fits_generalized,
    btc_fits_generalized_gauss,
    plot_aic_bic_compare_with_batch_labels,
):
    _btc_models_dict = {
        "BTC asym": btc_fits_generalized,
        "BTC sym": btc_fits_generalized_gauss,
        "BTC exp": btc_fits_exp_generalized
    }
    plot_aic_bic_compare_with_batch_labels(_btc_models_dict,"bic")
    return


@app.cell
def _(np):

    def compute_delta_scores(fits_model1, fits_model2):
        """
        Computes:
        1) global ΔAIC = avg(AIC_model2) - avg(AIC_model1)
        2) per-row ΔAIC = AIC_model2[i] - AIC_model1[i]
        3) per-scale ΔAIC aggregated across batches
        Same for BIC.
        """

        aic1_all = []
        aic2_all = []
        bic1_all = []
        bic2_all = []

        # collect all AIC/BIC values across batches/scales
        for batch1, batch2 in zip(fits_model1, fits_model2):
            aic1_all.extend(batch1[2]["aic"])
            aic2_all.extend(batch2[2]["aic"])
            bic1_all.extend(batch1[2]["bic"])
            bic2_all.extend(batch2[2]["bic"])

        aic1_all = np.array(aic1_all)
        aic2_all = np.array(aic2_all)
        bic1_all = np.array(bic1_all)
        bic2_all = np.array(bic2_all)

        # remove NaNs
        mask = ~np.isnan(aic1_all) & ~np.isnan(aic2_all)
        aic1_all = aic1_all[mask]
        aic2_all = aic2_all[mask]

        mask = ~np.isnan(bic1_all) & ~np.isnan(bic2_all)
        bic1_all = bic1_all[mask]
        bic2_all = bic2_all[mask]

        # 1) GLOBAL ΔAIC (difference of averages)
        delta_aic_global = np.mean(aic2_all) - np.mean(aic1_all)
        delta_bic_global = np.mean(bic2_all) - np.mean(bic1_all)

        # 2) PER-ROW ΔAIC
        delta_aic = aic2_all - aic1_all
        delta_bic = bic2_all - bic1_all

        # 3) PER-SCALE ΔAIC aggregated across batches
        # determine max number of scales
        max_scales = max(len(batch[2]["aic"]) for batch in fits_model1)

        delta_per_scale = []

        for scale_idx in range(max_scales):
            scale_aic1 = []
            scale_aic2 = []
            scale_bic1 = []
            scale_bic2 = []

            for batch1, batch2 in zip(fits_model1, fits_model2):
                aic1 = batch1[2]["aic"]
                aic2 = batch2[2]["aic"]
                bic1 = batch1[2]["bic"]
                bic2 = batch2[2]["bic"]

                if scale_idx < len(aic1):
                    if not np.isnan(aic1[scale_idx]) and not np.isnan(aic2[scale_idx]):
                        scale_aic1.append(aic1[scale_idx])
                        scale_aic2.append(aic2[scale_idx])

                    if not np.isnan(bic1[scale_idx]) and not np.isnan(bic2[scale_idx]):
                        scale_bic1.append(bic1[scale_idx])
                        scale_bic2.append(bic2[scale_idx])

            if len(scale_aic1) > 0:
                scale_delta_aic = np.array(scale_aic2) - np.array(scale_aic1)
                scale_delta_bic = np.array(scale_bic2) - np.array(scale_bic1)

                delta_per_scale.append({
                    "scale": scale_idx,
                    "mean_delta_aic": np.mean(scale_delta_aic),
                    "sd_delta_aic": np.std(scale_delta_aic),
                    "median_delta_aic": np.median(scale_delta_aic),
                    "mean_delta_bic": np.mean(scale_delta_bic),
                    "sd_delta_bic": np.std(scale_delta_bic),
                    "median_delta_bic": np.median(scale_delta_bic),
                })

        return {
            # global difference between avg AIC/BIC
            "delta_aic_global": delta_aic_global,
            "delta_bic_global": delta_bic_global,

            # per-row ΔAIC/ΔBIC
            "delta_aic": delta_aic,
            "delta_bic": delta_bic,

            # mean ± SD of per-row ΔAIC/BIC
            "mean_delta_aic": np.mean(delta_aic),
            "sd_delta_aic": np.std(delta_aic),
            "median_delta_aic": np.median(delta_aic),
            # "perc_delta_aic_gt_10": np.median
            "mean_delta_bic": np.mean(delta_bic),
            "sd_delta_bic": np.std(delta_bic),
            "median_delta_bic": np.median(delta_bic),
            # "perc_delta_aic_gt_10":
            # per-scale ΔAIC/BIC aggregated across batches
            "delta_per_scale": delta_per_scale
        }


    return (compute_delta_scores,)


@app.cell
def _(
    btc_fits_exp_generalized,
    btc_fits_generalized,
    btc_fits_generalized_gauss,
    compute_delta_scores,
    np,
    pd,
    snp_fits_exp_generalized,
    snp_fits_generalized,
    snp_fits_generalized_gauss,
):

    sp_gauss_vs_asym = compute_delta_scores(snp_fits_generalized_gauss,snp_fits_generalized)
    sp_asym_vs_exp   = compute_delta_scores(snp_fits_generalized,snp_fits_exp_generalized)
    btc_gauss_vs_asym = compute_delta_scores(btc_fits_generalized_gauss,btc_fits_generalized)
    btc_asym_vs_exp   = compute_delta_scores(btc_fits_generalized,btc_fits_exp_generalized)

    comparison1 = "Gauss vs assymetric Gauss"
    comparison2 = "assymetric Gauss vs assymetric Exp"

    df_delta_aic = pd.DataFrame([
        {
            "Asset": "S&P500",
            "Comparison": comparison1,
            "Mean ΔAIC": sp_gauss_vs_asym["mean_delta_aic"],
            "SD ΔAIC": sp_gauss_vs_asym["sd_delta_aic"],
            "Median ΔAIC": sp_gauss_vs_asym["median_delta_aic"],
            "% abs(ΔAIC) > 10": 100*np.sum(np.abs(sp_gauss_vs_asym["delta_aic"])>10)/len(sp_gauss_vs_asym["delta_aic"]),
            "Mean ΔBIC": sp_gauss_vs_asym["mean_delta_bic"],
            "SD ΔBIC": sp_gauss_vs_asym["sd_delta_bic"],
            "Median ΔBIC": sp_gauss_vs_asym["median_delta_bic"],
        },
        {
            "Asset": "S&P500",
            "Comparison": comparison2,
            "Mean ΔAIC": sp_asym_vs_exp["mean_delta_aic"],
            "SD ΔAIC": sp_asym_vs_exp["sd_delta_aic"],
            "Median ΔAIC": sp_asym_vs_exp["median_delta_aic"],
            "% abs(ΔAIC) > 10": 100*np.sum(np.abs(sp_asym_vs_exp["delta_aic"])>10)/len(sp_asym_vs_exp["delta_aic"]),
            "Mean ΔBIC": sp_asym_vs_exp["mean_delta_bic"],
            "SD ΔBIC": sp_asym_vs_exp["sd_delta_bic"],
            "Median ΔBIC": sp_asym_vs_exp["median_delta_bic"],
        },
        {
            "Asset": "BTC",
            "Comparison": comparison1,
            "Mean ΔAIC": btc_gauss_vs_asym["mean_delta_aic"],
            "SD ΔAIC": btc_gauss_vs_asym["sd_delta_aic"],
            "Median ΔAIC": btc_gauss_vs_asym["median_delta_aic"],
            "% abs(ΔAIC) > 10": 100*np.sum(np.abs(btc_gauss_vs_asym["delta_aic"])>10)/len(btc_gauss_vs_asym["delta_aic"]),
            "Mean ΔBIC": btc_gauss_vs_asym["mean_delta_bic"],
            "SD ΔBIC": btc_gauss_vs_asym["sd_delta_bic"],
            "Median ΔBIC": btc_gauss_vs_asym["median_delta_bic"],
        },
        {
            "Asset": "BTC",
            "Comparison": comparison2,
            "Mean ΔAIC": btc_asym_vs_exp["mean_delta_aic"],
            "SD ΔAIC": btc_asym_vs_exp["sd_delta_aic"],
            "Median ΔAIC": btc_asym_vs_exp["median_delta_aic"],
            "% abs(ΔAIC) > 10": 100*np.sum(np.abs(btc_asym_vs_exp["delta_aic"])>10)/len(btc_asym_vs_exp["delta_aic"]),
            "Mean ΔBIC": btc_asym_vs_exp["mean_delta_bic"],
            "SD ΔBIC": btc_asym_vs_exp["sd_delta_bic"],
            "Median ΔBIC": btc_asym_vs_exp["median_delta_bic"],
        }
    ])

    df_delta_aic
    return (sp_asym_vs_exp,)


@app.cell
def _(np, sp_asym_vs_exp):
    np.abs(sp_asym_vs_exp["delta_aic"])>10
    return


@app.cell
def _():
    return


@app.cell
def _(np, plt):

    def plot_delta_aic_per_scale_boxes(fits_model1, fits_model2, scales, title="ΔAIC per scale (model2 - model1)", meansdon=False,score="aic"):
        """
        For each scale s:
          - collect ΔAIC_s across all batches
          - draw a box at x = s (distribution across batches)
          - overlay mean ± std as errorbars
        """

        # determine max number of scales
        max_scales = max(len(batch[2][score]) for batch in fits_model1)

        # collect per-scale ΔAIC across batches
        delta_per_scale = []
        scale_indices = []

        for scale_idx in range(max_scales):
            deltas = []

            for batch1, batch2 in zip(fits_model1, fits_model2):
                aic1 = batch1[2][score]
                aic2 = batch2[2][score]

                if scale_idx < len(aic1):
                    if not np.isnan(aic1[scale_idx]) and not np.isnan(aic2[scale_idx]):
                        deltas.append(aic2[scale_idx] - aic1[scale_idx])

            if len(deltas) > 0:
                delta_per_scale.append(deltas)
                scale_indices.append(scale_idx)

        # compute mean and std per scale
        means = [np.mean(d) for d in delta_per_scale]
        stds  = [np.std(d) for d in delta_per_scale]

        plt.figure(figsize=(14, 6))

        # box per scale
        plt.boxplot(
            delta_per_scale,
            positions=scale_indices,
            widths=0.6,
            patch_artist=True
        )
        if meansdon:
        # mean ± std per scale
            plt.errorbar(
                scale_indices,
                means,
                yerr=stds,
                fmt='o',
                color='black',
                capsize=5,
                label='Mean ± SD'
            )

        plt.axhline(0, color="black", lw=1, ls="--")
        plt.xlabel("Scale index")
        plt.ylabel(f"Δ{score.upper()} (model2 - model1)")
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()

    return (plot_delta_aic_per_scale_boxes,)


@app.cell
def _(
    btc_fits_exp_generalized,
    btc_fits_generalized,
    btc_fits_generalized_gauss,
    plot_delta_aic_per_scale_boxes,
    scales,
    snp_fits_exp_generalized,
    snp_fits_generalized,
    snp_fits_generalized_gauss,
):
    plot_delta_aic_per_scale_boxes(snp_fits_generalized_gauss,snp_fits_generalized,scales,"ΔAIC per scale (assymetric Gauss - Gauss) S&P500")
    plot_delta_aic_per_scale_boxes(snp_fits_generalized,snp_fits_exp_generalized,scales,"ΔAIC per scale (assymetric Gauss - assymetric Exp) S&P500")

    plot_delta_aic_per_scale_boxes(btc_fits_generalized_gauss,btc_fits_generalized,scales,"ΔAIC per scale (assymetric Gauss - Gauss) BTC")
    plot_delta_aic_per_scale_boxes(btc_fits_generalized,btc_fits_exp_generalized,scales,"ΔAIC per scale (assymetric Gauss - assymetric Exp) BTC")
    return


@app.cell
def _(
    btc_fits_exp_generalized,
    btc_fits_generalized,
    btc_fits_generalized_gauss,
    plot_delta_aic_per_scale_boxes,
    scales,
    snp_fits_exp_generalized,
    snp_fits_generalized,
    snp_fits_generalized_gauss,
):
    plot_delta_aic_per_scale_boxes(snp_fits_generalized_gauss,snp_fits_generalized,scales,"ΔBIC per scale (assymetric Gauss - Gauss) S&P500",score="bic")
    plot_delta_aic_per_scale_boxes(snp_fits_generalized,snp_fits_exp_generalized,scales,"ΔBIC per scale (assymetric Gauss - assymetric Exp) S&P500",score="bic")

    plot_delta_aic_per_scale_boxes(btc_fits_generalized_gauss,btc_fits_generalized,scales,"ΔBIC per scale (assymetric Gauss - Gauss) BTC",score="bic")
    plot_delta_aic_per_scale_boxes(btc_fits_generalized,btc_fits_exp_generalized,scales,"ΔBIC per scale (assymetric Gauss - assymetric Exp) BTC",score="bic")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # descriptor statistical analysis
    """)
    return


@app.cell
def _(btc_fits_generalized, snp_fits_generalized):
    # parameters you want to extract
    parameters = ["center", "sigma_left", "sigma_right"]

    # initialize pool
    sample_pool = {
        "BTC": {param: {} for param in parameters},
        "S&P":  {param: {} for param in parameters}
    }

    # helper to fill pool for one dataset
    def fill_pool(sample_pool, dataset_name, fits_generalized):
        """
        dataset_name: "BTC" or "SP"
        fits_generalized: list of batches
            fits_generalized[batch][param_tuple_id][param_name][scale]
        """

        for batch_id, batch in enumerate(fits_generalized):
            param_tuple = batch[0]   # your parameter dict is at index 0

            for param in parameters:
                values = param_tuple[param]  # array of values per scale

                for scale_idx, value in enumerate(values):
                    if scale_idx not in sample_pool[dataset_name][param]:
                        sample_pool[dataset_name][param][scale_idx] = []

                    sample_pool[dataset_name][param][scale_idx].append(value)


    # fill BTC and S&P pools
    fill_pool(sample_pool, "BTC", btc_fits_generalized)
    fill_pool(sample_pool, "S&P", snp_fits_generalized)

    return fill_pool, parameters, sample_pool


@app.cell
def _():

    # btc_vals = sample_pool["BTC"]["sigma_left"][0]
    # sp_vals  = sample_pool["S&P"]["sigma_left"][0]

    # _stat,_p = mannwhitneyu(btc_vals, sp_vals, alternative='two-sided')
    # _res = permutation_test((btc_vals, sp_vals),
    #                      statistic=lambda x, y: x.mean() - y.mean(),
    #                      permutation_type='independent')

    # print("Scale 0 — Mann–Whitney p:", _p)
    # print("Permutation p-value:",_res.pvalue)

    return


@app.cell
def _(np, pd):
    from scipy.stats import mannwhitneyu
    from scipy.stats import permutation_test

    def compare_all_parameters_all_scales(sample_pool,scales):
        """
        sample_pool["BTC"][param][scale] = list of values
        sample_pool["S&P"][param][scale] = list of values
        """

        rows = []

        for param in sample_pool["BTC"].keys():
            for scale in sample_pool["BTC"][param].keys():

                btc_vals = np.array(sample_pool["BTC"][param][scale])
                sp_vals  = np.array(sample_pool["S&P"][param][scale])

                # skip empty scales
                if len(btc_vals) == 0 or len(sp_vals) == 0:
                    continue
                btc_vals = btc_vals[~np.isnan(btc_vals)]
                sp_vals = sp_vals[~np.isnan(sp_vals)]

                # skip empty scales
                if len(btc_vals) < 5 or len(sp_vals) < 5:
                    print(f"sample size was smaller than 5 {param,scale}")
                    continue
            
                # Mann–Whitney U
                _, p_mw = mannwhitneyu(btc_vals, sp_vals, alternative='two-sided')

                # Permutation test
                perm_res = permutation_test(
                    (btc_vals, sp_vals),
                    statistic=lambda x, y: x.mean() - y.mean(),
                    permutation_type='independent'
                )
                p_perm = perm_res.pvalue

                rows.append({
                    "parameter": param,
                    "scale": scales[scale],
                    "sample sizes(BTCvS&P)":(len(btc_vals),len(sp_vals)),
                    "mann_whitney_p": p_mw,
                    "permutation_p": p_perm,
                    "btc_mean": np.mean(btc_vals),
                    "sp_mean": np.mean(sp_vals),
                    "btc_std": np.std(btc_vals),
                    "sp_std": np.std(sp_vals),
                    "difference_mean": np.mean(btc_vals) - np.mean(sp_vals)
                })

        return pd.DataFrame(rows)


    return (compare_all_parameters_all_scales,)


@app.cell
def _(np):
    np.array([1,2,3,np.nan])[~np.isnan([1,2,3,np.nan])]
    return


@app.cell
def _(compare_all_parameters_all_scales, sample_pool, scales):
    compare_all_parameters_all_scales(sample_pool,scales)



    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ROC/AUC
    """)
    return


@app.cell
def _(np, plt):
    from sklearn.metrics import roc_auc_score

    def compute_auc_per_scale(sample_pool, param):
        """
        Computes AUC for BTC vs S&P500 for a given parameter across all scales.
        sample_pool["BTC"][param][scale] = list of values
        sample_pool["S&P"][param][scale] = list of values
        """

        auc_dict = {}

        for scale in sample_pool["BTC"][param].keys():

            btc_vals = np.array(sample_pool["BTC"][param][scale])
            sp_vals  = np.array(sample_pool["S&P"][param][scale])

            # concatenate
            scores = np.concatenate([btc_vals, sp_vals])
            labels = np.array([1]*len(btc_vals) + [0]*len(sp_vals))

            # remove NaNs
            mask = ~np.isnan(scores)
            scores = scores[mask]
            labels = labels[mask]

            # if only one class remains or no data
            if len(scores) == 0 or len(np.unique(labels)) < 2:
                auc_dict[scale] = np.nan
                continue

            # compute AUC
            auc = roc_auc_score(labels, scores)
            auc_dict[scale] = auc

        return auc_dict


    def plot_auc_curve(auc_dict, title):
        scales = sorted(auc_dict.keys())
        aucs = [auc_dict[s] for s in scales]

        plt.plot(scales, aucs, marker="o")
        plt.axhline(0.5, color="black", ls="--")
        plt.title(title)
        plt.xlabel("Scale")
        plt.ylabel("AUC")
        plt.grid(alpha=0.3)
        plt.show()


    from sklearn.metrics import roc_curve

    def plot_roc_per_scale(sample_pool, param,scale_list=[]):
        """
        Plots ROC curves for BTC vs S&P500 for a given parameter across all scales.
        """

        plt.figure(figsize=(12, 8))

        if not scale_list :
            scale_list=sample_pool["BTC"][param].keys()
        for scale in scale_list:

            btc_vals = np.array(sample_pool["BTC"][param][scale])
            sp_vals  = np.array(sample_pool["S&P"][param][scale])

            scores = np.concatenate([btc_vals, sp_vals])
            labels = np.array([1]*len(btc_vals) + [0]*len(sp_vals))

            # remove NaNs
            mask = ~np.isnan(scores)
            scores = scores[mask]
            labels = labels[mask]

            # skip scales with insufficient data
            if len(scores) == 0 or len(np.unique(labels)) < 2:
                continue

            fpr, tpr, _ = roc_curve(labels, scores)

            plt.plot(fpr, tpr, ls=":",label=f"Scale {scale}")

        plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC=0.5)")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title(f"ROC curves for parameter: {param}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


    return compute_auc_per_scale, plot_auc_curve, plot_roc_per_scale


@app.cell
def _(compute_auc_per_scale, sample_pool):
    auc_sigma_left  = compute_auc_per_scale(sample_pool, "sigma_left")
    auc_sigma_right = compute_auc_per_scale(sample_pool, "sigma_right")
    auc_center      = compute_auc_per_scale(sample_pool, "center")

    return auc_center, auc_sigma_left, auc_sigma_right


@app.cell
def _(auc_center, auc_sigma_left, auc_sigma_right, plot_auc_curve):
    plot_auc_curve(auc_sigma_left,  "AUC of sigma_left vs scale")
    plot_auc_curve(auc_sigma_right, "AUC of sigma_right vs scale")
    plot_auc_curve(auc_center,      "AUC of center vs scale")

    return


@app.cell
def _(plot_roc_per_scale, sample_pool):
    plot_roc_per_scale(sample_pool, "sigma_left",scale_list=[0,1,2,3,4,5,6,7,8,9])
    plot_roc_per_scale(sample_pool, "sigma_right",scale_list=[0,1,2,3,4,5,6,7,8,9])
    plot_roc_per_scale(sample_pool, "center",scale_list=[0,1,2,3,4,5,6,7,8,9])

    return


@app.cell
def _(plot_roc_per_scale, sample_pool):
    plot_roc_per_scale(sample_pool, "sigma_left",scale_list=[10,11,12,13,14,15,16,17,18,19])
    plot_roc_per_scale(sample_pool, "sigma_right",scale_list=[10,11,12,13,14,15,16,17,18,19])
    plot_roc_per_scale(sample_pool, "center",scale_list=[10,11,12,13,14,15,16,17,18,19])

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EXP
    """)
    return


@app.cell
def _(
    btc_fits_exp_generalized,
    compute_auc_per_scale,
    fill_pool,
    parameters,
    snp_fits_exp_generalized,
):
    sample_pool_exp = {
        "BTC": {param: {} for param in parameters},
        "S&P":  {param: {} for param in parameters}
    }

    fill_pool(sample_pool_exp, "BTC", btc_fits_exp_generalized)
    fill_pool(sample_pool_exp, "S&P", snp_fits_exp_generalized)

    auc_sigma_left_exp  = compute_auc_per_scale(sample_pool_exp, "sigma_left")
    auc_sigma_right_exp = compute_auc_per_scale(sample_pool_exp, "sigma_right")
    auc_center_exp      = compute_auc_per_scale(sample_pool_exp, "center")

    return auc_center_exp, auc_sigma_left_exp, auc_sigma_right_exp


@app.cell
def _(auc_center_exp, auc_sigma_left_exp, auc_sigma_right_exp, plot_auc_curve):
    plot_auc_curve(auc_sigma_left_exp,  "AUC of sigma_left vs scale(exp)")
    plot_auc_curve(auc_sigma_right_exp, "AUC of sigma_right vs scale(exp)")
    plot_auc_curve(auc_center_exp,      "AUC of center vs scale(exp)")

    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
