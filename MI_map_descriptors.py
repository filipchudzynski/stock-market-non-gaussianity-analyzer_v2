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
    return results_btc, results_snp


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


    return plot_comparison_with_ridges, plot_raw_mi_map


@app.cell
def _():
    return


@app.cell
def _(np, plot_comparison_with_ridges, results_snp, smooth_mi_map):
    input_map = results_snp[0]["S&P500"]["mi_map"]
    scales = results_snp[0]["S&P500"]["scales"]
    lags = np.linspace(-800, 800, 801)  # or your actual lag array

    sigmas = [(0,0),(0.8, 1.5), (1.0, 2.0), (1.2, 2.5)]

    plot_comparison_with_ridges(
        input_map=input_map,
        scales=scales,
        sigmas=sigmas,
        lags=lags,
        smooth_fn=smooth_mi_map
    )
    return input_map, lags, scales


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


@app.cell
def _(np, plt, smooth_mi_map):
    from lmfit import Model
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
def _(extract_single_ridge_and_plot, np, results_snp, scales):
    snp_fits=[]
    for _i in range(9):
        snp_fits.append(extract_single_ridge_and_plot(
            mi_map          = np.array(results_snp[_i]["S&P500"]["mi_map_normalized"]),
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
def _(extract_single_ridge_and_plot, np, results_btc, scales):
    btc_fits=[]
    for _i in range(10):
        btc_fits.append(extract_single_ridge_and_plot(
            mi_map          = np.array(results_btc[_i]["BTC"]["mi_map_normalized"]),
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
def _(np, plt, results_snp):
    def plot_signal_grid():
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.flatten()
    
        for i in range(9):
            ax = axes[i]
            signal = results_snp[i]["S&P500"]["signal"]
            ax.plot(0.01 * np.exp(signal))
            ax.set_title(f"S&P500 batch {i}")
            ax.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    plot_signal_grid()
    return


@app.cell
def _(np, plt, results_btc):
    def plot_signal_grid_btc():
        fig, axes = plt.subplots(4, 3, figsize=(12, 10))
        axes = axes.flatten()
    
        for i in range(10):
            ax = axes[i]
            signal = results_btc[i]["BTC"]["signal"]
            ax.plot(np.exp(signal))
            ax.set_title(f"btc batch {i}")
            ax.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    plot_signal_grid_btc()
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
def _(scales):
    scales[15]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
