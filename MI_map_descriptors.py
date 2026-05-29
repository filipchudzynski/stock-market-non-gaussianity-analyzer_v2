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
    return (results_snp,)


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
    plt.colorbar(label='MI')
    plt.show()
    return


@app.cell
def _(gaussian_filter):
    def smooth_mi_map(mi_map, sigma_scale=1.0, sigma_lag=2.0, alpha=0.0):
        """
        alpha = regularisation strength (0 = none)
        """
        smoothed = gaussian_filter(mi_map, sigma=[sigma_scale, sigma_lag])

        if alpha > 0:
            # Tikhonov regularisation: (I + αL)^(-1) M
            # Here: simple shrinkage toward smoothness
            smoothed = (1 - alpha) * smoothed + alpha * gaussian_filter(smoothed, sigma=3)

        return smoothed

    return (smooth_mi_map,)


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
    def plot_comparison_with_ridges(input_map, scales, sigmas, lags, smooth_fn):
        """
        input_map : 2D MI map (scales × lags)
        scales    : array of scale values (top→bottom)
        sigmas    : list of (sigma_scale, sigma_lag)
        lags      : array of lag values
        smooth_fn : smoothing function, e.g. smooth_mi_map
        """

        # --- 1. Plot raw input ---
        plt.figure(figsize=(12, 5))
        plt.imshow(
            input_map,
            interpolation=None,
            aspect="auto",
            extent=[lags[0], lags[-1], scales[-1], scales[0]],

        )
        plt.title("S&P500 — Raw MI map")
        plt.xlabel("Time lag")
        plt.ylabel("Scale")
        plt.colorbar(label="MI")
        plt.tight_layout()
        plt.show()

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
        
            plt.legend()
            plt.tight_layout()
            plt.show()


    return (plot_comparison_with_ridges,)


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
    return (input_map,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # why ridge is breaking?
    """)
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


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
