import numpy as np
from scipy.stats import kurtosis
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.feature_selection import mutual_info_regression

from detrending import moving_average
from increments import increments
from intermittency import lambda2
from intermittency_epjst_extension3 import lambda2_lognormal, mutual_information_knn
from mi import mutual_information

# -------------------------
# Config
# -------------------------
SCALES = [32, 64, 128, 256, 512]
DETREND_FACTOR = 0.1
BOOTSTRAP_SAMPLES = 400
CI_LEVEL = 0.95
N = 5000

from config import MIN_SAMPLES

# -------------------------
# Bootstrap λ₂(s)
# -------------------------
def bootstrap_lambda2(inc, estimator=lambda2.estimate_lambda2, B=BOOTSTRAP_SAMPLES, ci=CI_LEVEL):
    """Bootstrap an intermittency estimator on increments.

    Parameters:
        inc: array-like increments
        estimator: callable that accepts an increments array and returns a scalar λ₂ estimate
    """
    inc = inc[~np.isnan(inc)]
    n = len(inc)
    if n < MIN_SAMPLES:
        return np.nan, np.nan, np.nan

    stats = []
    for _ in range(B):
        sample = np.random.choice(inc, size=n, replace=True)
        stats.append(estimator(sample))

    stats = np.array(stats)
    lower = np.percentile(stats, (1-ci)*50)
    upper = np.percentile(stats, 100 - (1-ci)*50)
    return np.mean(stats), lower, upper


# -------------------------
# Bootstrap MI (KSG via mutual_info_regression)
# -------------------------
def bootstrap_mi(x, y, B=BOOTSTRAP_SAMPLES, ci=CI_LEVEL):
    x = x.reshape(-1, 1)
    n = len(x)
    stats = []

    for _ in range(B):
        idx = np.random.choice(n, size=n, replace=True)
        xb = x[idx]
        yb = y[idx]
        mi = mutual_info_regression(xb, yb)
        stats.append(mi[0])

    stats = np.array(stats)
    lower = np.percentile(stats, (1-ci)*50)
    upper = np.percentile(stats, 100 - (1-ci)*50)
    return np.mean(stats), lower, upper


# -------------------------
# Detrending helper
# -------------------------
def detrend_series(data, detrender, window):
    trend = np.array([detrender.detrend_point(data, i, window) for i in range(len(data))])
    return data - trend, trend


# -------------------------
# Calculate kurtosis variance
# -------------------------

def kurtosis_error_band(list_of_samples):
    sigma = [np.sqrt(24/len(sample)) for sample in list_of_samples]
    lower = [-2*s for s in sigma]
    upper = [ 2*s for s in sigma]
    return lower, upper


# -------------------------
# Plotting helpers (refactor: keep logic identical)
# -------------------------
def plot_raw_noise(data, title_prefix):
    fig0 = px.line(y=data, title=f"{title_prefix}: Generated Noise")
    fig0.show()


def plot_per_scale(scales, increments_dict, detrended_dict, detrended_increments_dict,
                   detrend_windows,
                   lambda2_raw_std, lambda2_detr_std,
                   lambda2_raw_std_ci, lambda2_detr_std_ci,
                   bin_factor, global_hist_ymax, title_prefix):
    for s in scales:

        print(f"λ₂ raw_std={lambda2_raw_std[s]:.3f}"
              f"[{lambda2_raw_std_ci[s][0]:.3f}, {lambda2_raw_std_ci[s][1]:.3f}]\n"
              f"λ₂ detr_std={lambda2_detr_std[s]:.3f}"
              f"[{lambda2_detr_std_ci[s][0]:.3f}, {lambda2_detr_std_ci[s][1]:.3f}]\n")

        subplot_title = (
            f"s={s} — Distributions<br>"
        )

        fig = make_subplots(
            rows=1,
            cols=4,
            subplot_titles=[
                f"s={s} — Increments",
                f"s={s} — Detrended (window={detrend_windows[s]})",
                f"s={s} — Detrended increments",
                subplot_title
            ]
        )

        # 1. Increments
        fig.add_trace(
            go.Scatter(y=increments_dict[s], mode="lines",
                       name=f"Increments (s={s})"),
            row=1, col=1
        )

        # 2. Detrended signal
        fig.add_trace(
            go.Scatter(y=detrended_dict[s], mode="lines",
                       name=f"Detrended"),
            row=1, col=2
        )

        # 3. Detrended increments
        fig.add_trace(
            go.Scatter(y=detrended_increments_dict[s], mode="lines",
                       name=f"Detrended increments"),
            row=1, col=3
        )

        # 4. Standardized histograms with adaptive binning
        raw_inc = increments_dict[s]
        detr_inc = detrended_increments_dict[s]

        raw_std = (raw_inc - np.mean(raw_inc)) / np.std(raw_inc)
        detr_std = (detr_inc - np.mean(detr_inc)) / np.std(detr_inc)

        N = len(raw_std)
        bins = int(bin_factor * np.sqrt(N))

        counts_raw, edges = np.histogram(raw_std, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        counts_detr, _ = np.histogram(detr_std, bins=edges, density=True)

        fig.add_trace(
            go.Scatter(
                x=centers,
                y=counts_raw,
                mode="markers",
                marker=dict(size=6, color="blue"),
                name="Raw increments (std)"
            ),
            row=1, col=4
        )

        fig.add_trace(
            go.Scatter(
                x=centers,
                y=counts_detr,
                mode="markers",
                marker=dict(size=6, color="red"),
                name="Detrended increments (std)"
            ),
            row=1, col=4
        )

        fig.update_yaxes(range=[0, global_hist_ymax], row=1, col=4)

        fig.update_layout(
            height=350,
            width=2200,
            showlegend=True,
            title_text=f"{title_prefix} — Scale s={s}"
        )
        fig.show()


def plot_lambda2_with_ci(scales,
                         lambda2_raw_std, lambda2_detr_std,
                         lambda2_raw_std_ci, lambda2_detr_std_ci,
                         increments_dict,estimator_confidence_interval, title_prefix):
    fig3 = go.Figure()

    # standardized raw
    fig3.add_trace(go.Scatter(
        x=scales,
        y=[lambda2_raw_std[s] for s in scales],
        mode="lines+markers",
        name="raw increments (std)",
        line=dict(color="green")
    ))
    fig3.add_trace(go.Scatter(
        x=scales + scales[::-1],
        y=[lambda2_raw_std_ci[s][0] for s in scales] +
          [lambda2_raw_std_ci[s][1] for s in scales[::-1]],
        fill="toself",
        fillcolor="rgba(0,255,0,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="raw increments (std) CI"
    ))

    # standardized detrended
    fig3.add_trace(go.Scatter(
        x=scales,
        y=[lambda2_detr_std[s] for s in scales],
        mode="lines+markers",
        name="detrended increments (std)",
        line=dict(color="purple")
    ))
    fig3.add_trace(go.Scatter(
        x=scales + scales[::-1],
        y=[lambda2_detr_std_ci[s][0] for s in scales] +
          [lambda2_detr_std_ci[s][1] for s in scales[::-1]],
        fill="toself",
        fillcolor="rgba(128,0,128,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="detrended increments (std) CI"
    ))
    
    if estimator_confidence_interval:
      est_err_lower,est_err_upper = estimator_confidence_interval(increments_dict.values())

      fig3.add_trace(go.Scatter(
          x=scales,
          y=est_err_lower,
          mode="lines+markers",
          name="est ci -2σ [σ=sqrt(24/Ns)]",
          line=dict(color="black",dash="dash")
      ))

      fig3.add_trace(go.Scatter(
          x=scales,
          y=est_err_upper,
          mode="lines+markers",
          name="est ci +2σ [σ=sqrt(24/Ns]",
          line=dict(color="black",dash="dash")
      ))

    fig3.update_layout(
        title=f"{title_prefix}: λ₂ across scales with bootstrap confidence intervals",
        xaxis_title="scale s",
        yaxis_title="λ₂",
    )
    fig3.show()


def plot_mutual_info(noise_generator, title_prefix, bootstrap_mi_func):
    x = noise_generator()
    y = noise_generator()

    mi_reg = mutual_info_regression(x.reshape(-1, 1), y)[0]
    mi_mean, mi_low, mi_high = bootstrap_mi_func(x, y)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=x, mode="lines", name="sample 1"))
    fig4.add_trace(go.Scatter(y=y, mode="lines", name="sample 2"))

    fig4.update_layout(
        title=(
            f"{title_prefix}: Mutual Information (KSG) = {mi_reg:.4f}<br>"
            f"Bootstrap CI: [{mi_low:.4f}, {mi_high:.4f}]"
        )
    )
    fig4.show()


# ============================================================
#   MAIN FUNCTION
# ============================================================
def analyze_noise(noise_generator,noise_len=N, bin_factor=1,detrend_factor=DETREND_FACTOR, title_prefix="Noise", intermittency_estimator=None, estimator_confidence_interval=None, **kwargs):
    """
    Runs full intermittency + MI analysis on a noise model.
    Produces:
        - per-scale increment/detrending plots
        - λ₂(s) (raw/detrended/std) with bootstrap CI
        - MI with bootstrap CI
    """

    print("\n=== Running noise analysis ===\n")
    print("Parameters")
    print(f"N={noise_len}")
    print(f"SCALES={SCALES}")
    print(f"detrend_factor = {detrend_factor} i.e. increment_window*detrend_factor=detrend_window")
    print(f"bin_factor = {bin_factor} i.e. #bins = bin_factor*sqrt(len)")
    print(f"CI_LEVEL={CI_LEVEL}")
    print(f"BOOTSTRAP_SAMPLES={BOOTSTRAP_SAMPLES}")
    # choose intermittency estimator (callable)
    estimator = intermittency_estimator if intermittency_estimator is not None else lambda2.estimate_lambda2
    # -------------------------
    # Generate noise
    # -------------------------
    data = noise_generator(N=noise_len,**kwargs)

    # -------------------------
    # Moving-average detrender
    # -------------------------
    detrender = moving_average.MovingAverageDetrender(10)

    # -------------------------
    # Precompute per-scale quantities
    # -------------------------
    increments_dict = {}
    detrended_dict = {}
    detrended_increments_dict = {}

    lambda2_raw_std = {}
    lambda2_detr_std = {}
    lambda2_raw_std_ci = {}
    lambda2_detr_std_ci = {}

    detrend_windows = {}

    # For global histogram ymax
    global_hist_ymax = 0.0

    for s in SCALES:
        # increments at scale s
        inc = increments.compute_increments(data, s)
        increments_dict[s] = inc

        # detrending window
        detrend_window = int(detrend_factor * s)
        detrend_windows[s] = detrend_window

        # detrend original series
        detrended, trend = detrend_series(data, detrender, detrend_window)
        detrended_dict[s] = detrended

        # increments of detrended series
        detr_inc = increments.compute_increments(detrended, s)
        detrended_increments_dict[s] = detr_inc

        # -------------------------
        # λ₂ estimates (raw, detrended, standardized)
        # -------------------------

        # standardized raw
        inc_std = (inc - np.mean(inc)) / np.std(inc)
        lambda2_raw_std[s] = estimator(inc_std)
        mean_raw_std, low_raw_std, high_raw_std = bootstrap_lambda2(inc_std, estimator=estimator)
        lambda2_raw_std_ci[s] = (low_raw_std, high_raw_std)

        # standardized detrended
        detr_inc_std = (detr_inc - np.mean(detr_inc)) / np.std(detr_inc)
        lambda2_detr_std[s] = estimator(detr_inc_std)
        mean_detr_std, low_detr_std, high_detr_std = bootstrap_lambda2(detr_inc_std, estimator=estimator)
        lambda2_detr_std_ci[s] = (low_detr_std, high_detr_std)

        # -------------------------
        # Precompute histogram ymax across scales (standardized)
        # -------------------------
        raw_std = inc_std
        detr_std = detr_inc_std

        N = len(raw_std)
        bins = int(bin_factor * np.sqrt(N))

        counts_raw, edges = np.histogram(raw_std, bins=bins, density=True)
        counts_detr, _ = np.histogram(detr_std, bins=edges, density=True)

        local_max = max(counts_raw.max(), counts_detr.max())
        if local_max > global_hist_ymax:
            global_hist_ymax = local_max

    global_hist_ymax *= 1.1  # padding

    # -------------------------
    # Plots (delegated to helper functions)
    # -------------------------
    plot_raw_noise(data, title_prefix)

    plot_per_scale(
        SCALES,
        increments_dict,
        detrended_dict,
        detrended_increments_dict,
        detrend_windows,
        lambda2_raw_std,
        lambda2_detr_std,
        lambda2_raw_std_ci,
        lambda2_detr_std_ci,
        bin_factor,
        global_hist_ymax,
        title_prefix,
    )

    plot_lambda2_with_ci(
        SCALES,
        lambda2_raw_std,
        lambda2_detr_std,
        lambda2_raw_std_ci,
        lambda2_detr_std_ci,
        increments_dict,
        estimator_confidence_interval,
        title_prefix,
    )

    plot_mutual_info(noise_generator, title_prefix, bootstrap_mi)

    print("\n=== Analysis complete ===\n")
