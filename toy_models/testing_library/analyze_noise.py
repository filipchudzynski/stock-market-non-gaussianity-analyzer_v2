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
# λ₂ estimator
# -------------------------
def estimate_lambda2(inc):
    inc = inc[~np.isnan(inc)]
    if len(inc) < MIN_SAMPLES:
        return np.nan
    return kurtosis(inc, fisher=True)


# -------------------------
# Bootstrap λ₂(s)
# -------------------------
def bootstrap_lambda2(inc, B=BOOTSTRAP_SAMPLES, ci=CI_LEVEL):
    inc = inc[~np.isnan(inc)]
    n = len(inc)
    if n < MIN_SAMPLES:
        return np.nan, np.nan, np.nan

    stats = []
    for _ in range(B):
        sample = np.random.choice(inc, size=n, replace=True)
        stats.append(estimate_lambda2(sample))

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


# ============================================================
#   MAIN FUNCTION
# ============================================================
def analyze_noise(noise_generator,noise_len=N, bin_factor=1,detrend_factor=DETREND_FACTOR, title_prefix="Noise"):
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
    # -------------------------
    # Generate noise
    # -------------------------
    data = noise_generator(N=noise_len)

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

    lambda2_raw = {}
    lambda2_detr = {}
    lambda2_raw_ci = {}
    lambda2_detr_ci = {}

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

        # raw
        lambda2_raw[s] = estimate_lambda2(inc)
        mean_raw, low_raw, high_raw = bootstrap_lambda2(inc)
        lambda2_raw_ci[s] = (low_raw, high_raw)

        # detrended
        lambda2_detr[s] = estimate_lambda2(detr_inc)
        mean_detr, low_detr, high_detr = bootstrap_lambda2(detr_inc)
        lambda2_detr_ci[s] = (low_detr, high_detr)

        # standardized raw
        inc_std = (inc - np.mean(inc)) / np.std(inc)
        lambda2_raw_std[s] = estimate_lambda2(inc_std)
        mean_raw_std, low_raw_std, high_raw_std = bootstrap_lambda2(inc_std)
        lambda2_raw_std_ci[s] = (low_raw_std, high_raw_std)

        # standardized detrended
        detr_inc_std = (detr_inc - np.mean(detr_inc)) / np.std(detr_inc)
        lambda2_detr_std[s] = estimate_lambda2(detr_inc_std)
        mean_detr_std, low_detr_std, high_detr_std = bootstrap_lambda2(detr_inc_std)
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
    # Plot raw noise
    # -------------------------
    fig0 = px.line(y=data, title=f"{title_prefix}: Generated Noise")
    fig0.show()

    # -------------------------
    # Per-scale plots
    # -------------------------
    for s in SCALES:

        subplot_title = (
            f"s={s} — Distributions<br>"
            f"λ₂ raw={lambda2_raw[s]:.3f} "
            f"[{lambda2_raw_ci[s][0]:.3f}, {lambda2_raw_ci[s][1]:.3f}]<br>"
            f"λ₂ detr={lambda2_detr[s]:.3f} "
            f"[{lambda2_detr_ci[s][0]:.3f}, {lambda2_detr_ci[s][1]:.3f}]<br>"
            f"λ₂ raw_std={lambda2_raw_std[s]:.3f} "
            f"[{lambda2_raw_std_ci[s][0]:.3f}, {lambda2_raw_std_ci[s][1]:.3f}]<br>"
            f"λ₂ detr_std={lambda2_detr_std[s]:.3f} "
            f"[{lambda2_detr_std_ci[s][0]:.3f}, {lambda2_detr_std_ci[s][1]:.3f}]"
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

    # -------------------------
    # λ₂(s) with CI (all four variants)
    # -------------------------
    fig3 = go.Figure()

    # raw
    fig3.add_trace(go.Scatter(
        x=SCALES,
        y=[lambda2_raw[s] for s in SCALES],
        mode="lines+markers",
        name="raw increments",
        line=dict(color="blue")
    ))
    fig3.add_trace(go.Scatter(
        x=SCALES + SCALES[::-1],
        y=[lambda2_raw_ci[s][0] for s in SCALES] +
          [lambda2_raw_ci[s][1] for s in SCALES[::-1]],
        fill="toself",
        fillcolor="rgba(0,0,255,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="raw increments CI"
    ))

    # detrended
    fig3.add_trace(go.Scatter(
        x=SCALES,
        y=[lambda2_detr[s] for s in SCALES],
        mode="lines+markers",
        name="detrended increments",
        line=dict(color="red")
    ))
    fig3.add_trace(go.Scatter(
        x=SCALES + SCALES[::-1],
        y=[lambda2_detr_ci[s][0] for s in SCALES] +
          [lambda2_detr_ci[s][1] for s in SCALES[::-1]],
        fill="toself",
        fillcolor="rgba(255,0,0,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="detrended increments CI"
    ))

    # standardized raw
    fig3.add_trace(go.Scatter(
        x=SCALES,
        y=[lambda2_raw_std[s] for s in SCALES],
        mode="lines+markers",
        name="raw increments (std)",
        line=dict(color="green")
    ))
    fig3.add_trace(go.Scatter(
        x=SCALES + SCALES[::-1],
        y=[lambda2_raw_std_ci[s][0] for s in SCALES] +
          [lambda2_raw_std_ci[s][1] for s in SCALES[::-1]],
        fill="toself",
        fillcolor="rgba(0,255,0,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="raw increments (std) CI"
    ))

    # standardized detrended
    fig3.add_trace(go.Scatter(
        x=SCALES,
        y=[lambda2_detr_std[s] for s in SCALES],
        mode="lines+markers",
        name="detrended increments (std)",
        line=dict(color="purple")
    ))
    fig3.add_trace(go.Scatter(
        x=SCALES + SCALES[::-1],
        y=[lambda2_detr_std_ci[s][0] for s in SCALES] +
          [lambda2_detr_std_ci[s][1] for s in SCALES[::-1]],
        fill="toself",
        fillcolor="rgba(128,0,128,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="detrended increments (std) CI"
    ))

    fig3.update_layout(
        title=f"{title_prefix}: λ₂ across scales with bootstrap confidence intervals",
        xaxis_title="scale s",
        yaxis_title="λ₂",
    )
    fig3.show()

    # -------------------------
    # Mutual information with CI
    # -------------------------
    x = noise_generator()
    y = noise_generator()

    mi_reg = mutual_info_regression(x.reshape(-1, 1), y)[0]
    mi_mean, mi_low, mi_high = bootstrap_mi(x, y)

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

    print("\n=== Analysis complete ===\n")
