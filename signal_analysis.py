import numpy as np
import pywt
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

try:
    import multiprocess as mp
except ImportError:  # pragma: no cover
    mp = None


def compute_wavelet_details(signal, wavelet="haar", max_level=10):
    """Compute discrete wavelet detail series for a 1D signal."""
    signal = np.asarray(signal, dtype=float)
    coeffs = pywt.wavedec(signal, wavelet, level=max_level)
    detail_series = []
    scales = [2**i for i in range(1, max_level + 1)]

    for level in range(1, max_level + 1):
        cD = coeffs[-level]
        up = pywt.upcoef("d", cD, wavelet, level=level, take=len(signal))
        detail_series.append(up)

    return np.array(detail_series), np.array(scales)


def compute_local_volatility(detail_series, window=50):
    """Compute a local volatility series from wavelet detail coefficients."""
    vol_series = []
    for d in detail_series:
        sq = d**2
        vol = np.sqrt(pd.Series(sq).rolling(window, min_periods=1).mean().values)
        vol_series.append(vol)
    return np.array(vol_series)


def compute_log_volatility(detail_series, window=50, eps=1e-12):
    """Return log-volatility series for each scale."""
    vol_series = compute_local_volatility(detail_series, window)
    return np.log(vol_series + eps)


def center_series(series):
    """Center a 1D array by subtracting its mean."""
    arr = np.asarray(series, dtype=float)
    return arr - np.mean(arr)


def _compute_mi_task(args):
    s_idx, dt = args
    w_other = _worker_w_list[s_idx]
    w_ref = _worker_w_ref

    if dt >= 0:
        x = w_other[:-dt or None]
        y = w_ref[dt:]
    else:
        x = w_other[-dt:]
        y = w_ref[:dt or None]

    if len(x) < 10:
        return 0.0

    return float(mutual_info_regression(x.reshape(-1, 1), y, n_jobs=1)[0])


def _init_worker(w_list, w_ref):
    global _worker_w_list, _worker_w_ref
    _worker_w_list = w_list
    _worker_w_ref = w_ref


def compute_mutual_information_map(
    log_vol_series,
    ref_idx=0,
    max_time_lag=2,
    use_parallel=False,
    n_jobs=None,
):
    """Compute the mutual information map between scales and time lags."""
    log_vol_series = [center_series(log_vol) for log_vol in log_vol_series]
    w_ref = log_vol_series[ref_idx]

    tasks = [(s_idx, dt) for s_idx in range(len(log_vol_series)) for dt in range(-max_time_lag, max_time_lag + 1)]

    if use_parallel:
        if mp is None:
            raise ImportError("multiprocess is not installed. Install it or use use_parallel=False.")
        if n_jobs is None:
            n_jobs = mp.cpu_count()

        with mp.Pool(processes=n_jobs, initializer=_init_worker, initargs=(log_vol_series, w_ref)) as pool:
            results = pool.map(_compute_mi_task, tasks)
    else:
        _init_worker(log_vol_series, w_ref)
        results = [_compute_mi_task(task) for task in tasks]

    I_map = np.zeros((len(log_vol_series), 2 * max_time_lag + 1))
    for (s_idx, dt), value in zip(tasks, results):
        I_map[s_idx, dt + max_time_lag] = value

    return I_map


def analyze_signal(
    signal,
    wavelet="haar",
    max_level=10,
    window=50,
    ref_idx=0,
    max_time_lag=2,
    use_parallel=False,
    n_jobs=None,
):
    """Analyze one signal and return log-volatility and mutual-information results."""
    detail_series, scales = compute_wavelet_details(signal, wavelet=wavelet, max_level=max_level)
    log_vol_series = compute_log_volatility(detail_series, window=window)
    mi_map = compute_mutual_information_map(
        log_vol_series,
        ref_idx=ref_idx,
        max_time_lag=max_time_lag,
        use_parallel=use_parallel,
        n_jobs=n_jobs,
    )

    return {
        "signal": signal,
        "detail_series": detail_series,
        "scales": scales,
        "log_vol_series": log_vol_series,
        "mi_map": mi_map,
    }
