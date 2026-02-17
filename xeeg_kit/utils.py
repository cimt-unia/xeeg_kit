# xeeg_kit/utils.py
"""
General-purpose utilities shared across modules.
"""
import time
import os
from typing import List, Tuple, Dict, Any
import numpy as np
from scipy.stats import median_abs_deviation
from pathlib import Path


def _log(msg: str, subject_id: Optional[str] = None):
    """Internal logger with optional subject prefix."""
    timestamp = time.strftime("%H:%M:%S")
    if subject_id:
        print(f"[{timestamp}] [{subject_id}] {msg}")
    else:
        print(f"[{timestamp}] {msg}")


def detect_bad_channels(
    raw: 'mne.io.Raw',
    mad_threshold: float = 10.0,
    min_amplitude_uv: float = 0.1
) -> List[str]:
    """Detect flat and noisy EEG channels."""
    import mne
    raw_eeg = raw.copy().pick("eeg")
    data_uv = raw_eeg.get_data() * 1e6
    amplitude = np.ptp(data_uv, axis=1)
    variance = np.var(data_uv, axis=1)

    flat_mask = amplitude < min_amplitude_uv
    flat_chs = [ch for ch, is_flat in zip(raw_eeg.ch_names, flat_mask) if is_flat]

    noisy_mask = np.zeros(len(amplitude), dtype=bool)
    for feat in (variance, amplitude):
        mad = median_abs_deviation(feat, scale="normal", nan_policy="omit")
        if not np.isnan(mad) and mad > 1e-12:
            z = (feat - np.nanmedian(feat)) / mad
            noisy_mask |= z > mad_threshold
    noisy_chs = [ch for ch, is_noisy in zip(raw_eeg.ch_names, noisy_mask) if is_noisy]

    return sorted(set(flat_chs + noisy_chs))

def find_cleanest_segment(
    raw: 'mne.io.Raw',
    duration_sec: float = 30.0,
    step_sec: float = 2.0,
    start_sec: float = 0.0
) -> Tuple[np.ndarray, float]:
    """Find the cleanest continuous segment for ASR calibration."""
    from .utils import log  # Avoid circular import; or keep log here
    sfreq = raw.info["sfreq"]
    duration_samp = int(duration_sec * sfreq)
    step_samp = int(step_sec * sfreq)
    total_samp = raw.n_times

    if total_samp < duration_samp:
        log(f"⚠️ Trial too short ({total_samp / sfreq:.1f}s). Using full trial.")
        return raw.get_data(), 0.0

    data_v = raw.get_data()
    start_samp = int(start_sec * sfreq)
    n_windows = (total_samp - duration_samp - start_samp) // step_samp + 1

    variances = []
    amplitudes = []

    for i in range(n_windows):
        start, end = i * step_samp + start_samp, i * step_samp + start_samp + duration_samp
        win = data_v[:, start:end]
        if np.ptp(win) < 1e-9:
            continue
        var_metric = np.median(np.var(win, axis=1))
        amp_metric = np.median(np.ptp(win, axis=1))
        if np.isfinite(var_metric) and np.isfinite(amp_metric):
            variances.append(var_metric)
            amplitudes.append(amp_metric)

    if not variances:
        log("⚠️ No valid windows found. Using first segment as calibration.")
        return data_v[:, :duration_samp], 0.0

    variances = np.array(variances)
    amplitudes = np.array(amplitudes)

    def mad_zscore(x):
        x = np.atleast_1d(x)
        x_clean = x[np.isfinite(x)]
        if len(x_clean) == 0:
            return np.zeros_like(x)
        med = np.median(x_clean)
        mad = median_abs_deviation(x_clean, scale='normal', nan_policy='omit')
        if not np.isfinite(mad) or mad == 0:
            return np.zeros_like(x)
        z = (x - med) / mad
        z[~np.isfinite(z)] = 0
        return z

    score = mad_zscore(variances) + mad_zscore(amplitudes)
    best_idx = np.argmin(score)
    best_start = best_idx * step_samp + start_samp
    calib_data_v = data_v[:, best_start:best_start + duration_samp]
    start_time = best_start / sfreq
    log(f"✅ Cleanest segment at t={start_time:.1f}s (score={score[best_idx]:.2f})")
    return calib_data_v, start_time



