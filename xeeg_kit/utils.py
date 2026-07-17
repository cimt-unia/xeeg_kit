# xeeg_kit/utils.py

# General-purpose utilities and mathematical helpers for EEG processing.
import logging
from typing import List, Tuple
import numpy as np
from scipy.stats import median_abs_deviation
import mne

logger = logging.getLogger(__name__)

DEFAULT_MAD_THRESHOLD = 15.0
DEFAULT_MIN_AMPLITUDE_UV = 0.1
DEFAULT_CLEANEST_DURATION = 30.0
DEFAULT_CLEANEST_STEP = 2.0
DEFAULT_CLEANEST_START = 0.0

def detect_bad_channels(
    raw: mne.io.Raw,
    mad_threshold: float = DEFAULT_MAD_THRESHOLD,
    min_amplitude_uv: float = DEFAULT_MIN_AMPLITUDE_UV
) -> List[str]:
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
    raw: mne.io.Raw,
    duration_sec: float = DEFAULT_CLEANEST_DURATION,
    step_sec: float = DEFAULT_CLEANEST_STEP,
    start_sec: float = DEFAULT_CLEANEST_START
) -> Tuple[np.ndarray, float]:
    sfreq = raw.info["sfreq"]
    duration_samp = int(duration_sec * sfreq)
    step_samp = int(step_sec * sfreq)
    total_samp = raw.n_times

    if total_samp < duration_samp:
        logger.warning("Trial too short (%.1fs). Using full trial.", total_samp / sfreq)
        return raw.get_data(), 0.0

    data_v = raw.get_data()
    start_samp = int(start_sec * sfreq)
    n_windows = (total_samp - duration_samp - start_samp) // step_samp + 1
    assert n_windows > 0, "Invalid window calculation parameters."

    variances = []
    amplitudes = []

    for i in range(n_windows):
        start = i * step_samp + start_samp
        end = start + duration_samp
        win = data_v[:, start:end]
        if np.ptp(win) < 1e-9:
            continue
        var_metric = np.median(np.var(win, axis=1))
        amp_metric = np.median(np.ptp(win, axis=1))
        if np.isfinite(var_metric) and np.isfinite(amp_metric):
            variances.append(var_metric)
            amplitudes.append(amp_metric)

    if not variances:
        logger.warning("No valid windows found. Using first segment as calibration.")
        return data_v[:, :duration_samp], 0.0

    variances_arr = np.array(variances)
    amplitudes_arr = np.array(amplitudes)

    def mad_zscore(x: np.ndarray) -> np.ndarray:
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

    score = mad_zscore(variances_arr) + mad_zscore(amplitudes_arr)
    best_idx = int(np.argmin(score))
    best_start = best_idx * step_samp + start_samp
    calib_data_v = data_v[:, best_start:best_start + duration_samp]
    start_time = best_start / sfreq
    
    logger.info("Cleanest segment at t=%.1fs (score=%.2f)", start_time, score[best_idx])
    return calib_data_v, start_time



