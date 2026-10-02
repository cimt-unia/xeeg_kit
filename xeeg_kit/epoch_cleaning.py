# xeeg_kit/epoch_cleaning.py
"""
Epoch-level artifact cleaning: RANSAC and AutoReject.

Filtering is NOT performed here. Data must be filtered on continuous
raw before epoching to avoid edge artifacts at epoch boundaries.
ICA/ICLabel is intentionally excluded; stereotyped artifact removal
should be performed on continuous data before epoching.
"""

import logging
from pathlib import Path
from typing import Optional, List, Union, Tuple

import mne
import numpy as np
import pandas as pd

try:
    from autoreject import AutoReject, Ransac, RejectLog
    HAS_AUTOREJECT = True
except ImportError:
    HAS_AUTOREJECT = False

from .viz import get_anatomical_summary, plot_bad_channels_3d, load_bel_channel_map

logger = logging.getLogger(__name__)

# ─── Library Default Parameters ──────────────────────────────────────────────
DEFAULT_AR_N_INTERPOLATES = np.array([1, 4, 8, 16])
DEFAULT_AR_CONSENSUS_PERCS = np.linspace(0.0, 1.0, 11)
DEFAULT_AR_THRESH_METHOD = "bayesian_optimization"
DEFAULT_AR_CV = 10

# Conservative RANSAC defaults for high-density EEG (280-ch BEL).
DEFAULT_RANSAC_N_RESAMPLE = 50
DEFAULT_RANSAC_MIN_CHANNELS = 0.25
DEFAULT_RANSAC_MIN_CORR = 0.30
DEFAULT_RANSAC_UNBROKEN_TIME = 0.99


def _generate_bad_channel_report(
    bad_chs: List[str],
    report_dir: Path,
    subject_id: str,
    epochs: Optional[mne.Epochs] = None,
) -> None:
    """Generate 3D anatomical report for bad channels."""
    if not bad_chs:
        logger.info("No bad channels detected; skipping report generation.")
        return
    map_df = load_bel_channel_map()
    summary = get_anatomical_summary(bad_chs, map_df)
    logger.info("Anatomical Distribution:\n%s", summary)
    if epochs is not None:
        output_file = report_dir / f"{subject_id}_bad_channels_3d.html"
        plot_bad_channels_3d(epochs, bad_chs, map_df=map_df, output_file=str(output_file))
        logger.info("3D bad channel report saved to: %s", output_file)


def execute_ransac(
    epochs: mne.Epochs,
    n_resample: int = DEFAULT_RANSAC_N_RESAMPLE,
    min_channels: float = DEFAULT_RANSAC_MIN_CHANNELS,
    min_corr: float = DEFAULT_RANSAC_MIN_CORR,
    unbroken_time: float = DEFAULT_RANSAC_UNBROKEN_TIME,
    n_jobs: int = -1,
    random_state: int = 42,
    generate_report: bool = False,
    report_dir: Optional[Union[str, Path]] = None,
    subject_id: str = "sub-unknown",
    verbose: bool = True,
) -> Tuple[mne.Epochs, List[str]]:
    """Detect globally bad channels using RANSAC, interpolate them, and clear bads list."""
    if not HAS_AUTOREJECT:
        raise ImportError("The 'autoreject' package is required. Install via 'pip install autoreject'.")
    if not epochs.preload:
        raise ValueError("Epochs must be preloaded for RANSAC.")
    if epochs.get_montage() is None:
        raise ValueError("Epochs must have a montage set for RANSAC interpolation.")

    if verbose:
        logger.info("Running RANSAC for global bad channel detection...")

    ransac = Ransac(
        n_resample=n_resample, min_channels=min_channels, min_corr=min_corr,
        unbroken_time=unbroken_time, n_jobs=n_jobs, random_state=random_state,
        picks='eeg', verbose=verbose,
    )
    # fit_transform internally interpolates bad channels AND resets info['bads']
    epochs_clean = ransac.fit_transform(epochs)
    bad_chs = ransac.bad_chs_

    if verbose:
        logger.info("RANSAC detected and interpolated %d globally bad channels.", len(bad_chs))

    if generate_report and bad_chs:
        r_dir = Path(report_dir) if report_dir else Path.cwd()
        r_dir.mkdir(parents=True, exist_ok=True)
        _generate_bad_channel_report(bad_chs, r_dir, f"{subject_id}_ransac", epochs)

    # Ensure bads list is empty so AutoReject evaluates ALL channels
    epochs_clean.info['bads'] = []
    if verbose and bad_chs:
        logger.info(
            "Cleared bads list after RANSAC interpolation. "
            "%d channels now available for AutoReject CV.",
            len(epochs_clean.ch_names),
        )

    return epochs_clean, bad_chs


def execute_autoreject(
    epochs: mne.Epochs,
    n_interpolates: Optional[np.ndarray] = None,
    consensus_percs: Optional[np.ndarray] = None,
    thresh_method: str = DEFAULT_AR_THRESH_METHOD,
    cv: int = DEFAULT_AR_CV,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True,
) -> Tuple[mne.Epochs, "RejectLog"]:
    """Run AutoReject for cross-validated epoch rejection and channel interpolation."""
    if not HAS_AUTOREJECT:
        raise ImportError("The 'autoreject' package is required. Install via 'pip install autoreject'.")
    if not epochs.preload:
        raise ValueError("Epochs must be preloaded to use AutoReject.")
    if epochs.get_montage() is None:
        raise ValueError("Epochs must have a montage set for AutoReject interpolation.")

    if verbose:
        logger.info("Starting AutoReject cleaning pipeline...")

    ar = AutoReject(
        n_interpolate=n_interpolates if n_interpolates is not None else DEFAULT_AR_N_INTERPOLATES,
        consensus=consensus_percs if consensus_percs is not None else DEFAULT_AR_CONSENSUS_PERCS,
        picks='eeg', thresh_method=thresh_method, cv=cv,
        random_state=random_state, n_jobs=n_jobs, verbose=verbose,
    )
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

    n_original = len(epochs)
    n_dropped = n_original - len(epochs_clean)
    drop_rate = (n_dropped / n_original * 100) if n_original > 0 else 0.0

    logger.info(
        "AutoReject complete. Dropped %d/%d epochs (%.1f%%).",
        n_dropped, n_original, drop_rate,
    )
    return epochs_clean, reject_log


def save_epoch_qc_report(
    reject_log: "RejectLog",
    report_dir: Path,
    subject_id: str,
) -> None:
    """Save QC metrics and full rejection log for a single subject."""
    report_dir.mkdir(parents=True, exist_ok=True)

    n_original = len(reject_log.bad_epochs)
    n_dropped = int(np.sum(reject_log.bad_epochs))
    n_interp_per_epoch = np.sum(reject_log.labels == 2, axis=1)

    qc_data = {
        "subject_id": subject_id,
        "n_original_epochs": n_original,
        "n_dropped_epochs": n_dropped,
        "drop_rate_pct": round((n_dropped / n_original * 100), 2) if n_original > 0 else 0.0,
        "mean_interp_channels_per_epoch": float(np.mean(n_interp_per_epoch)),
        "max_interp_channels_per_epoch": int(np.max(n_interp_per_epoch)),
    }

    csv_path = report_dir / f"{subject_id}_epoch_qc.csv"
    pd.DataFrame([qc_data]).to_csv(csv_path, index=False)
    logger.info("Saved QC summary: %s", csv_path)

    log_path = report_dir / f"{subject_id}_reject_log.npz"
    reject_log.save(str(log_path), overwrite=True)
    logger.info("Saved full reject log: %s", log_path)
