# xeeg_kit/epoch_cleaning.py
"""
Epoch-level artifact cleaning: RANSAC, AutoReject, and ICLabel-based ICA.

Filtering is NOT performed here. Data must be filtered on continuous
raw before epoching to avoid edge artifacts at epoch boundaries.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple

import mne
import numpy as np
import pandas as pd

try:
    from autoreject import AutoReject, Ransac, RejectLog
    HAS_AUTOREJECT = True
except ImportError:
    HAS_AUTOREJECT = False

try:
    from mne_icalabel import label_components
    HAS_ICLABEL = True
except ImportError:
    HAS_ICLABEL = False

from .viz import get_anatomical_summary, plot_bad_channels_3d, load_bel_channel_map

logger = logging.getLogger(__name__)

# ─── Default Parameters ──────────────────────────────────────────────────────
DEFAULT_ICA_COMP = 0.99
DEFAULT_ICA_SEED = 42
DEFAULT_ICALABEL_THRESH = 0.85
MIN_HIGHPASS_FOR_ICA = 1.0

DEFAULT_AR_N_INTERPOLATES = np.array([1, 4, 8, 16])
DEFAULT_AR_CONSENSUS_PERCS = np.linspace(0.0, 1.0, 11)
DEFAULT_AR_THRESH_METHOD = "bayesian_optimization"
DEFAULT_AR_CV = 10

DEFAULT_RANSAC_N_RESAMPLE = 50
DEFAULT_RANSAC_MIN_CHANNELS = 0.25
DEFAULT_RANSAC_MIN_CORR = 0.75
DEFAULT_RANSAC_UNBROKEN_TIME = 0.4


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
    n_jobs: int = 1,
    random_state: int = 42,
    generate_report: bool = False,
    report_dir: Optional[Union[str, Path]] = None,
    subject_id: str = "sub-unknown",
    verbose: bool = True,
) -> Tuple[mne.Epochs, List[str]]:
    """Detect globally bad channels using RANSAC (PREP pipeline method).

    Interpolates each channel from random subsets of neighbors and flags
    channels whose predicted signal consistently fails to correlate with
    the observed signal across epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data with montage set. Must be preloaded.
    n_resample : int
        Number of random channel subsets to draw.
    min_channels : float
        Fraction of channels used in each random subset.
    min_corr : float
        Minimum correlation threshold for a channel to be considered good.
    unbroken_time : float
        Fraction of epochs where correlation must exceed min_corr.
    n_jobs : int
        Number of parallel jobs.
    random_state : int
        Random seed for reproducibility.
    generate_report : bool
        Whether to generate a 3D anatomical report.
    report_dir : Path | None
        Directory for report output.
    subject_id : str
        Subject identifier for report filename.
    verbose : bool
        Verbosity flag.

    Returns
    -------
    epochs_clean : mne.Epochs
        Epochs with globally bad channels interpolated.
    bad_chs : list of str
        List of RANSAC-detected bad channel names.
    """
    if not HAS_AUTOREJECT:
        raise ImportError("The 'autoreject' package is required. Install via 'pip install autoreject'.")
    if not epochs.preload:
        raise ValueError("Epochs must be preloaded for RANSAC.")
    if epochs.get_montage() is None:
        raise ValueError("Epochs must have a montage set for RANSAC interpolation.")

    if verbose:
        logger.info("Running RANSAC for global bad channel detection...")

    ransac = Ransac(
        n_resample=n_resample,
        min_channels=min_channels,
        min_corr=min_corr,
        unbroken_time=unbroken_time,
        n_jobs=n_jobs,
        random_state=random_state,
        picks='eeg',
        verbose=verbose,
    )
    epochs_clean = ransac.fit_transform(epochs)
    bad_chs = ransac.bad_chs_

    if verbose:
        logger.info("RANSAC detected %d globally bad channels: %s", len(bad_chs), bad_chs)

    if generate_report and bad_chs:
        r_dir = Path(report_dir) if report_dir else Path.cwd()
        r_dir.mkdir(parents=True, exist_ok=True)
        _generate_bad_channel_report(bad_chs, r_dir, f"{subject_id}_ransac", epochs)

    return epochs_clean, bad_chs


def execute_autoreject(
    epochs: mne.Epochs,
    n_interpolates: Optional[np.ndarray] = None,
    consensus_percs: Optional[np.ndarray] = None,
    thresh_method: str = DEFAULT_AR_THRESH_METHOD,
    cv: int = DEFAULT_AR_CV,
    random_state: int = 42,
    n_jobs: int = 1,
    verbose: bool = True,
) -> Tuple[mne.Epochs, "RejectLog"]:
    """Run AutoReject for cross-validated epoch rejection and channel interpolation.

    Learns per-channel peak-to-peak thresholds via cross-validation, repairs
    bad channels within trials via interpolation, and drops trials where too
    many channels exceed thresholds.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data (ideally after RANSAC). Must be preloaded.
    n_interpolates : array-like | None
        Candidate values for rho (max channels to interpolate per trial).
    consensus_percs : array-like | None
        Candidate values for kappa (fraction of channels that must agree to drop).
    thresh_method : str
        'bayesian_optimization' or 'random_search'.
    cv : int
        Number of cross-validation folds.
    random_state : int
        Random seed.
    n_jobs : int
        Parallel jobs.
    verbose : bool
        Verbosity flag.

    Returns
    -------
    epochs_clean : mne.Epochs
        Cleaned epochs with bad trials dropped and bad channels interpolated.
    reject_log : RejectLog
        Detailed rejection log for QC.
    """
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
        picks='eeg',
        thresh_method=thresh_method,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

    n_original = len(epochs)
    n_clean = len(epochs_clean)
    n_dropped = n_original - n_clean
    drop_rate = (n_dropped / n_original * 100) if n_original > 0 else 0.0

    logger.info(
        "AutoReject complete. Dropped %d/%d epochs (%.1f%%).",
        n_dropped, n_original, drop_rate,
    )
    return epochs_clean, reject_log


def execute_icalabel_epochs(
    epochs: mne.Epochs,
    icalabel_thresholds: Optional[Dict[str, float]] = None,
    n_components: float = DEFAULT_ICA_COMP,
    random_state: int = DEFAULT_ICA_SEED,
    verbose: bool = True,
) -> mne.Epochs:
    """Fit and apply ICLabel-based ICA on epoched data.

    Best results when run AFTER an initial AutoReject pass. Enforces
    minimum highpass (must be applied to raw before epoching) and
    re-applies CAR before ICA fitting.

    Parameters
    ----------
    epochs : mne.Epochs
        Cleaned epochs (ideally after AutoReject).
    icalabel_thresholds : dict | None
        Mapping of artifact label to probability threshold for exclusion.
    n_components : float
        Variance explained threshold for ICA component selection.
    random_state : int
        Random seed for ICA reproducibility.
    verbose : bool
        Verbosity flag.

    Returns
    -------
    epochs_clean : mne.Epochs
        Epochs with ICA artifact components removed.
    """
    if not HAS_ICLABEL:
        raise ImportError("The 'mne-icalabel' package is required. Install via 'pip install mne-icalabel'.")

    if verbose:
        logger.info("Starting ICA + ICLabel on epochs...")

    if epochs.info.get('highpass') is None or epochs.info['highpass'] < MIN_HIGHPASS_FOR_ICA:
        raise ValueError(
            f"Data must be high-pass filtered at >= {MIN_HIGHPASS_FOR_ICA} Hz before epoching. "
            f"Current highpass: {epochs.info.get('highpass')}. "
            "Apply filters to continuous raw before creating epochs."
        )

    if icalabel_thresholds is None:
        icalabel_thresholds = {
            k: DEFAULT_ICALABEL_THRESH
            for k in ['eye blink', 'heart beat', 'muscle artifact', 'line noise', 'channel noise']
        }

    epochs_clean = epochs.copy().load_data()
    epochs_clean.set_eeg_reference('average', projection=False, verbose=False)

    try:
        logger.info("Fitting ICA on epochs (%.0f%% variance)...", n_components * 100)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            ica = mne.preprocessing.ICA(
                n_components=n_components,
                method='picard',
                fit_params=dict(ortho=False, extended=True),
                random_state=random_state,
                max_iter='auto',
            )
            ica.fit(epochs_clean, picks='eeg')
        logger.info("ICA fitted with %d components.", ica.n_components_)

        logger.info("Running ICLabel...")
        labels_dict = label_components(epochs_clean, ica, method='iclabel')

        excluded = [
            i for i, (label, prob_vec) in enumerate(
                zip(labels_dict['labels'], labels_dict['y_pred_proba'])
            )
            if label.lower().strip() in icalabel_thresholds
            and np.max(prob_vec) > icalabel_thresholds[label.lower().strip()]
        ]
        ica.exclude = sorted(set(excluded))

        if verbose and ica.exclude:
            logger.info("Excluding ICA components: %s", ica.exclude)
            for i in ica.exclude:
                logger.info(
                    "  C%02d: %-18s (%.2f)",
                    i, labels_dict['labels'][i],
                    np.max(labels_dict['y_pred_proba'][i]),
                )
        epochs_clean = ica.apply(epochs_clean)

    except Exception as e:
        logger.warning("ICA failed on epochs (%s). Skipping ICA step.", str(e)[:120])

    return epochs_clean


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
