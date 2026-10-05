# xeeg_kit/pre_cleaning.py
"""Pre-cleaning toolkit for BEL EEG data.

Provides two modes of operation:

1. Interactive (Jupyter): inspect_bad_channels → fit_ica_with_labels →
   apply_pre_cleaning. Designed for manual review and component selection.

2. Automated conservative: auto_preclean. Single-call pipeline with high
   thresholds for use before epoching. Filters real data first, then removes
   genuinely broken channels and unambiguous artifacts; defers trial-specific
   decisions to downstream MEEGKit cleaning on concatenated epochs.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np

from xeeg_kit.artifact_cleaning import (
    DEFAULT_HIGHPASS,
    DEFAULT_LOWPASS,
    DEFAULT_NOTCH,
    DEFAULT_ICALABEL_THRESH,
    MIN_HIGHPASS_FOR_ICA,
)
from xeeg_kit.utils import detect_bad_channels
from xeeg_kit.viz import (
    get_anatomical_summary,
    load_bel_channel_map,
    plot_bad_channels_3d,
)

logger = logging.getLogger(__name__)


def inspect_bad_channels(
    raw: mne.io.Raw,
    mad_threshold: float = 25.0,
    min_amplitude_uv: float = 0.1,
    manual_bads: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    subject_id: str = "sub-unknown",
) -> List[str]:
    """Detect bad channels automatically and generate 3D anatomical report.

    Uses xeeg_kit's detect_bad_channels plus BEL anatomical mapping to help
    you decide which channels to mark as bad before ICA.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data with montage already set. Should be filtered before
        calling this function to avoid drift-inflated MAD scores.
    mad_threshold : float
        MAD z-score threshold for noisy channel detection.
    min_amplitude_uv : float
        Peak-to-peak amplitude below which a channel is considered flat.
    manual_bads : list of str, optional
        Additional channels to flag regardless of auto-detection.
    output_dir : Path, optional
        Directory to save the 3D HTML report. If None, uses cwd.
    subject_id : str
        Subject identifier for report filename.

    Returns
    -------
    bad_channels : list of str
        Combined list of auto-detected and manual bad channel names.
        Call ``raw.info['bads'] = bad_channels`` to apply.
    """
    auto_bads = detect_bad_channels(
        raw, mad_threshold=mad_threshold, min_amplitude_uv=min_amplitude_uv
    )
    all_bads = sorted(set(auto_bads + (manual_bads or [])))

    map_df = load_bel_channel_map()
    summary = get_anatomical_summary(all_bads, map_df)
    logger.info("Bad channel candidates (%d):\n%s", len(all_bads), summary)

    out_dir = Path(output_dir) if output_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{subject_id}_bad_channels_3d.html"
    plot_bad_channels_3d(raw, all_bads, map_df=map_df, output_file=str(report_path))
    logger.info("3D bad-channel report: %s", report_path)

    return all_bads


def fit_ica_with_labels(
    raw: mne.io.Raw,
    n_components: float = 0.99,
    highpass: float = DEFAULT_HIGHPASS,
    lowpass: float = DEFAULT_LOWPASS,
    notch_freq: float = DEFAULT_NOTCH,
    random_state: int = 42,
) -> Tuple[mne.preprocessing.ICA, Dict]:
    """Filter data, fit ICA, and run ICLabel.

    For interactive use: filters an internal copy, fits ICA, returns labels
    for manual inspection. Does NOT log per-component classifications.

    For automated use (auto_preclean): safe to call on already-filtered data;
    internal filtering is harmless/idempotent.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data. Will be copied internally; original is not modified.
    n_components : float
        Variance explained threshold for ICA.
    highpass, lowpass, notch_freq : float
        Filter parameters required for valid ICLabel classification.
    random_state : int
        Random seed for ICA reproducibility.

    Returns
    -------
    ica : mne.preprocessing.ICA
        Fitted ICA instance. Inspect with ``ica.plot_sources(raw_for_ica)``.
    labels_dict : dict
        ICLabel output with keys 'labels' and 'y_pred_proba'.
    """
    raw_filt = raw.copy().load_data()
    raw_filt.filter(l_freq=highpass, h_freq=lowpass, picks="eeg", n_jobs=1, verbose=False)
    nyquist = raw_filt.info["sfreq"] / 2.0
    notch_freqs = [f for f in [notch_freq, notch_freq * 2.0] if f <= min(lowpass, nyquist)]
    if notch_freqs:
        raw_filt.notch_filter(freqs=notch_freqs, picks="eeg", method="fir", verbose=False)
    raw_filt._data = np.real(raw_filt._data).astype(np.float64)

    raw_filt.set_eeg_reference("average", projection=False, verbose=False)

    logger.info("Fitting ICA (%.0f%% variance)...", n_components * 100)
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="picard",
        fit_params=dict(ortho=False, extended=True),
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw_filt, picks="eeg")
    logger.info("ICA fitted: %d components.", ica.n_components_)

    from mne_icalabel import label_components
    labels_dict = label_components(raw_filt, ica, method="iclabel")

    return ica, labels_dict


def apply_pre_cleaning(
    raw: mne.io.Raw,
    ica: mne.preprocessing.ICA,
    bad_channels: List[str],
    output_path: Path,
    interpolate_bads: bool = True,
    overwrite: bool = True,
) -> mne.io.Raw:
    """Apply ICA exclusion, interpolate bad channels, re-reference, and save.

    Call this AFTER you have inspected the data and set:
      - ``raw.info['bads']`` (via inspect_bad_channels or manually)
      - ``ica.exclude`` (via fit_ica_with_labels + manual review)

    Parameters
    ----------
    raw : mne.io.Raw
        Original raw data (will be copied internally).
    ica : mne.preprocessing.ICA
        Fitted ICA with ``exclude`` list set by user.
    bad_channels : list of str
        Channels to interpolate.
    output_path : Path
        Where to save the pre-cleaned FIF file.
    interpolate_bads : bool
        Whether to interpolate bad channels after ICA application.
    overwrite : bool
        Whether to overwrite existing output file.

    Returns
    -------
    cleaned_raw : mne.io.Raw
        Pre-cleaned data ready for downstream automatic pipeline.
    """
    cleaned = raw.copy().load_data()

    cleaned.info["bads"] = list(bad_channels)
    logger.info("Applying ICA exclusion: %s", ica.exclude)
    cleaned = ica.apply(cleaned)

    if interpolate_bads and cleaned.info["bads"]:
        cleaned.interpolate_bads(reset_bads=True)
        logger.info("Interpolated %d bad channels.", len(bad_channels))

    cleaned.set_eeg_reference("average", projection=False, verbose=False)
    logger.info("Average reference applied.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.save(str(output_path), overwrite=overwrite, verbose=False)
    logger.info("Pre-cleaned data saved: %s", output_path)

    return cleaned


# ─── Automated Conservative Pre-Cleaning ─────────────────────────────────────

CONSERVATIVE_MAD_THRESH: float = 50.0
CONSERVATIVE_ARTIFACT_THRESH: float = 0.85
CONSERVATIVE_ARTIFACT_CLASSES = {
    "eye blink", "muscle artifact", "heart beat", "line noise", "channel noise",
}

def auto_preclean(
    raw: mne.io.Raw,
    output_dir: Path,
    subject_id: str,
    mad_threshold: float = CONSERVATIVE_MAD_THRESH,
    artifact_threshold: float = CONSERVATIVE_ARTIFACT_THRESH,
    n_components: float = 0.99,
    highpass: float = DEFAULT_HIGHPASS,
    lowpass: float = DEFAULT_LOWPASS,
    notch_freq: float = DEFAULT_NOTCH,
    random_state: int = 42,
    drop_channels: Optional[List[str]] = None,
    overwrite: bool = True,
) -> mne.io.Raw:
    """Apply conservative automated pre-cleaning to full continuous data.

    Pipeline order:
      0. Drop specified channels (e.g., hardware reference + jaw)
      1. Filter real data (highpass + lowpass + notch) ← APPLIED TO SAVED OUTPUT
      2. Bad channel detection (MAD on filtered data)
      3. ICA fitting + ICLabel (on already-filtered data)
      4. Auto-exclusion of artifact components
      5. ICA application + bad channel interpolation + CAR + save

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data with montage already set. Must be standardized
        (e.g., via BELStandardizer) BEFORE calling this function.
    output_dir : Path
        Directory to save reports and cleaned FIF.
    subject_id : str
        Subject identifier for filenames.
    mad_threshold : float
        MAD z-score for bad channel detection. Default 50.0 (conservative).
    artifact_threshold : float
        ICLabel probability for auto-exclusion. Default 0.85.
    n_components : float
        Variance explained for ICA. Default 0.99.
    highpass : float
        High-pass filter Hz. Default 1.0.
    lowpass : float
        Low-pass filter Hz. Default 100.0.
    notch_freq : float
        Notch filter base Hz. Default 60.0.
    random_state : int
        Random seed for ICA.
    drop_channels : list of str | None
        Channel names to drop before cleaning. Use this to remove hardware
        reference channels (e.g., ['Cz'] for BEL 280) and non-neural sensors
        (e.g., jaw EMG channels) that are redundant after CAR. If None, no
        channels are dropped.
    overwrite : bool
        Overwrite existing output files.

    Returns
    -------
    cleaned_raw : mne.io.Raw
        Conservatively pre-cleaned and filtered data ready for epoching.

    Raises
    ------
    RuntimeError
        If ALL requested drop_channels are missing from raw.ch_names,
        indicating that BELStandardizer.standardize() was likely not called.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: Drop specified channels (e.g., hardware reference + jaw)
    if drop_channels:
        # FIX: Compute BOTH lists BEFORE mutating raw.ch_names via drop_channels()
        existing = [ch for ch in drop_channels if ch in raw.ch_names]
        missing = [ch for ch in drop_channels if ch not in raw.ch_names]

        if existing:
            raw.drop_channels(existing)
            logger.info("Dropped %d channel(s) before cleaning: %s", len(existing), existing)

        if missing:
            # Fail hard if ALL requested channels are missing (likely un-standardized data)
            if len(missing) == len(drop_channels):
                raise RuntimeError(
                    f"NONE of the requested drop_channels were found in data. "
                    f"Requested: {drop_channels}. "
                    f"Available (first 10): {raw.ch_names[:10]}. "
                    f"Ensure BELStandardizer.standardize() was called before auto_preclean()."
                )
            logger.warning("Some requested drop channels not found: %s", missing)

    # Step 1: Filter REAL data before MAD and ICA
    logger.info("Applying filters: %.1f–%.1f Hz + %.0f Hz notch", highpass, lowpass, notch_freq)
    raw.filter(l_freq=highpass, h_freq=lowpass, picks="eeg", n_jobs=1, verbose=False)
    nyquist = raw.info["sfreq"] / 2.0
    notch_freqs = [f for f in [notch_freq, notch_freq * 2.0] if f <= min(lowpass, nyquist)]
    if notch_freqs:
        raw.notch_filter(freqs=notch_freqs, picks="eeg", method="fir", verbose=False)
    raw._data = np.real(raw._data).astype(np.float64)

    logger.info("=" * 60)
    logger.info("Auto Pre-Clean: %s", subject_id)
    logger.info("  Channels: %d", len(raw.ch_names))
    logger.info("  Filter: %.1f–%.1f Hz + %.0f Hz notch (applied)", highpass, lowpass, notch_freq)
    logger.info("  MAD: %.1f | ICA: %.0f%% var | ICLabel: %.2f",
                mad_threshold, n_components * 100, artifact_threshold)
    logger.info("=" * 60)

    # Step 2: Bad channel detection (now on filtered data)
    bads = inspect_bad_channels(
        raw,
        mad_threshold=mad_threshold,
        min_amplitude_uv=0.1,
        manual_bads=None,
        output_dir=output_dir,
        subject_id=f"{subject_id}_autoclean",
    )
    raw.info["bads"] = bads

    # Step 3: ICA + ICLabel (data already filtered; internal copy filtering is harmless)
    ica, labels = fit_ica_with_labels(
        raw,
        n_components=n_components,
        highpass=highpass,
        lowpass=lowpass,
        notch_freq=notch_freq,
        random_state=random_state,
    )

    # Auto-exclude artifacts above threshold
    auto_exclude = [
        i for i, (label, prob_vec) in enumerate(zip(labels["labels"], labels["y_pred_proba"]))
        if label.lower().strip() in CONSERVATIVE_ARTIFACT_CLASSES
        and np.max(prob_vec) > artifact_threshold
    ]
    logger.info("Auto-excluding %d components (>%.2f):", len(auto_exclude), artifact_threshold)
    for i in auto_exclude:
        logger.info("  IC%02d: %s (%.3f)", i, labels["labels"][i], np.max(labels["y_pred_proba"][i]))

    ica.exclude = sorted(set(auto_exclude))

    # Step 4: Apply cleaning
    output_path = output_dir / f"{subject_id}_preclean_raw.fif"
    cleaned = apply_pre_cleaning(
        raw=raw,
        ica=ica,
        bad_channels=bads,
        output_path=output_path,
        interpolate_bads=True,
        overwrite=overwrite,
    )

    logger.info("Auto pre-clean complete: %s (%d channels)", output_path.name, len(cleaned.ch_names))
    return cleaned
