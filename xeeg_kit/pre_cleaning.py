# xeeg_kit/pre_cleaning.py

"""Interactive pre-cleaning toolkit for BEL EEG data.

Provides modular functions for semi-automated artifact removal:
1. Visualize and confirm bad channels using xeeg_kit anatomical maps.
2. Fit ICA with ICLabel assistance and manually select components.
3. Apply cleaning, interpolate, and save checkpoint for downstream pipelines.

Designed to be used interactively in Jupyter/IPython environments.
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
        Raw EEG data with montage already set.
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

    # Print anatomical summary to console
    map_df = load_bel_channel_map()
    summary = get_anatomical_summary(all_bads, map_df)
    logger.info("Bad channel candidates (%d):\n%s", len(all_bads), summary)

    # Generate interactive 3D report
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
    eye_blink_thresh: float = DEFAULT_ICALABEL_THRESH,
) -> Tuple[mne.preprocessing.ICA, Dict]:
    """Filter data, fit ICA, and run ICLabel to assist manual component selection.

    This function does NOT exclude any components automatically (except
    suggesting eye blinks). You inspect the results and set ``ica.exclude``
    yourself before calling ``apply_pre_cleaning``.

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
    eye_blink_thresh : float
        Probability threshold above which eye blink components are
        *suggested* for exclusion (logged but NOT auto-excluded).

    Returns
    -------
    ica : mne.preprocessing.ICA
        Fitted ICA instance. Inspect with ``ica.plot_sources(raw_for_ica)``.
    labels_dict : dict
        ICLabel output with keys 'labels' and 'y_pred_proba'.
    """
    # Work on a filtered copy for ICA fitting
    raw_filt = raw.copy().load_data()
    raw_filt.filter(l_freq=highpass, h_freq=lowpass, picks="eeg", n_jobs=1, verbose=False)
    nyquist = raw_filt.info["sfreq"] / 2.0
    notch_freqs = [f for f in [notch_freq, notch_freq * 2.0] if f <= min(lowpass, nyquist)]
    if notch_freqs:
        raw_filt.notch_filter(freqs=notch_freqs, picks="eeg", method="fir", verbose=False)
    raw_filt._data = np.real(raw_filt._data).astype(np.float64)

    # CAR is required for ICLabel
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

    # Run ICLabel
    from mne_icalabel import label_components
    labels_dict = label_components(raw_filt, ica, method="iclabel")

    # Log suggestions but do NOT auto-exclude
    logger.info("ICLabel classifications:")
    suggested_exclude = []
    for i, (label, prob_vec) in enumerate(
        zip(labels_dict["labels"], labels_dict["y_pred_proba"])
    ):
        max_prob = np.max(prob_vec)
        marker = ""
        if label.lower().strip() == "eye blink" and max_prob > eye_blink_thresh:
            suggested_exclude.append(i)
            marker = " <-- SUGGESTED EXCLUDE"
        logger.info("  IC%02d: %-20s %.3f%s", i, label, max_prob, marker)

    if suggested_exclude:
        logger.info(
            "Suggested eye-blink exclusions: %s. "
            "Review with ica.plot_sources() and set ica.exclude manually.",
            suggested_exclude,
        )
    else:
        logger.info("No eye-blink components exceeded threshold %.2f.", eye_blink_thresh)

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

    # Mark bads and apply ICA
    cleaned.info["bads"] = list(bad_channels)
    logger.info("Applying ICA exclusion: %s", ica.exclude)
    cleaned = ica.apply(cleaned)

    # Interpolate bad channels
    if interpolate_bads and cleaned.info["bads"]:
        cleaned.interpolate_bads(reset_bads=True)
        logger.info("Interpolated %d bad channels.", len(bad_channels))

    # Re-apply average reference after interpolation
    cleaned.set_eeg_reference("average", projection=False, verbose=False)
    logger.info("Average reference applied.")

    # Save checkpoint
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.save(str(output_path), overwrite=overwrite, verbose=False)
    logger.info("Pre-cleaned data saved: %s", output_path)

    return cleaned
