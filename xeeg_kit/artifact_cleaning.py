# xeeg_kit/artifact_cleaning.py

# Core artifact cleaning pipelines: MEEGKit and ICLabel-based ICA.
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import mne
import numpy as np
from .meegkit_vendor import asr, star, sns
from mne_icalabel import label_components
from .utils import detect_bad_channels, find_cleanest_segment
from .viz import get_anatomical_summary, plot_bad_channels_3d, load_bel_channel_map

logger = logging.getLogger(__name__)

DEFAULT_HIGHPASS = 1.0
DEFAULT_LOWPASS = 100.0
DEFAULT_NOTCH = 60.0
DEFAULT_MAD_THRESH = 15.0
DEFAULT_MIN_AMP_UV = 0.1
DEFAULT_ASR_CUTOFF = 3.0
DEFAULT_STAR_THRESH = 2.5
DEFAULT_SNS_NEIGHBORS = 8
DEFAULT_CALIB_START = 0.0
DEFAULT_CALIB_DUR = 30.0
DEFAULT_ICA_COMP = 0.99
DEFAULT_ICA_SEED = 99
DEFAULT_ICALABEL_THRESH = 0.75
MIN_HIGHPASS_FOR_ICA = 1.0

def _apply_eeg_filters(raw: mne.io.Raw, highpass: float, lowpass: Optional[float], notch_freq: Optional[float]) -> None:
    raw.filter(l_freq=highpass, h_freq=lowpass, picks='eeg', n_jobs=1, verbose=False)
    if notch_freq is not None:
        nyquist = raw.info['sfreq'] / 2.0
        max_f = min(lowpass or np.inf, nyquist)
        freqs = [f for f in [notch_freq, notch_freq * 2.0] if f <= max_f]
        if freqs:
            raw.notch_filter(freqs=freqs, picks='eeg', method='fir', filter_length='auto', n_jobs=1, verbose=False)
    raw._data = np.real(raw._data).astype(np.float64)

def _mark_bad_channels(raw: mne.io.Raw, mad_threshold: float, min_amplitude_uv: float) -> List[str]:
    bad_chs = detect_bad_channels(raw, mad_threshold=mad_threshold, min_amplitude_uv=min_amplitude_uv)
    raw.info['bads'] = bad_chs
    logger.info("Marked %d channels as bad.", len(bad_chs))
    return bad_chs

def _prepare_good_data(raw: mne.io.Raw) -> tuple[mne.io.Raw, List[int], np.ndarray]:
    raw.set_eeg_reference('average', projection=False, verbose=False)
    good_idx = mne.pick_types(raw.info, eeg=True, exclude='bads')

    if len(good_idx) == 0:
        raise ValueError("No good EEG channels found after bad channel detection.")
    good_names = [raw.ch_names[i] for i in good_idx]
    data_good = raw.get_data(picks=good_idx)
    info_good = mne.create_info(ch_names=good_names, sfreq=raw.info['sfreq'], ch_types='eeg')
    raw_good = mne.io.RawArray(data_good, info_good, verbose=False)
    if raw.get_montage() is not None:
        raw_good.set_montage(raw.get_montage(), on_missing='ignore')
    logger.info("CAR applied to good EEG channels.")
    return raw_good, good_idx, data_good

def _run_asr_star_sns(raw_good: mne.io.Raw, data_good: np.ndarray, asr_cutoff: float, star_thresh: float, sns_neighbors: int, calib_start: float, calib_dur: float) -> np.ndarray:
    calib_data, _ = find_cleanest_segment(raw_good, start_sec=calib_start, duration_sec=calib_dur)
    calib_data = np.real(calib_data).astype(np.float64)
    asr_model = asr.ASR(sfreq=raw_good.info['sfreq'], cutoff=asr_cutoff, estimator='oas')
    asr_model.fit(calib_data)
    cleaned = asr_model.transform(data_good)
    cleaned = np.real(cleaned).astype(np.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        denoised, _, _ = star(cleaned.T, thresh=star_thresh, verbose=False)
    cleaned = denoised.T
    try:
        denoised, _ = sns(cleaned.T, n_neighbors=sns_neighbors)
        cleaned = denoised.T
        logger.info("SNS applied successfully.")
    except Exception as e:
        if "SNS operator should be zero along diagonal" in str(e):
            logger.warning("SNS skipped: data too low-rank after ASR/STAR.")
        else:
            logger.warning("SNS failed: %s", str(e))
    assert cleaned.shape == data_good.shape, "ASR/STAR/SNS output shape mismatch."
    return cleaned

def _generate_bad_channel_report(raw: mne.io.Raw, bad_chs: List[str], report_dir: Path, subject_id: str) -> None:
    if not bad_chs:
        logger.info("No bad channels detected; skipping report generation.")
        return
    map_df = load_bel_channel_map()
    summary = get_anatomical_summary(bad_chs, map_df)
    logger.info("Anatomical Distribution:\n%s", summary)
    output_file = report_dir / f"{subject_id}_bad_channels_3d.html"
    plot_bad_channels_3d(raw, bad_chs, map_df=map_df, output_file=str(output_file))
    logger.info("3D bad channel report saved to: %s", output_file)

def execute_meegkit(
    raw: mne.io.Raw, highpass_filter: float = DEFAULT_HIGHPASS, low_pass_filter: Optional[float] = DEFAULT_LOWPASS,
    notch_filter_freq: Optional[float] = DEFAULT_NOTCH, mad_threshold: float = DEFAULT_MAD_THRESH,
    min_amplitude_uv: float = DEFAULT_MIN_AMP_UV, asr_cutoff: float = DEFAULT_ASR_CUTOFF,
    star_thresh: float = DEFAULT_STAR_THRESH, sns_neighbors: int = DEFAULT_SNS_NEIGHBORS,
    drop_cz: bool = True, interpolate_bads: bool = True, find_cleanest_segment_start: float = DEFAULT_CALIB_START,
    find_cleanest_segment_duration: float = DEFAULT_CALIB_DUR, generate_report: bool = False,
    report_dir: Optional[Union[str, Path]] = None, subject_id: str = "sub-unknown", verbose: bool = True
) -> mne.io.Raw:
    if verbose: logger.info("Starting MEEGKit cleaning pipeline...")
    raw = raw.copy().load_data()
    _apply_eeg_filters(raw, highpass_filter, low_pass_filter, notch_filter_freq)
    if drop_cz and 'Cz' in raw.ch_names:
        raw.drop_channels(['Cz'])
        logger.info("Dropped Cz reference channel.")
    bad_chs = _mark_bad_channels(raw, mad_threshold, min_amplitude_uv)
    raw_good, good_idx, data_good = _prepare_good_data(raw)
    cleaned_data = _run_asr_star_sns(raw_good, data_good, asr_cutoff, star_thresh, sns_neighbors, find_cleanest_segment_start, find_cleanest_segment_duration)
    raw._data[good_idx, :] = cleaned_data
    if interpolate_bads and bad_chs:
        raw.interpolate_bads(reset_bads=True)
        logger.info("Interpolated %d bad channels.", len(bad_chs))
    if generate_report:
        r_dir = Path(report_dir) if report_dir else Path.cwd()
        r_dir.mkdir(parents=True, exist_ok=True)
        _generate_bad_channel_report(raw, bad_chs, r_dir, subject_id)
    return raw

def execute_icalabel(
    raw: mne.io.Raw, icalabel_thresholds: Optional[Dict[str, float]] = None, mad_threshold: float = 20.0,
    min_amplitude_uv: float = DEFAULT_MIN_AMP_UV, n_components: Any = DEFAULT_ICA_COMP, random_state: int = DEFAULT_ICA_SEED,
    interpolate_bads: bool = True, generate_report: bool = False, report_dir: Optional[Union[str, Path]] = None,
    subject_id: str = "sub-unknown", verbose: bool = True
) -> mne.io.Raw:
    if verbose: logger.info("Starting ICA + ICLabel cleaning pipeline...")
    if icalabel_thresholds is None:
        icalabel_thresholds = {k: DEFAULT_ICALABEL_THRESH for k in ['eye blink', 'heart beat', 'muscle artifact', 'line noise', 'channel noise']}
    raw = raw.copy().load_data()
    raw._data = np.real(raw._data).astype(np.float64)
    bads = _mark_bad_channels(raw, mad_threshold, min_amplitude_uv)
    raw.set_eeg_reference('average', projection=False, verbose=False)
    logger.info("Re-applied average reference for ICA.")
    assert raw.info['highpass'] is not None, "Data must have a defined highpass filter."
    if raw.info['highpass'] > MIN_HIGHPASS_FOR_ICA:
        raise ValueError(f"Data must be high-pass filtered at <={MIN_HIGHPASS_FOR_ICA} Hz before ICA.")
    try:
        logger.info("Fitting ICA...")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            ica = mne.preprocessing.ICA(n_components=n_components, method='picard', fit_params=dict(ortho=False, extended=True), random_state=random_state, max_iter='auto')
            ica.fit(raw)
        logger.info("ICA fitted with %d components.", ica.n_components_)
        logger.info("Running ICLabel...")
        raw_eeg = raw.copy().pick("eeg")
        labels_dict = label_components(raw_eeg, ica, method="iclabel")
        excluded = [i for i, (label, prob_vec) in enumerate(zip(labels_dict["labels"], labels_dict["y_pred_proba"])) if label.lower().strip() in icalabel_thresholds and np.max(prob_vec) > icalabel_thresholds[label.lower().strip()]]
        ica.exclude = sorted(set(excluded))
        logger.info("Excluding ICA components: %s", ica.exclude)
        if ica.exclude:
            for i in ica.exclude:
                logger.info("  C%02d: %-18s (%.2f)", i, labels_dict["labels"][i], np.max(labels_dict["y_pred_proba"][i]))
        cleaned = ica.apply(raw)
    except Exception as e:
        logger.warning("ICA failed (%s). Skipping ICA.", str(e)[:120])
        cleaned = raw.copy()
    if interpolate_bads and bads:
        cleaned.info['bads'] = bads
        cleaned.interpolate_bads(reset_bads=True)
        logger.info("Interpolated %d originally bad channels.", len(bads))
        cleaned.set_eeg_reference('average', projection=False, verbose=False)
        logger.info("Re-applied average reference after interpolation.")
    if generate_report:
        r_dir = Path(report_dir) if report_dir else Path.cwd()
        r_dir.mkdir(parents=True, exist_ok=True)
        _generate_bad_channel_report(cleaned, bads, r_dir, subject_id)
    return cleaned








