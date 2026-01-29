# xeeg_kit/artifact_cleaning.py

# Core artifact cleaning pipelines: MEEGKit and ICLabel-based ICA.

import warnings
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from meegkit import asr, star, sns
from mne_icalabel import label_components
from .utils import log, detect_bad_channels, find_cleanest_segment

def execute_meegkit(
    raw: 'mne.io.Raw',
    low_pass_filter: Optional[float] = 100.0,
    notch_filter_freq: float = 60.0,
    mad_threshold: float = 10.0,
    min_amplitude_uv: float = 0.1,
    asr_cutoff: float = 3.0,
    star_thresh: float = 2.5,
    sns_neighbors: int = 8,
    drop_cz: bool = True,
    interpolate_bads: bool = True,
    find_cleanest_segment_start: float = 0.0,
    find_cleanest_segment_duration: float = 30.0,
    verbose: bool = True
) -> 'mne.io.Raw':
    """
    Execute the MEEGKit cleaning pipeline (ASR + STAR + SNS).
    
    Assumes `raw` is already standardized (channel names, montage set).
    """
    import mne
    if verbose:
        log("Starting MEEGKit cleaning pipeline...")

    raw = raw.copy().load_data()
    raw.filter(l_freq=1.0, h_freq=low_pass_filter, picks='eeg', n_jobs=1, verbose=False)
    raw.notch_filter(freqs=notch_filter_freq, picks='eeg', method='spectrum_fit',
                     filter_length='auto', mt_bandwidth=1.0, p_value=0.05, n_jobs=1, verbose=False)

    raw._data = np.real(raw._data).astype(np.float64)

    if drop_cz and 'Cz' in raw.ch_names:
        raw.drop_channels(['Cz'])
        if verbose:
            log("Dropped Cz (reference channel).")

    bad_chs = detect_bad_channels(raw, mad_threshold=mad_threshold, min_amplitude_uv=min_amplitude_uv)
    raw.info['bads'] = bad_chs
    if verbose:
        log(f"Marked {len(bad_chs)} channels as bad.")

    good_idx = mne.pick_types(raw.info, eeg=True, exclude='bads')
    if len(good_idx) == 0:
        raise ValueError("No good EEG channels found after bad channel detection.")
    
    good_ch_names = [raw.ch_names[i] for i in good_idx]
    data_good = raw.get_data(picks=good_idx)

    info_good = mne.create_info(ch_names=good_ch_names, sfreq=raw.info['sfreq'], ch_types='eeg')
    raw_good = mne.io.RawArray(data_good, info_good, verbose=False)
    if raw.get_montage() is not None:
        raw_good.set_montage(raw.get_montage(), on_missing='ignore')
    raw_good.set_eeg_reference('average', verbose=False)
    if verbose:
        log("CAR applied on good E-channels only.")

    calib_data, _ = find_cleanest_segment(
        raw_good, 
        start_sec=find_cleanest_segment_start, 
        duration_sec=find_cleanest_segment_duration
    )
    calib_data = np.real(calib_data).astype(np.float64)
    asr_model = asr.ASR(sfreq=raw_good.info['sfreq'], cutoff=asr_cutoff, estimator='oas')
    asr_model.fit(calib_data)
    cleaned_good = asr_model.transform(data_good)
    cleaned_good = np.real(cleaned_good).astype(np.float64)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        denoised_t_ch, _, _ = star.star(cleaned_good.T, thresh=star_thresh, verbose=False)
    cleaned_good = denoised_t_ch.T

    try:
        denoised_t_ch, _ = sns.sns(cleaned_good.T, n_neighbors=sns_neighbors)
        cleaned_good = denoised_t_ch.T
        if verbose:
            log("SNS applied successfully.")
    except Exception as e:
        if "SNS operator should be zero along diagonal" in str(e):
            if verbose: 
                log("⚠️ SNS skipped: data too low-rank after ASR/STAR.")
        else:
            if verbose:
                log(f"⚠️ SNS failed: {e}")

    raw._data[good_idx, :] = cleaned_good

    if interpolate_bads and bad_chs:
        raw.interpolate_bads(reset_bads=True)
        if verbose:
            log(f"Interpolated {len(bad_chs)} bad channels.")

    return raw


def execute_icalabel(
    raw: 'mne.io.Raw',
    icalabel_thresholds: Optional[Dict[str, float]] = None,
    mad_threshold: float = 15.0,
    min_amplitude_uv: float = 0.1,
    n_components: Any = 0.99,
    random_state: int = 99,
    interpolate_bads: bool = True,
    verbose: bool = True
) -> 'mne.io.Raw':
    """
    Execute ICA + ICLabel automatic component rejection.
    """
    import mne
    import numpy as np
    from warnings import catch_warnings, filterwarnings

    if icalabel_thresholds is None:
        icalabel_thresholds = {
            'eye blink': 0.70,
            'heart beat': 0.70,
            'muscle artifact': 0.70,
            'line noise': 0.70,
            'channel noise': 0.70
        }

    if verbose:
        log("Starting ICA + ICLabel cleaning pipeline...")

    raw = raw.copy().load_data()
    raw._data = np.real(raw._data).astype(np.float64)

    bads = detect_bad_channels(raw, mad_threshold=mad_threshold, min_amplitude_uv=min_amplitude_uv)
    raw.info['bads'] = bads
    if verbose:
        log(f"Detected {len(bads)} bad channels.")

    raw.set_eeg_reference('average', verbose=False)
    if verbose:
        log("Re-applied average reference.")

    if raw.info['highpass'] > 1.0:
        raise ValueError("Data must be high-pass filtered at ≤1 Hz before ICA.")

    try:
        if verbose:
            log("Fitting ICA...")
        with catch_warnings():
            filterwarnings('ignore')
            ica = mne.preprocessing.ICA(
                n_components=n_components,
                method='picard',
                fit_params=dict(ortho=False, extended=True),
                random_state=random_state,
                max_iter='auto'
            )
            ica.fit(raw)
        if verbose:
            log(f"ICA fitted with {ica.n_components_} components.")

        if verbose:
            log("Running ICLabel...")
        raw_eeg = raw.copy().pick("eeg")
        labels_dict = label_components(raw_eeg, ica, method="iclabel")

        # Verbose logging
        excluded = []
        for i, (label, prob_vec) in enumerate(zip(labels_dict["labels"], labels_dict["y_pred_proba"])):
            lbl = label.lower().strip()
            if lbl in icalabel_thresholds and np.max(prob_vec) > icalabel_thresholds[lbl]:
                excluded.append(i)

        ica.exclude = sorted(set(excluded))

        if verbose:
            log(f"Excluding ICA components: {ica.exclude}")
            if ica.exclude:
                log("ICLabel component classifications:")
                for i in ica.exclude:
                    label = labels_dict["labels"][i]
                    max_prob = np.max(labels_dict["y_pred_proba"][i])
                    log(f"  C{i:02d}: {label:<18} ({max_prob:.2f})")
        # END: Verbose logging

        cleaned = ica.apply(raw)

    except Exception as e:
        if verbose:
            log(f"⚠️ ICA failed ({str(e)[:120]}). Skipping ICA.")
        cleaned = raw.copy()

    if interpolate_bads and bads:
        cleaned.info['bads'] = bads
        cleaned.interpolate_bads(reset_bads=True)
        if verbose:
            log(f"Interpolated {len(bads)} originally bad channels.")

    return cleaned


