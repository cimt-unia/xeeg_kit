# Multi-Stage Artifact Removal

**Core Philosophy: “Repair, Not Reject”**

<br>

We present a comprehensive framework for processing continuous high-density EEG data (280 channels) that combines state-of-the-art artifact removal techniques. The pipeline integrates MEEGKit algorithms (ASR, STAR, SNS) with ICA deep learning component classification (ICLabel).

Our approach addresses the unique challenges of long-duration recordings while maintaining spatial precision through enhanced coregistration and coordinate normalization.

A preprocessing toolkit for high-density EEG data, optimized for the **BEL EEG System One (280-channel Geodesic HD-EEG)** but also supporting general-purpose artifact cleaning for any EEG system.

Built on [MNE-Python](https://mne.tools), `xeeg_kit` integrates:
-   **MEEGKit** (ASR + STAR + SNS) for robust denoising
-   **ICLabel** for automatic ICA component rejection
-   BEL-specific utilities for channel renaming and montage handling
-   Parallel processing for batch analysis of BEL datasets

<br>

## Installation

Install from GitHub (requires Python ≥ 3.9):

```bash
pip install git+https://github.com/cimt-unia/xeeg_kit.git
```

```python
!python -m pip install --user git+https://github.com/cimt-unia/xeeg_kit.git
```

> **Dependencies**: `mne`, `meegkit`, `mne-icalabel`, `numpy`, `scipy`, `joblib`, `plotly`, `pandas`, `matplotlib`

<br>

## Core Features

### 1. Generic Cleaning Pipeline (Any EEG Format)
Use `execute_meegkit` and `execute_icalabel` with **any MNE-supported format** (`.fif`, `.mff`, `.set`, etc.).

### 2. BEL 280-Channel Support
-   Parse `.gpsc` sensor position files
-   Standardize channel names (e.g., EGI `'1'` → BEL `'E1'`)
-   Apply correct montage with fiducials (`FidNz`, `FidT9`, `FidT10`)
-   Bundled GPSC file accessible via `get_default_gpsc_path()`

### 3. Sequential & Parallel Batch Processing
-   `preprocess_bel_trials`: Recursive discovery with structure-preserving output for single-machine workflows
-   `process_subjects_parallel`: Multi-core orchestration with per-subject logging for large cohorts

<br>

## Usage Examples

### A. General Preprocessing (Any EEG System)

```python
import mne
from xeeg_kit import execute_meegkit, execute_icalabel, get_default_gpsc_path
from xeeg_kit.bel_280 import parse_gpsc, create_montage_from_gpsc

raw = mne.io.read_raw_egi("subject.mff", preload=True)

rename_map = {str(i): f'E{i}' for i in range(1, 281)}
rename_map['REF CZ'] = 'Cz'
raw.rename_channels(rename_map)

channels = parse_gpsc(get_default_gpsc_path())
montage = create_montage_from_gpsc(channels)
raw.set_montage(montage, on_missing="warn")

clean_1 = execute_meegkit(
    raw,
    low_pass_filter=100.0,
    notch_filter_freq=60.0,
    asr_cutoff=3.5,
    star_thresh=3.5,
    verbose=True
)

clean_2 = execute_icalabel(
    clean_1,
    n_components=0.95,
    random_state=42,
    verbose=True
)

clean_2.save("final_cleaned.fif", overwrite=True)
```

### B. BEL 280 Sequential Batch Processing

```python
# Sequential batch preprocessing for BEL 280-channel EEG data.
import logging
from pathlib import Path
from xeeg_kit import preprocess_bel_trials

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

DATA_DIR = Path("/mnt/movement/users/jaizor/xtra/derivatives/ocd/trials/dp02/eeg_raw")
OUTPUT_DIR = Path("/mnt/movement/users/jaizor/xtra/derivatives/ocd/trials/dp02/eeg_clean")

MEEGKIT_PARAMS = {
    "highpass_filter": 1.0,          # Mandatory: ASR/ICA stability requirement
    "low_pass_filter": 100.0,        # Mandatory: ICLabel requires 1-100 Hz bandpass
    "notch_filter_freq": 60.0,       # Line noise removal
    "mad_threshold": 25.0,           # Noisy channel detection sensitivity
    "min_amplitude_uv": 0.1,         # Flat channel threshold in microvolts
    "asr_cutoff": 3.5,               # ASR burst rejection threshold
    "star_thresh": 3.0,              # STAR artifact removal threshold
    "sns_neighbors": 8,              # Sensor Noise Suppression neighbor count
    "drop_cz": True,                 # Remove Cz reference before cleaning
    "interpolate_bads": True,        # Spline-interpolate detected bad channels
    "verbose": True,                 # Enable stage-level logging
    "generate_report": True,         # Write interactive 3D HTML bad channel map
    # NOTE: Do NOT set report_dir or subject_id - pipeline manages these automatically
}

ICALABEL_PARAMS = {
    "mad_threshold": 50.0,           # Stricter residual bad channel detection
    "min_amplitude_uv": 0.1,         # Flat channel check on re-referenced data
    "n_components": 0.95,            # Explained variance for ICA dimensionality
    "random_state": 42,              # Reproducible Picard ICA initialization
    "interpolate_bads": True,        # Interpolate residual bads at ICLabel stage
    "verbose": True,                 # Log ICA fitting and component exclusion
    "generate_report": True,         # Second HTML report for ICLabel residuals
    # NOTE: Do NOT set report_dir or subject_id - pipeline manages these automatically
}

saved_paths = preprocess_bel_trials(
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    meegkit_params=MEEGKIT_PARAMS,
    icalabel_params=ICALABEL_PARAMS,
    pattern="*_eeg_raw.fif",         # Glob pattern for input file discovery
    overwrite=True,                  # Replace existing outputs without prompting
    verbose=True,                    # Enable top-level file iteration logging
)

logging.info("EEG processing complete: %d files", len(saved_paths))
```


<br>

## Important Notes

-   **BEL-Specific Modules**: `preprocess_bel_trials` and `process_subjects_parallel` are designed **only for BEL 280-channel data**. For other systems, use the generic `execute_meegkit`/`execute_icalabel` workflow.
-   **Filtering**: `execute_meegkit` always applies a **1.0 Hz high-pass filter** (required for ASR/ICA stability). The low-pass cutoff is configurable via `low_pass_filter`.
-   **Notch Filtering**: When `low_pass_filter < notch_filter_freq`, the notch is automatically skipped. Set `notch_filter_freq=None` to disable explicitly.
-   **Bundled Resources**: The default GPSC file is included in the package. Access via `get_default_gpsc_path()` without external downloads. Override by passing `gpsc_path=` explicitly.
-   **Output Naming**: Processed files use the `_proc` suffix by default (BIDS-aligned).
-   **ICLabel Filtering Requirement**: `execute_icalabel` requires input data bandpass filtered between **1–100 Hz**. Set `low_pass_filter=100.0` in `meegkit_params` to satisfy this constraint and avoid false-positive warnings. Apply narrower lowpass filters (e.g., 50 Hz) only *after* preprocessing completes to preserve ICA component separability.
-   **Bad Channel Reports**: Enable interactive 3D visualization by adding `"generate_report": True`, `"report_dir": <path>`, and `"subject_id": "<id>"` to both `meegkit_params` and `icalabel_params`. Two separate HTML reports are generated per subject because each stage detects distinct artifact types:
    -   `{subject_id}_meegkit_bad_channels_3d.html`: Sensor-level hardware/contact failures detected on filtered pre-CAR data via amplitude flatness and MAD-zscore thresholds.
    -   `{subject_id}_icalabel_bad_channels_3d.html`: Residual noisy sensors detected on post-MEEGKit-cleaned data that corrupt ICA decomposition.
    Both bad channel sets are independently interpolated. Use stage-prefixed `subject_id` values (e.g., `"dp02_meegkit"`, `"dp02_icalabel"`) to prevent filename collisions.

<br>



## License

MIT License — free for academic and commercial use.

<br>

## Acknowledgements

-   [MNE-Python](https://mne.tools)
-   [MEEGKit](https://nbara.github.io/python-meegkit/)
-   [MNE-ICALabel](https://mne.tools/mne-icalabel)
