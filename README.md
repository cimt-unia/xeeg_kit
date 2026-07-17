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
from pathlib import Path
from xeeg_kit import preprocess_bel_trials

meegkit_params = {
    "highpass_filter": 1.0,
    "low_pass_filter": 50.0,
    "notch_filter_freq": 60.0,
    "mad_threshold": 15.0,
    "asr_cutoff": 3.5,
    "star_thresh": 3.5,
    "sns_neighbors": 8,
    "drop_cz": True,
    "interpolate_bads": True,
    "verbose": True,
}

icalabel_params = {
    "mad_threshold": 20.0,
    "n_components": 0.95,
    "random_state": 42,
    "interpolate_bads": True,
    "verbose": True,
}

saved_paths = preprocess_bel_trials(
    data_dir=Path("/data/trials/dp02"),
    output_dir=Path("/data/trials/dp02/clean"),
    meegkit_params=meegkit_params,
    icalabel_params=icalabel_params,
    pattern="*_eeg_raw.fif",
    recursive=True,
    overwrite=True,
    verbose=True,
)
```

### C. BEL 280 Parallel Processing

```python
from xeeg_kit import process_subjects_parallel

input_files = ["sub01.mff", "sub02.mff", "sub03.mff"]
output_dir = "cleaned_data/"

results = process_subjects_parallel(
    input_files=input_files,
    output_dir=output_dir,
    meegkit_params=meegkit_params,
    icalabel_params=icalabel_params,
    n_jobs=-1,
    verbose=True
)
```

<br>

## Important Notes

-   **BEL-Specific Modules**: `preprocess_bel_trials` and `process_subjects_parallel` are designed **only for BEL 280-channel data**. For other systems, use the generic `execute_meegkit`/`execute_icalabel` workflow.
-   **Filtering**: `execute_meegkit` always applies a **1.0 Hz high-pass filter** (required for ASR/ICA stability). The low-pass cutoff is configurable via `low_pass_filter`.
-   **Notch Filtering**: When `low_pass_filter < notch_filter_freq`, the notch is automatically skipped. Set `notch_filter_freq=None` to disable explicitly.
-   **Bundled Resources**: The default GPSC file is included in the package. Access via `get_default_gpsc_path()` without external downloads. Override by passing `gpsc_path=` explicitly.
-   **Output Naming**: Processed files use the `_proc` suffix by default (BIDS-aligned).
-   
-   **ICLabel Filtering Requirement**: `execute_icalabel` requires input data bandpass filtered between **1–100 Hz**. Set `low_pass_filter=100.0` in `meegkit_params` to satisfy this constraint and avoid false-positive warnings. Apply narrower lowpass filters (e.g., 50 Hz) only *after* preprocessing completes to preserve ICA component separability.
-   **Bad Channel Reports**: Enable interactive 3D visualization by adding `"generate_report": True`, `"report_dir": <path>`, and `"subject_id": "<id>"` to both `meegkit_params` and `icalabel_params`. Two separate HTML reports are generated per subject because each stage detects distinct artifact types:
    -   `{subject_id}_meegkit_bad_channels_3d.html`: Sensor-level hardware/contact failures detected on filtered pre-CAR data via amplitude flatness and MAD-zscore thresholds.
    -   `{subject_id}_icalabel_bad_channels_3d.html`: Residual noisy sensors detected on post-MEEGKit-cleaned data that corrupt ICA decomposition.
    Both bad channel sets are independently interpolated. Use stage-prefixed `subject_id` values (e.g., `"dp02_meegkit"`, `"dp02_icalabel"`) to prevent filename collisions.

<br>

## Project Structure

```
xeeg_kit/
├── __init__.py           # Public API exports
├── _config.py            # BLAS thread limits (critical for parallelism)
├── artifact_cleaning.py  # Core cleaning pipelines (format-agnostic)
├── bel_280.py            # BEL geometry, montage, and standardization
├── bel_pipeline.py       # BEL sequential batch processing
├── parallel.py           # BEL parallel orchestration
├── utils.py              # Bad channel detection, resource loading, calibration
├── viz.py                # Anatomical mapping and 3D visualization
├── analysis.py           # Channel selection and comparison plotting
└── data/                 # Bundled GPSC and channel map CSV
```

<br>

## License

MIT License — free for academic and commercial use.

<br>

## Acknowledgements

-   [MNE-Python](https://mne.tools)
-   [MEEGKit](https://nbara.github.io/python-meegkit/)
-   [MNE-ICALabel](https://mne.tools/mne-icalabel)
