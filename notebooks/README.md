# BEL 280 Sequential Batch Processing Guide

This guide explains how to configure and run `preprocess_bel_trials` for BEL EEG System One data. It covers required file structures, parameter configuration, and output interpretation.

## Required File Structure

The pipeline expects a BIDS-like derivative layout. Input files must be FIF format and match the specified glob pattern.

```text
data_dir/
├── condition_1/
│   ├── sub-01_condition_1_eeg_raw.fif      # Must end with _eeg_raw.fif by default
│   └── sub-02_condition_1_eeg_raw.fif
└── condition_2/
    └── sub-01/
        └── ses-01/
            └── sub-01_condition_2_eeg_raw.fif  # Recursive search finds nested files
```

-   **Input Format**: MNE-Python compatible FIF files (`.fif`).
-   **Naming Convention**: Default pattern is `*_eeg_raw.fif`. Files not matching this pattern are ignored.
-   **Channel Naming**: Raw files may use numeric EGI names (`'1'`, `'2'`) or BEL names (`'E1'`, `'E2'`). The pipeline automatically applies `DEFAULT_RENAME_MAP` to standardize to BEL nomenclature before processing.
-   **Montage**: Sensor positions are loaded from the bundled `ghw280_from_egig.gpsc` file. No external montage file is required unless using a custom sensor layout.

## Configuration Script

```python
# Sequential batch preprocessing for BEL 280-channel EEG epochs.
import logging
from pathlib import Path
from xeeg_kit import preprocess_bel_trials

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

DATA_DIR = Path("/path/to/derivatives/study_name/eeg")
OUTPUT_DIR = Path("/path/to/derivatives/study_name/eeg/clean")

MEEGKIT_PARAMS = {
    "highpass_filter": 1.0,          # Mandatory: ASR/ICA stability requirement
    "low_pass_filter": 100.0,        # Mandatory: ICLabel requires 1-100 Hz bandpass
    "notch_filter_freq": 60.0,       # Line noise removal; set None to disable
    "mad_threshold": 25.0,           # Noisy channel detection sensitivity (higher = stricter)
    "min_amplitude_uv": 0.1,         # Flat channel threshold in microvolts
    "asr_cutoff": 3.5,               # Artifact Subspace Reconstruction burst threshold
    "star_thresh": 3.5,              # Sparse Time-Artifact Removal threshold
    "sns_neighbors": 8,              # Sensor Noise Suppression neighbor count
    "drop_cz": True,                 # Remove Cz reference before cleaning
    "interpolate_bads": True,        # Spline-interpolate detected bad channels
    "verbose": True,                 # Enable stage-level logging
    "generate_report": True,         # Write interactive 3D HTML bad channel map
    "report_dir": OUTPUT_DIR,        # Explicit directory for QA reports
    "subject_id": "sub-01_meegkit",  # Stage-prefixed ID prevents filename collisions
}

ICALABEL_PARAMS = {
    "mad_threshold": 35.0,           # Stricter residual bad channel detection post-MEEGKit
    "min_amplitude_uv": 0.1,         # Flat channel check on re-referenced data
    "n_components": 0.95,            # Explained variance for ICA dimensionality selection
    "random_state": 42,              # Reproducible Picard ICA initialization
    "interpolate_bads": True,        # Interpolate residual bads found at ICLabel stage
    "verbose": True,                 # Log ICA fitting and component exclusion details
    "generate_report": True,         # Second HTML report for ICLabel-stage residuals
    "report_dir": OUTPUT_DIR,        # Same directory as MEEGKit reports
    "subject_id": "sub-01_icalabel", # Distinct suffix avoids overwriting MEEGKit report
}

run_xeegkit = preprocess_bel_trials(
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    meegkit_params=MEEGKIT_PARAMS,
    icalabel_params=ICALABEL_PARAMS,
    pattern="*_eeg_raw.fif",         # Glob pattern for input file discovery
    recursive=True,                  # Traverse subdirectories preserving structure
    overwrite=True,                  # Replace existing outputs without prompting
    verbose=True                     # Enable top-level file iteration logging
)

logging.info("EEG processing complete: %d files", len(saved_paths))
```

## Parameter Reference

### MEEGKit Stage

| Parameter            | Type    | Default       | Description                                                                 |
| -------------------- | ------- | ------------- | --------------------------------------------------------------------------- |
| `highpass_filter`    | float   | 1.0           | High-pass cutoff in Hz. Must be ≥1.0 for ASR/ICA stability.                |
| `low_pass_filter`    | float   | 100.0         | Low-pass cutoff in Hz. **Must be 100.0** for ICLabel compatibility.         |
| `notch_filter_freq`  | float   | 60.0          | Notch frequency in Hz. Automatically skipped if > `low_pass_filter`.        |
| `mad_threshold`      | float   | 15.0          | MAD-zscore threshold for noisy channel detection. Higher values are stricter. |
| `min_amplitude_uv`   | float   | 0.1           | Minimum peak-to-peak amplitude in µV. Channels below are flagged as flat.   |
| `asr_cutoff`         | float   | 3.5           | ASR burst rejection threshold in standard deviation units.                  |
| `star_thresh`        | float   | 3.5           | STAR artifact subspace reconstruction threshold.                            |
| `sns_neighbors`      | int     | 8             | Number of nearest neighbors for Sensor Noise Suppression.                   |
| `drop_cz`            | bool    | True          | Whether to remove Cz reference channel before processing.                   |
| `interpolate_bads`   | bool    | True          | Whether to spline-interpolate detected bad channels.                        |
| `generate_report`    | bool    | False         | Enable interactive 3D HTML bad channel visualization.                       |
| `report_dir`         | Path    | CWD           | Directory for HTML reports. Defaults to current working directory if omitted. |
| `subject_id`         | str     | "sub-unknown" | Identifier for report filename. Use stage prefix to avoid collisions.       |

### ICLabel Stage

| Parameter            | Type    | Default       | Description                                                                 |
| -------------------- | ------- | ------------- | --------------------------------------------------------------------------- |
| `mad_threshold`      | float   | 20.0          | Residual bad channel threshold on post-MEEGKit cleaned data.                |
| `min_amplitude_uv`   | float   | 0.1           | Flat channel detection on re-referenced data.                               |
| `n_components`       | float   | 0.95          | Explained variance threshold for ICA component selection.                   |
| `random_state`       | int     | 42            | Seed for reproducible Picard ICA initialization.                            |
| `interpolate_bads`   | bool    | True          | Interpolate residual bad channels detected at this stage.                   |
| `generate_report`    | bool    | False         | Enable second HTML report for ICLabel-stage bad channels.                   |
| `report_dir`         | Path    | CWD           | Directory for HTML reports.                                                 |
| `subject_id`         | str     | "sub-unknown" | Distinct suffix prevents overwriting MEEGKit report.                        |

### Pipeline Arguments

| Argument             | Type    | Default          | Description                                                    |
| -------------------- | ------- | ---------------- | -------------------------------------------------------------- |
| `data_dir`           | Path    | Required         | Root directory containing input FIF files.                     |
| `output_dir`         | Path    | Required         | Root directory for cleaned outputs and reports.                |
| `pattern`            | str     | `*_eeg_raw.fif`  | Glob pattern for input file discovery.                         |
| `recursive`          | bool    | True             | Search subdirectories recursively.                             |
| `overwrite`          | bool    | True             | Overwrite existing output files.                               |
| `preload`            | bool    | True             | Load entire FIF into memory. Required for ASR/ICA.             |
| `gpsc_path`          | Path    | Bundled          | Custom GPSC montage file. Uses bundled default if omitted.     |
| `rename_map`         | Dict    | BEL default      | Custom channel renaming map. Overrides default if provided.    |

## Output Structure

The pipeline preserves the relative directory structure of `data_dir` within `output_dir`.

```text
output_dir/
├── condition_1/
│   ├── sub-01_condition_1_eeg_raw_proc.fif          # Cleaned continuous data
│   ├── sub-02_condition_1_eeg_raw_proc.fif
│   ├── sub-01_meegkit_bad_channels_3d.html          # MEEGKit-stage bad channel report
│   └── sub-01_icalabel_bad_channels_3d.html         # ICLabel-stage bad channel report
└── condition_2/
    └── sub-01/
        └── ses-01/
            ├── sub-01_condition_2_eeg_raw_proc.fif
            ├── sub-01_meegkit_bad_channels_3d.html
            └── sub-01_icalabel_bad_channels_3d.html
```

-   **Cleaned Data**: FIF files with `_proc` suffix containing interpolated, artifact-cleaned continuous EEG.
-   **QA Reports**: Interactive Plotly HTML files showing 3D bad channel locations colored by anatomical region. Two reports per subject when both stages have `generate_report=True`.
-   **Return Value**: `Dict[str, Path]` mapping relative input paths to absolute output paths for programmatic downstream access.

## Critical Constraints

1.  **ICLabel Bandpass Requirement**: `low_pass_filter` must be exactly `100.0` in `MEEGKIT_PARAMS`. Values other than 100 Hz trigger a `RuntimeWarning` from `mne_icalabel` because the classifier was trained exclusively on 1–100 Hz data. Apply narrower filters (e.g., 50 Hz) only after preprocessing completes.
2.  **Report Filename Collisions**: When processing multiple subjects, `subject_id` must be unique per stage. Hardcoding a single identifier causes all subjects' reports to overwrite each other. Extract subject IDs dynamically from filenames in production batches.
3.  **Output Naming Warning**: The default `_proc` suffix triggers an MNE naming convention warning. This does not affect data integrity but may cause issues with BIDS validators. Rename outputs to `_eeg.fif` if strict compliance is required.
4.  **Dual Bad Channel Detection**: MEEGKit detects sensor-level hardware failures on pre-CAR filtered data. ICLabel detects residual biological/environmental artifacts on post-cleaning re-referenced data. Both sets are independently interpolated. The two reports provide complementary QA information and are not redundant.
