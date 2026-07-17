# BEL 280 Sequential Batch Processing Guide

This guide explains how to configure and run `preprocess_bel_trials` for BEL EEG System One data. It covers required file structures, parameter configuration, and output interpretation.

## Required File Structure

The pipeline expects input FIF files at the root level of `data_dir`. Files are discovered non-recursively by default.

```text
data_dir/
├── DP02_gain_eeg_raw.fif      # Must match the glob pattern (default: *_eeg_raw.fif)
├── DP02_loss_eeg_raw.fif
├── DP03_gain_eeg_raw.fif
└── DP03_loss_eeg_raw.fif
```

-   **Input Format**: MNE-Python compatible FIF files (`.fif`).
-   **Naming Convention**: Default pattern is `*_eeg_raw.fif`. Files not matching this pattern are ignored.
-   **Flat Structure Only**: The pipeline uses non-recursive glob discovery. All input files must reside directly in `data_dir`. Subdirectories are not traversed.
-   **Filename Uniqueness**: Flat output means files with identical stems will overwrite each other when `overwrite=True`. Ensure unique filenames within `data_dir`.
-   **Channel Naming**: Raw files may use numeric EGI names (`'1'`, `'2'`) or BEL names (`'E1'`, `'E2'`). The pipeline automatically applies `DEFAULT_RENAME_MAP` to standardize to BEL nomenclature before processing.
-   **Montage**: Sensor positions are loaded from the bundled `ghw280_from_egig.gpsc` file. No external montage file is required unless using a custom sensor layout.

## Configuration Script

```python
# Sequential batch preprocessing for BEL 280-channel EEG data.
import logging
from pathlib import Path
from xeeg_kit import preprocess_bel_trials

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

DATA_DIR = Path("/path/to/data")
OUTPUT_DIR = Path("/path/to/output")

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
    # Note: Do NOT set report_dir or subject_id here.
    # The pipeline manages these automatically to prevent collisions.
}

ICALABEL_PARAMS = {
    "mad_threshold": 35.0,           # Stricter residual bad channel detection post-MEEGKit
    "min_amplitude_uv": 0.1,         # Flat channel check on re-referenced data
    "n_components": 0.95,            # Explained variance for ICA dimensionality selection
    "random_state": 42,              # Reproducible Picard ICA initialization
    "interpolate_bads": True,        # Interpolate residual bads found at ICLabel stage
    "verbose": True,                 # Log ICA fitting and component exclusion details
    "generate_report": True,         # Second HTML report for ICLabel-stage residuals
    # Note: Do NOT set report_dir or subject_id here.
    # The pipeline manages these automatically to prevent collisions.
}

saved_paths = preprocess_bel_trials(
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    meegkit_params=MEEGKIT_PARAMS,
    icalabel_params=ICALABEL_PARAMS,
    pattern="*_eeg_raw.fif",         # Glob pattern for input file discovery
    overwrite=True,                  # Replace existing outputs without prompting
    verbose=True                     # Enable top-level file iteration logging
)

logging.info("EEG processing complete: %d files", len(saved_paths))
```

## Important Changes from Previous Versions

### Pipeline-Managed Parameters
The pipeline now **automatically manages** `report_dir` and `subject_id` for both processing stages. Do **not** include these parameters in `MEEGKIT_PARAMS` or `ICALABEL_PARAMS`. The pipeline:
- Extracts the subject ID from each filename stem
- Creates a dedicated `reports/` subdirectory within `output_dir`
- Generates unique report filenames per subject and stage

Any `report_dir` or `subject_id` values you provide will be silently overridden.

### Non-Recursive File Discovery
The `recursive` parameter has been removed. The pipeline now uses flat, non-recursive glob discovery only. All input FIF files must reside directly in `data_dir`.

### Flat Output Structure
All cleaned files are saved directly in `output_dir` with no subdirectory nesting. This eliminates the risks associated with recursive directory traversal while simplifying output management.

## Parameter Reference

### MEEGKit Stage

| Parameter            | Type    | Default       | Description                                                                 |
| -------------------- | ------- | ------------- | --------------------------------------------------------------------------- |
| `highpass_filter`    | float   | 1.0           | High-pass cutoff in Hz. Must be ≥1.0 for ASR/ICA stability.                |
| `low_pass_filter`    | float   | 100.0         | Low-pass cutoff in Hz. **Must be 100.0** for ICLabel compatibility.         |
| `notch_filter_freq`  | float   | 60.0          | Notch frequency in Hz. Automatically skipped if > `low_pass_filter`.        |
| `mad_threshold`      | float   | 25.0          | MAD-zscore threshold for noisy channel detection. Higher values are stricter. |
| `min_amplitude_uv`   | float   | 0.1           | Minimum peak-to-peak amplitude in µV. Channels below are flagged as flat.   |
| `asr_cutoff`         | float   | 3.5           | ASR burst rejection threshold in standard deviation units.                  |
| `star_thresh`        | float   | 3.5           | STAR artifact subspace reconstruction threshold.                            |
| `sns_neighbors`      | int     | 8             | Number of nearest neighbors for Sensor Noise Suppression.                   |
| `drop_cz`            | bool    | True          | Whether to remove Cz reference channel before processing.                   |
| `interpolate_bads`   | bool    | True          | Whether to spline-interpolate detected bad channels.                        |
| `generate_report`    | bool    | False         | Enable interactive 3D HTML bad channel visualization.                       |
| `report_dir`         | Path    | Auto-managed  | **Do not set.** Pipeline manages this automatically.                        |
| `subject_id`         | str     | Auto-managed  | **Do not set.** Pipeline extracts from filename and appends stage suffix.   |

### ICLabel Stage

| Parameter            | Type    | Default       | Description                                                                 |
| -------------------- | ------- | ------------- | --------------------------------------------------------------------------- |
| `mad_threshold`      | float   | 20.0          | Residual bad channel threshold on post-MEEGKit cleaned data.                |
| `min_amplitude_uv`   | float   | 0.1           | Flat channel detection on re-referenced data.                               |
| `n_components`       | float   | 0.95          | Explained variance threshold for ICA component selection.                   |
| `random_state`       | int     | 42            | Seed for reproducible Picard ICA initialization.                            |
| `interpolate_bads`   | bool    | True          | Interpolate residual bad channels detected at this stage.                   |
| `generate_report`    | bool    | False         | Enable second HTML report for ICLabel-stage bad channels.                   |
| `report_dir`         | Path    | Auto-managed  | **Do not set.** Pipeline manages this automatically.                        |
| `subject_id`         | str     | Auto-managed  | **Do not set.** Pipeline extracts from filename and appends stage suffix.   |

### Pipeline Arguments

| Argument             | Type    | Default          | Description                                                    |
| -------------------- | ------- | ---------------- | -------------------------------------------------------------- |
| `data_dir`           | Path    | Required         | Root directory containing input FIF files at top level.        |
| `output_dir`         | Path    | Required         | Flat destination directory for cleaned outputs and reports.    |
| `pattern`            | str     | `*_eeg_raw.fif`  | Glob pattern for input file discovery (non-recursive).         |
| `overwrite`          | bool    | True             | Overwrite existing output files.                               |
| `preload`            | bool    | True             | Load entire FIF into memory. Required for ASR/ICA.             |
| `gpsc_path`          | Path    | Bundled          | Custom GPSC montage file. Uses bundled default if omitted.     |
| `rename_map`         | Dict    | BEL default      | Custom channel renaming map. Overrides default if provided.    |

## Output Structure

The pipeline creates a flat output directory with a dedicated `reports/` subdirectory.

```text
output_dir/
├── DP02_gain_eeg_raw_eeg.fif          # Cleaned continuous data
├── DP02_loss_eeg_raw_eeg.fif
├── DP03_gain_eeg_raw_eeg.fif
├── DP03_loss_eeg_raw_eeg.fif
└── reports/
    ├── DP02_meegkit_bad_channels_3d.html          # MEEGKit-stage bad channel report
    ├── DP02_icalabel_bad_channels_3d.html         # ICLabel-stage bad channel report
    ├── DP03_meegkit_bad_channels_3d.html
    └── DP03_icalabel_bad_channels_3d.html
```

-   **Cleaned Data**: FIF files with `_eeg` suffix containing interpolated, artifact-cleaned continuous EEG.
-   **QA Reports**: Interactive Plotly HTML files showing 3D bad channel locations colored by anatomical region. All reports are consolidated in the `reports/` subdirectory. Two reports per subject when both stages have `generate_report=True`.
-   **Missing Reports**: When no bad channels are detected in a stage, the corresponding HTML report is not generated. Fewer report files than expected is normal and indicates clean data.
-   **Return Value**: `Dict[str, Path]` mapping input filenames to absolute output paths for programmatic downstream access.

## Critical Constraints

1.  **ICLabel Bandpass Requirement**: `low_pass_filter` must be exactly `100.0` in `MEEGKIT_PARAMS`. Values other than 100 Hz trigger a `RuntimeWarning` from `mne_icalabel` because the classifier was trained exclusively on 1–100 Hz data. Apply narrower filters (e.g., 50 Hz) only after preprocessing completes.

2.  **Pipeline-Managed Parameters**: Do not set `report_dir` or `subject_id` in `MEEGKIT_PARAMS` or `ICALABEL_PARAMS`. The pipeline automatically manages these to ensure unique filenames and consistent report organization. Any values you provide will be silently overridden by the local merge operation.

3.  **Flat Structure Requirements**: All input files must reside directly in `data_dir` (no subdirectories). The pipeline uses non-recursive glob discovery. Filenames must be unique to prevent output collisions.

4.  **Output Naming**: The default output suffix is `_eeg` (producing files like `sub-01_task_eeg_raw_eeg.fif`). This prevents BIDS validation issues while maintaining clear provenance of the processing pipeline.

5.  **Dual Bad Channel Detection**: MEEGKit detects sensor-level hardware failures on pre-CAR filtered data. ICLabel detects residual biological/environmental artifacts on post-cleaning re-referenced data. Both sets are independently interpolated. The two reports provide complementary QA information and are not redundant.


