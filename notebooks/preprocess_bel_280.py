"""
BEL 280 EEG Preprocessing: Individual Subject Tutorial Usage
=================================================
This script demonstrates the minimal setup required to run the BEL 280
sequential batch processing pipeline using xeeg_kit.

"""

# ------------------------------------------------------------------------------
# 1. IMPORTS & LOGGING
# ------------------------------------------------------------------------------
import logging
from pathlib import Path
from xeeg_kit import preprocess_bel_trials

# Configure logging to see real-time progress in HH:MM:SS format.
# This is essential for monitoring long batch jobs.
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S"
)

# ------------------------------------------------------------------------------
# 2. DIRECTORY CONFIGURATION
# ------------------------------------------------------------------------------
# Define input and output paths. The pipeline expects a flat input directory
# with all FIF files at the root level (non-recursive discovery).
# Input files must be .fif format and match the glob pattern defined below.
DATA_DIR = Path("/path/to/data")
OUTPUT_DIR = Path("/path/to/output")

# ------------------------------------------------------------------------------
# 3. MEEGKIT STAGE PARAMETERS (Initial Cleaning)
# ------------------------------------------------------------------------------
# This stage handles filtering, bad channel detection, and artifact subspace
# reconstruction (ASR). These settings are optimized for BEL 280-channel data.
MEEGKIT_PARAMS = {
    # FILTERING (Critical for ICLabel compatibility)
    "highpass_filter": 1.0,      # ≥1.0 Hz required for ASR/ICA stability
    "low_pass_filter": 100.0,    # MUST be 100.0 Hz; ICLabel was trained on 1-100 Hz
    "notch_filter_freq": 60.0,   # Removes line noise (set None to disable)

    # BAD CHANNEL DETECTION
    "mad_threshold": 25.0,       # Noisy channel sensitivity (higher = stricter)
    "min_amplitude_uv": 0.1,     # Flat channel threshold in µV

    # ARTIFACT REMOVAL
    "asr_cutoff": 3.5,           # ASR burst rejection threshold (SD units)
    "star_thresh": 3.5,          # STAR artifact removal threshold
    "sns_neighbors": 8,          # Sensor Noise Suppression neighbor count

    # PROCESSING OPTIONS
    "drop_cz": True,             # Remove Cz reference before cleaning
    "interpolate_bads": True,    # Spline-interpolate detected bad channels

    # REPORTING
    "verbose": True,             # Enable stage-level logging
    "generate_report": True,     # Create interactive 3D HTML bad channel map
    # NOTE: Do NOT set report_dir or subject_id here.
    # The pipeline manages these automatically to prevent collisions.
    # Reports are saved to output_dir/reports/ with auto-generated filenames.
}

# ------------------------------------------------------------------------------
# 4. ICLABEL STAGE PARAMETERS (Component Classification)
# ------------------------------------------------------------------------------
# This stage runs after MEEGKit. It uses ICA + machine learning to remove
# residual biological/environmental artifacts from re-referenced data.
ICALABEL_PARAMS = {
    # RESIDUAL BAD CHANNEL DETECTION (stricter post-cleaning)
    "mad_threshold": 35.0,       # Higher threshold for already-cleaned data
    "min_amplitude_uv": 0.1,     # Flat channel check on re-referenced data

    # ICA DECOMPOSITION
    "n_components": 0.95,        # Retain components explaining 95% variance
    "random_state": 42,          # Fixed seed for reproducible Picard ICA

    # PROCESSING & REPORTING
    "interpolate_bads": True,    # Interpolate residual bads found here
    "verbose": True,             # Log ICA fitting and component exclusion
    "generate_report": True,     # Second HTML report for ICLabel residuals
    # NOTE: Do NOT set report_dir or subject_id here.
    # The pipeline manages these automatically to prevent collisions.
}

# ------------------------------------------------------------------------------
# 5. EXECUTE PIPELINE
# ------------------------------------------------------------------------------
# Run the sequential batch processor. Returns a dict mapping input filenames
# to their corresponding cleaned output paths.
saved_paths = preprocess_bel_trials(
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    meegkit_params=MEEGKIT_PARAMS,
    icalabel_params=ICALABEL_PARAMS,
    pattern="*_eeg_raw.fif",     # Glob pattern for file discovery (non-recursive)
    overwrite=True,              # Replace existing outputs without prompting
    verbose=True                 # Log top-level file iteration
)

# ------------------------------------------------------------------------------
# 6. SUMMARY
# ------------------------------------------------------------------------------
logging.info("EEG processing complete: %d files cleaned", len(saved_paths))

# ------------------------------------------------------------------------------
# 7. OUTPUT STRUCTURE EXPLANATION
# ------------------------------------------------------------------------------
# The pipeline creates a flat output directory with a dedicated reports folder:
#
# output_dir/
# ├── DP02_gain_eeg_raw_eeg.fif          # Cleaned continuous data
# ├── DP02_loss_eeg_raw_eeg.fif
# └── reports/
#     ├── DP02_meegkit_bad_channels_3d.html    # MEEGKit-stage QA report
#     └── DP02_icalabel_bad_channels_3d.html   # ICLabel-stage QA report
#
# KEY CHANGES FROM PREVIOUS VERSIONS:
# - Pipeline now manages report_dir and subject_id automatically
# - Non-recursive file discovery (no subdirectory traversal)
# - Flat output structure with reports/ subdirectory
# - Output files use _eeg suffix instead of _proc
# - recursive parameter has been removed from the API

# NOTE ABOUT MISSING REPORTS:
# When no bad channels are detected in a stage, the corresponding HTML
# report is not generated. Fewer report files than expected is normal
# and indicates clean data.
