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
# Define input and output paths. The pipeline expects a BIDS-like structure.
# Input files must be .fif format and match the glob pattern defined below.
DATA_DIR = Path("/path/to/derivatives/study_name/eeg")
OUTPUT_DIR = Path("/path/to/derivatives/study_name/eeg/clean")

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
    "report_dir": OUTPUT_DIR,    # Save reports to output directory
    "subject_id": "sub-01_meegkit",  # Unique ID prevents report collisions
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
    "report_dir": OUTPUT_DIR,    # Same directory as MEEGKit reports
    "subject_id": "sub-01_icalabel",  # Distinct suffix avoids overwriting
}

# ------------------------------------------------------------------------------
# 5. EXECUTE PIPELINE
# ------------------------------------------------------------------------------
# Run the sequential batch processor. Returns a dict mapping input paths
# to their corresponding cleaned output paths.
saved_paths = preprocess_bel_trials(
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    meegkit_params=MEEGKIT_PARAMS,
    icalabel_params=ICALABEL_PARAMS,
    pattern="*_eeg_raw.fif",     # Glob pattern for file discovery
    recursive=True,              # Search subdirectories (preserves structure)
    overwrite=True,              # Replace existing outputs without prompting
    verbose=True                 # Log top-level file iteration
)

# ------------------------------------------------------------------------------
# 6. SUMMARY
# ------------------------------------------------------------------------------
logging.info("EEG processing complete: %d files cleaned", len(saved_paths))

# NOTE FOR PRODUCTION BATCHES:
# The subject_id above is hardcoded for tutorial clarity. When processing
# multiple subjects, extract IDs dynamically from filenames to prevent
# all reports from overwriting each other. Example:
#   subject_id = filepath.stem.split("_")[0] + "_meegkit"