# xeeg_kit/epoch_bel_pipeline.py
"""
BEL EEG epoch preprocessing pipeline.

Epoch-level counterpart to bel_pipeline.py. Takes pre-epoched data (*_epo.fif),
applies RANSAC, AutoReject, and optional ICA/ICLabel, then outputs cleaned
epochs (*_epo_cleaned.fif) optimized for source estimation.

Filtering is NOT performed here. Data must be filtered on continuous raw
before epoching to avoid edge artifacts at epoch boundaries.

Pipeline order:
  1. Standardize (rename channels + apply GPSC montage)
  2. RANSAC (global bad channel detection via spatial prediction)
  3. AutoReject (cross-validated per-trial repair + epoch rejection)
  4. (Optional) ICA + ICLabel (stereotyped artifact removal)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import mne

from .bel_280 import parse_gpsc, create_montage_from_gpsc, BELStandardizer
from .epoch_cleaning import (
    execute_ransac,
    execute_autoreject,
    execute_icalabel_epochs,
    save_epoch_qc_report,
    DEFAULT_AR_N_INTERPOLATES,
    DEFAULT_AR_CONSENSUS_PERCS,
)
from .utils import get_default_gpsc_path

logger = logging.getLogger(__name__)

BEL_CHANNEL_COUNT = 280
DEFAULT_EPOCH_PATTERN = "*_epo.fif"
DEFAULT_OUTPUT_SUFFIX = "_epo_cleaned"
REPORT_SUBDIR_NAME = "reports"

DEFAULT_RENAME_MAP: Dict[str, str] = {
    **{str(i): f"E{i}" for i in range(1, BEL_CHANNEL_COUNT + 1)},
    "REF CZ": "Cz",
}


def _process_single_bel_epoch_subject(
    fif_path: Path,
    out_path: Path,
    report_dir: Path,
    standardizer: BELStandardizer,
    ransac_params: Dict[str, Any],
    autoreject_params: Dict[str, Any],
    use_ransac: bool,
    use_icalabel: bool,
    icalabel_params: Dict[str, Any],
    preload: bool,
    overwrite: bool,
    verbose: bool,
) -> None:
    """Process a single subject's epoched data through the full pipeline."""
    # 1. Load Epochs (force full preload for EpochsFIF compatibility)
    epochs = mne.read_epochs(str(fif_path), preload=preload, verbose="WARNING")
    if not epochs.preload:
        epochs.load_data()
    logger.info(
        "Loaded: %d epochs, %d channels, %.1f Hz",
        len(epochs), len(epochs.ch_names), epochs.info['sfreq'],
    )

    # 2. Standardize (Rename + Montage)
    existing_renames = {
        k: v for k, v in standardizer.rename_map.items() if k in epochs.ch_names
    }
    if existing_renames:
        epochs.rename_channels(existing_renames)
    if epochs.get_montage() is None:
        channels = parse_gpsc(standardizer.gpsc_file)
        montage = create_montage_from_gpsc(channels)
        epochs.set_montage(montage, on_missing='warn', verbose="WARNING")

    subject_id = fif_path.stem.split("_")[0]

    # 3. RANSAC (global bad channel detection)
    if use_ransac:
        if verbose:
            logger.info("Running RANSAC...")
        epochs, ransac_bads = execute_ransac(
            epochs,
            n_resample=ransac_params.get('n_resample', 50),
            min_channels=ransac_params.get('min_channels', 0.25),
            min_corr=ransac_params.get('min_corr', 0.30),
            unbroken_time=ransac_params.get('unbroken_time', 0.99),
            n_jobs=ransac_params.get('n_jobs', -1),
            random_state=ransac_params.get('random_state', 42),
            generate_report=True,
            report_dir=report_dir,
            subject_id=f"{subject_id}_ransac",
            verbose=verbose,
        )
        # NOTE: We intentionally DO NOT set epochs.info['bads'] = ransac_bads here.
        # execute_ransac already interpolated these channels and cleared the bads
        # list so that AutoReject includes ALL channels in CV threshold learning.
        
    elif verbose:
        logger.info("RANSAC skipped (use_ransac=False).")

    # 4. AutoReject (per-trial repair + epoch rejection)
    if verbose:
        logger.info("Running AutoReject...")
    epochs_clean, reject_log = execute_autoreject(
        epochs,
        n_interpolates=autoreject_params.get('n_interpolates', DEFAULT_AR_N_INTERPOLATES),
        consensus_percs=autoreject_params.get('consensus_percs', DEFAULT_AR_CONSENSUS_PERCS),
        thresh_method=autoreject_params.get('thresh_method', 'bayesian_optimization'),
        cv=autoreject_params.get('cv', 10),
        random_state=autoreject_params.get('random_state', 42),
        n_jobs=autoreject_params.get('n_jobs', -1),
        verbose=verbose,
    )

    # 5. Optional: ICA + ICLabel
    if use_icalabel:
        if verbose:
            logger.info("Running ICA + ICLabel...")
        epochs_clean = execute_icalabel_epochs(
            epochs_clean,
            icalabel_thresholds=icalabel_params.get('icalabel_thresholds'),
            n_components=icalabel_params.get('n_components', 0.99),
            random_state=icalabel_params.get('random_state', 42),
            verbose=verbose,
        )

    # 6. Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs_clean.save(str(out_path), overwrite=overwrite, verbose="WARNING")
    logger.info("Saved cleaned epochs: %s (%d trials)", out_path.name, len(epochs_clean))

    # 7. QC Report
    save_epoch_qc_report(reject_log, report_dir, f"{subject_id}_autoreject")


def preprocess_bel_epochs(
    data_dir: Path,
    output_dir: Path,
    gpsc_path: Optional[Path] = None,
    ransac_params: Optional[Dict[str, Any]] = None,
    autoreject_params: Optional[Dict[str, Any]] = None,
    use_ransac: bool = True,
    use_icalabel: bool = False,
    icalabel_params: Optional[Dict[str, Any]] = None,
    pattern: str = DEFAULT_EPOCH_PATTERN,
    rename_map: Optional[Dict[str, str]] = None,
    preload: bool = True,
    overwrite: bool = True,
    verbose: bool = False,
) -> Dict[str, Path]:
    """Batch process epoched BEL EEG data through the full cleaning pipeline."""
    if verbose:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    resolved_gpsc = Path(gpsc_path) if gpsc_path is not None else get_default_gpsc_path()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not resolved_gpsc.exists():
        raise FileNotFoundError(f"GPSC montage file not found: {resolved_gpsc}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / REPORT_SUBDIR_NAME
    report_dir.mkdir(parents=True, exist_ok=True)

    active_rename_map = dict(rename_map) if rename_map is not None else dict(DEFAULT_RENAME_MAP)
    standardizer = BELStandardizer(gpsc_file=resolved_gpsc, rename_map=active_rename_map)

    fif_files: List[Path] = sorted(data_dir.glob(pattern))
    if not fif_files:
        raise ValueError(f"No files matching '{pattern}' found in {data_dir}")

    saved_paths: Dict[str, Path] = {}
    for fif_path in fif_files:
        out_path = output_dir / f"{fif_path.stem}{DEFAULT_OUTPUT_SUFFIX}.fif"
        logger.info("Processing epochs: %s", fif_path.name)
        _process_single_bel_epoch_subject(
            fif_path=fif_path,
            out_path=out_path,
            report_dir=report_dir,
            standardizer=standardizer,
            ransac_params=ransac_params or {},
            autoreject_params=autoreject_params or {},
            use_ransac=use_ransac,
            use_icalabel=use_icalabel,
            icalabel_params=icalabel_params or {},
            preload=preload,
            overwrite=overwrite,
            verbose=verbose,
        )
        saved_paths[fif_path.name] = out_path

    logger.info("Pipeline complete. %d file(s) processed.", len(saved_paths))
    return saved_paths
