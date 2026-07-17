# xeeg_kit/bel_pipeline.py

# BEL EEG System One preprocessing pipeline supporting recursive BIDS-like layouts.
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import mne
from .bel_280 import parse_gpsc, create_montage_from_gpsc
from .artifact_cleaning import execute_meegkit, execute_icalabel

logger = logging.getLogger(__name__)

BEL_CHANNEL_COUNT = 280
DEFAULT_OUTPUT_SUFFIX = "__preproc"
DEFAULT_EEG_PATTERN = "*_eeg_raw.fif"
DEFAULT_RENAME_MAP: Dict[str, str] = {
    **{str(i): f"E{i}" for i in range(1, BEL_CHANNEL_COUNT + 1)},
    "REF CZ": "Cz"
}


def _process_single_bel_subject(
    fif_path: Path,
    out_path: Path,
    montage: mne.channels.DigMontage,
    rename_map: Dict[str, str],
    meegkit_params: Dict[str, Any],
    icalabel_params: Dict[str, Any],
    preload: bool,
    overwrite: bool
) -> None:
    raw = mne.io.read_raw_fif(str(fif_path), preload=preload, verbose="WARNING")

    existing_renames = {k: v for k, v in rename_map.items() if k in raw.ch_names}
    if existing_renames:
        raw.rename_channels(existing_renames)

    raw.set_montage(montage, on_missing="warn", verbose="WARNING")

    clean_1 = execute_meegkit(raw, **meegkit_params)
    clean_2 = execute_icalabel(clean_1, **icalabel_params)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_2.save(str(out_path), overwrite=overwrite, verbose="WARNING")
    logger.info("Saved: %s", out_path)


def preprocess_bel_trials(
    data_dir: Path,
    output_dir: Path,
    gpsc_path: Path,
    meegkit_params: Dict[str, Any],
    icalabel_params: Dict[str, Any],
    pattern: str = DEFAULT_EEG_PATTERN,
    recursive: bool = True,
    rename_map: Optional[Dict[str, str]] = None,
    preload: bool = True,
    overwrite: bool = True,
    verbose: bool = False
) -> Dict[str, Path]:
    if verbose:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    gpsc_path = Path(gpsc_path)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not gpsc_path.exists():
        raise FileNotFoundError(f"GPSC montage file not found: {gpsc_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    active_rename_map = dict(rename_map) if rename_map is not None else dict(DEFAULT_RENAME_MAP)
    channels = parse_gpsc(str(gpsc_path))
    montage = create_montage_from_gpsc(channels)

    glob_fn = data_dir.rglob if recursive else data_dir.glob
    fif_files: List[Path] = sorted(glob_fn(pattern))

    if not fif_files:
        raise ValueError(f"No files matching '{pattern}' found in {data_dir} (recursive={recursive})")

    saved_paths: Dict[str, Path] = {}
    for fif_path in fif_files:
        relative_parent = fif_path.parent.relative_to(data_dir)
        out_path = output_dir / relative_parent / f"{fif_path.stem}{DEFAULT_OUTPUT_SUFFIX}.fif"

        logger.info("Processing: %s", fif_path.relative_to(data_dir))

        _process_single_bel_subject(
            fif_path=fif_path,
            out_path=out_path,
            montage=montage,
            rename_map=active_rename_map,
            meegkit_params=meegkit_params,
            icalabel_params=icalabel_params,
            preload=preload,
            overwrite=overwrite
        )
        saved_paths[str(fif_path.relative_to(data_dir))] = out_path

    logger.info("Pipeline complete. %d file(s) processed.", len(saved_paths))
    return saved_paths
