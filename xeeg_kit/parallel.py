# xeeg_kit/parallel.py

"""
Parallel processing for the BEL EEG System One (280-channel Geodesic HD-EEG).

⚠️ This module is designed exclusively for BEL 280-channel datasets.
It requires:
- A .gpsc file for sensor positions
- Channel names compatible with BEL convention (e.g., 'E1', 'E2', ...)
- Optional renaming map if input channels differ (e.g., EGI exports as '1', '2', ...)

For general-purpose preprocessing (any EEG system), use:
    from xeeg_kit import execute_meegkit, execute_icalabel
"""

import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable
from joblib import Parallel, delayed
from .utils import log, verify_parallel_config


class LogCapture:
    def __init__(self):
        self.redirect = None

    def __call__(self, msg: str):
        if self.redirect is not None:
            self.redirect(msg)
        else:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {msg}")


# Global log instance
log = LogCapture()


def _parse_subject_from_stem(stem: str) -> Optional[str]:
    """
    Extract the first 'sub-XX' token from a filename stem.
    e.g. 'rest_off_sub-08_c_eeg' → 'sub-08'
    Returns None if no sub-XX token is found.
    """
    match = re.search(r'(sub-\d+)', stem)
    return match.group(1) if match else None


def process_single_subject(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    subject_id: str,
    gpsc_file: Optional[Union[str, Path]] = None,
    channel_rename_map: Optional[Dict[str, str]] = None,
    run_meegkit: bool = True,
    run_icalabel: bool = True,
    meegkit_params: Optional[Dict] = None,
    icalabel_params: Optional[Dict] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Process a single subject and save its log to a .log file next to the output.

    Fix applied: log_path.parent.mkdir(parents=True, exist_ok=True) is now
    called inside the finally block BEFORE opening the log file, so the log
    directory is guaranteed to exist regardless of whether the try block ran
    (e.g. when workers race on the same subject directory).
    """
    from .artifact_cleaning import execute_meegkit, execute_icalabel
    from .bel_280 import BELStandardizer
    import mne

    local_logs = []

    def capture_log(msg: str):
        if verbose:
            timestamp = time.strftime("%H:%M:%S")
            full_msg = f"[{timestamp}] [{subject_id}] {msg}"
            local_logs.append(full_msg)

    original_redirect = log.redirect
    log.redirect = capture_log

    start_time = time.time()
    result = {
        'subject_id': subject_id,
        'input_file': str(input_file),
        'output_file': str(output_file),
        'success': False,
        'error': None,
        'processing_time': 0,
        'log_file': None
    }

    try:
        log(f"Loading data from {Path(input_file).name}")
        raw = mne.io.read_raw(input_file, preload=True, verbose=False)

        # Apply BEL standardization if needed
        if gpsc_file is not None or channel_rename_map is not None:
            standardizer = BELStandardizer(gpsc_file, channel_rename_map)
            raw = standardizer.standardize(raw)

        if run_meegkit:
            log("Running MEEGKit cleaning...")
            raw = execute_meegkit(raw, **(meegkit_params or {}))

        if run_icalabel:
            log("Running ICA + ICLabel...")
            raw = execute_icalabel(raw, **(icalabel_params or {}))

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw.save(output_file, overwrite=True, verbose=False)

        result['success'] = True
        result['processing_time'] = time.time() - start_time
        log(f"✅ Completed in {result['processing_time']:.1f}s → {output_path.name}")

    except Exception as e:
        error_msg = str(e)
        result['error'] = error_msg
        result['processing_time'] = time.time() - start_time
        log(f"❌ Failed: {error_msg[:100]}")

    finally:
        log.redirect = original_redirect
        output_path = Path(output_file)
        log_path = output_path.parent / f"{output_path.stem}.log"
        try:
            # FIX: always ensure the log directory exists before writing,
            # regardless of whether the try block created it (e.g. on failure,
            # or when parallel workers race on the same subject directory).
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'w') as f:
                for msg in local_logs:
                    f.write(msg + '\n')
            result['log_file'] = str(log_path)
        except Exception as e:
            if verbose:
                print(f"[WARNING] Could not write log for {subject_id}: {e}")

    return result


def process_subjects_parallel(
    input_files: List[Union[str, Path]],
    output_dir: Union[str, Path],
    subject_ids: Optional[List[str]] = None,
    n_jobs: int = -1,
    gpsc_file: Optional[Union[str, Path]] = None,
    channel_rename_map: Optional[Dict[str, str]] = None,
    run_meegkit: bool = True,
    run_icalabel: bool = True,
    meegkit_params: Optional[Dict] = None,
    icalabel_params: Optional[Dict] = None,
    output_suffix: str = "_cleaned",
    output_structure: str = "flat",
    custom_output_fn: Optional[Callable] = None,
    verify_config: bool = True,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Process multiple subjects in parallel.

    Fixes applied vs original:
    - subject_ids now defaults to parsing 'sub-XX' from each filename stem
      instead of sequential numbering (sub-01..sub-N). This means passing
      48 trial files from 12 subjects correctly yields 12 unique subject IDs
      rather than 48 unique ones.
    - "parsed" output_structure now finds the subject token by scanning for
      'sub-XX' in the stem parts, then treats everything before it as the
      dataset name. This correctly handles multi-word dataset names like
      'rest_off' that contain underscores.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verify_config and verbose:
        config = verify_parallel_config()
        if config['warnings']:
            log("\n" + "=" * 60)
            log("⚠️  PARALLEL CONFIGURATION WARNING")
            log("=" * 60)
            for warning in config['warnings']:
                log(warning)
            log("\nContinuing anyway, but performance will be suboptimal...")
            log("=" * 60 + "\n")

    # FIX: default to parsing sub-XX from filename instead of sequential numbering
    if subject_ids is None:
        subject_ids = []
        for f in input_files:
            parsed = _parse_subject_from_stem(Path(f).stem)
            subject_ids.append(parsed if parsed else f"sub-{len(subject_ids)+1:02d}")

    if len(subject_ids) != len(input_files):
        raise ValueError(
            f"Number of subject_ids ({len(subject_ids)}) must match "
            f"input_files ({len(input_files)})"
        )

    output_files = []
    for input_file, subject_id in zip(input_files, subject_ids):
        input_path = Path(input_file)
        stem = input_path.stem

        if output_structure == "flat":
            output_name = stem + output_suffix + ".fif"
            output_files.append(output_dir / output_name)

        elif output_structure == "mirror":
            rel_path = input_path.relative_to(input_path.parent.parent)
            output_path = output_dir / rel_path.parent / (rel_path.stem + output_suffix + ".fif")
            output_files.append(output_path)

        elif output_structure == "parsed":
            # FIX: scan parts for the sub-XX token instead of assuming parts[0]
            # is the dataset. This handles dataset names with underscores
            # (e.g. 'rest_off_sub-08_c_eeg' → dataset='rest_off', subject='sub-08').
            parts = stem.split('_')
            sub_idx = next(
                (i for i, p in enumerate(parts) if re.match(r'^sub-\d+$', p)),
                None
            )
            if sub_idx is not None and sub_idx > 0:
                dataset = '_'.join(parts[:sub_idx])          # e.g. 'rest_off'
                subject = parts[sub_idx]                     # e.g. 'sub-08'
            else:
                dataset = parts[0] if parts else 'unknown'
                subject = subject_id
            output_path = output_dir / subject / dataset / (stem + output_suffix + ".fif")
            output_files.append(output_path)

        elif output_structure == "custom":
            if custom_output_fn is None:
                raise ValueError(
                    "custom_output_fn must be provided when output_structure='custom'"
                )
            output_path = custom_output_fn(input_path, output_dir, subject_id)
            output_files.append(Path(output_path))

        else:
            raise ValueError(
                f"Unknown output_structure: '{output_structure}'. "
                f"Choose from: 'flat', 'mirror', 'parsed', 'custom'"
            )

    if verbose:
        log(f"Processing {len(input_files)} subjects with n_jobs={n_jobs}")
        log(f"Output directory: {output_dir}")
        log(f"Output structure: {output_structure}")

    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10 if verbose else 0)(
        delayed(process_single_subject)(
            input_file=input_file,
            output_file=output_file,
            subject_id=subject_id,
            gpsc_file=gpsc_file,
            channel_rename_map=channel_rename_map,
            run_meegkit=run_meegkit,
            run_icalabel=run_icalabel,
            meegkit_params=meegkit_params,
            icalabel_params=icalabel_params,
            verbose=True
        )
        for input_file, output_file, subject_id in zip(input_files, output_files, subject_ids)
    )

    if verbose:
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        total_time = sum(r['processing_time'] for r in results)
        avg_time = total_time / len(results) if results else 0

        log("\n" + "=" * 60)
        log(f"SUMMARY: {len(successful)}/{len(results)} subjects processed successfully")
        log(f"Average processing time: {avg_time:.1f}s per subject")
        log(f"Total wall time: {max(r['processing_time'] for r in results):.1f}s")

        if failed:
            log(f"\n❌ Failed subjects ({len(failed)}):")
            for r in failed:
                log(f"  - {r['subject_id']}: {r['error'][:80]}")
        log("=" * 60)

    return results
