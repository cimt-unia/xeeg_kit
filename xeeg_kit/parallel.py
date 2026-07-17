# xeeg_kit/parallel.py

# Parallel processing orchestration for high-density EEG datasets.
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable
import mne
from joblib import Parallel, delayed
from .artifact_cleaning import execute_meegkit, execute_icalabel
from .bel_280 import BELStandardizer

logger = logging.getLogger(__name__)

def verify_parallel_config() -> Dict[str, Any]:
    config = {'blas_threads_limited': False, 'warnings': [], 'recommendations': []}
    thread_vars = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]
    limited = all(os.environ.get(var) == "1" for var in thread_vars)
    config['blas_threads_limited'] = limited
    if not limited:
        config['warnings'].append("BLAS threading not limited! This will cause severe slowdown when running ICA in parallel. Set OMP/OPENBLAS/MKL/VECLIB NUM_THREADS to '1'.")
    try:
        import multiprocessing
        n_cpus = multiprocessing.cpu_count()
        config['n_cpus'] = n_cpus
        config['recommendations'].append(f"Detected {n_cpus} CPU cores. Use n_jobs=-1 to utilize all cores.")
    except Exception as e:
        config['recommendations'].append(f"Could not detect CPU count: {e}")
    return config

def _parse_subject_from_stem(stem: str) -> Optional[str]:
    match = re.search(r'(sub-\d+)', stem)
    return match.group(1) if match else None

def process_single_subject(
    input_file: Union[str, Path], output_file: Union[str, Path], subject_id: str,
    gpsc_file: Optional[Union[str, Path]] = None, channel_rename_map: Optional[Dict[str, str]] = None,
    run_meegkit: bool = True, run_icalabel: bool = True, meegkit_params: Optional[Dict] = None,
    icalabel_params: Optional[Dict] = None, verbose: bool = True
) -> Dict[str, Any]:
    output_path = Path(output_file)
    log_path = output_path.parent / f"{output_path.stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    subject_logger = logging.getLogger(f"xeeg_kit.subject.{subject_id}")
    subject_logger.handlers.clear()
    subject_logger.setLevel(logging.INFO if verbose else logging.WARNING)
    
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(formatter)
    subject_logger.addHandler(fh)
    
    start_time = time.time()
    result = {'subject_id': subject_id, 'input_file': str(input_file), 'output_file': str(output_file), 'success': False, 'error': None, 'processing_time': 0, 'log_file': str(log_path)}

    try:
        subject_logger.info("Loading data from %s", Path(input_file).name)
        raw = mne.io.read_raw(input_file, preload=True, verbose=False)
        if gpsc_file is not None or channel_rename_map is not None:
            standardizer = BELStandardizer(gpsc_file, channel_rename_map)
            raw = standardizer.standardize(raw)
        if run_meegkit:
            subject_logger.info("Running MEEGKit cleaning...")
            raw = execute_meegkit(raw, **(meegkit_params or {}))
        if run_icalabel:
            subject_logger.info("Running ICA + ICLabel...")
            raw = execute_icalabel(raw, **(icalabel_params or {}))
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw.save(output_file, overwrite=True, verbose=False)
        result['success'] = True
        result['processing_time'] = time.time() - start_time
        subject_logger.info("Completed in %.1fs -> %s", result['processing_time'], output_path.name)
    except Exception as e:
        result['error'] = str(e)
        result['processing_time'] = time.time() - start_time
        subject_logger.error("Failed: %s", str(e)[:100])
    finally:
        fh.close()
        
    return result

def process_subjects_parallel(
    input_files: List[Union[str, Path]], output_dir: Union[str, Path], subject_ids: Optional[List[str]] = None,
    n_jobs: int = -1, gpsc_file: Optional[Union[str, Path]] = None, channel_rename_map: Optional[Dict[str, str]] = None,
    run_meegkit: bool = True, run_icalabel: bool = True, meegkit_params: Optional[Dict] = None,
    icalabel_params: Optional[Dict] = None, output_suffix: str = "_cleaned", output_structure: str = "flat",
    custom_output_fn: Optional[Callable] = None, verify_config: bool = True, verbose: bool = True
) -> List[Dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if verify_config and verbose:
        config = verify_parallel_config()
        if config['warnings']:
            logger.warning("PARALLEL CONFIGURATION WARNING")
            for warning in config['warnings']: logger.warning(warning)
            
    if subject_ids is None:
        subject_ids = [_parse_subject_from_stem(Path(f).stem) or f"sub-{i+1:02d}" for i, f in enumerate(input_files)]
    if len(subject_ids) != len(input_files):
        raise ValueError("Number of subject_ids must match input_files")

    output_files = []
    for input_file, subject_id in zip(input_files, subject_ids):
        input_path = Path(input_file)
        stem = input_path.stem
        if output_structure == "flat": output_files.append(output_dir / f"{stem}{output_suffix}.fif")
        elif output_structure == "mirror": output_files.append(output_dir / input_path.relative_to(input_path.parent.parent).parent / f"{stem}{output_suffix}.fif")
        elif output_structure == "parsed":
            parts = stem.split('_')
            sub_idx = next((i for i, p in enumerate(parts) if re.match(r'^sub-\d+$', p)), None)
            dataset = '_'.join(parts[:sub_idx]) if sub_idx is not None and sub_idx > 0 else (parts[0] if parts else 'unknown')
            output_files.append(output_dir / subject_id / dataset / f"{stem}{output_suffix}.fif")
        elif output_structure == "custom":
            if custom_output_fn is None: raise ValueError("custom_output_fn must be provided")
            output_files.append(Path(custom_output_fn(input_path, output_dir, subject_id)))
        else: raise ValueError(f"Unknown output_structure: '{output_structure}'")

    if verbose:
        logger.info("Processing %d subjects with n_jobs=%d", len(input_files), n_jobs)
        
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10 if verbose else 0)(
        delayed(process_single_subject)(input_file, output_file, subject_id, gpsc_file, channel_rename_map, run_meegkit, run_icalabel, meegkit_params, icalabel_params, verbose)
        for input_file, output_file, subject_id in zip(input_files, output_files, subject_ids)
    )

    if verbose:
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        total_time = sum(r['processing_time'] for r in results)
        logger.info("SUMMARY: %d/%d subjects processed successfully", len(successful), len(results))
        logger.info("Average processing time: %.1fs per subject", total_time / len(results) if results else 0)
        if failed:
            logger.error("Failed subjects (%d):", len(failed))
            for r in failed: logger.error("  - %s: %s", r['subject_id'], r['error'][:80])
            
    return results

