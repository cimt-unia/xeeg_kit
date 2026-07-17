# xeeg_kit/parallel.py

# Parallel processing orchestration for BEL EEG System One datasets.
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from joblib import Parallel, delayed
from .bel_280 import parse_gpsc, create_montage_from_gpsc
from .bel_pipeline import _process_single_bel_subject, DEFAULT_RENAME_MAP

logger = logging.getLogger(__name__)

BLAS_THREAD_VARS = [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"
]


def verify_parallel_config() -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "blas_threads_limited": False,
        "warnings": [],
        "recommendations": []
    }
    limited = all(os.environ.get(var) == "1" for var in BLAS_THREAD_VARS)
    config["blas_threads_limited"] = limited
    if not limited:
        config["warnings"].append(
            "BLAS threading not limited. Set OMP/OPENBLAS/MKL/VECLIB/NUMEXPR NUM_THREADS to '1' "
            "to prevent severe slowdown during parallel ICA."
        )
    try:
        import multiprocessing
        n_cpus = multiprocessing.cpu_count()
        config["n_cpus"] = n_cpus
        config["recommendations"].append(f"Detected {n_cpus} CPU cores. Use n_jobs=-1 to utilize all.")
    except Exception as e:
        config["recommendations"].append(f"Could not detect CPU count: {e}")
    return config


def _parse_subject_from_stem(stem: str) -> Optional[str]:
    match = re.search(r"(sub-\d+)", stem)
    return match.group(1) if match else None


def _resolve_output_path(
    input_path: Path,
    output_dir: Path,
    subject_id: str,
    output_suffix: str,
    output_structure: str
) -> Path:
    stem = input_path.stem
    if output_structure == "flat":
        return output_dir / f"{stem}{output_suffix}.fif"
    if output_structure == "mirror":
        rel_parent = input_path.relative_to(input_path.parent.parent).parent
        return output_dir / rel_parent / f"{stem}{output_suffix}.fif"
    if output_structure == "parsed":
        parts = stem.split("_")
        sub_idx = next((i for i, p in enumerate(parts) if re.match(r"^sub-\d+$", p)), None)
        dataset = "_".join(parts[:sub_idx]) if sub_idx is not None and sub_idx > 0 else (parts[0] if parts else "unknown")
        return output_dir / subject_id / dataset / f"{stem}{output_suffix}.fif"
    raise ValueError(f"Unknown output_structure: '{output_structure}'. Choose from: flat, mirror, parsed")


def process_subjects_parallel(
    input_files: List[Union[str, Path]],
    output_dir: Union[str, Path],
    gpsc_path: Union[str, Path],
    meegkit_params: Dict[str, Any],
    icalabel_params: Dict[str, Any],
    subject_ids: Optional[List[str]] = None,
    n_jobs: int = -1,
    rename_map: Optional[Dict[str, str]] = None,
    output_suffix: str = "_cleaned",
    output_structure: str = "flat",
    preload: bool = True,
    overwrite: bool = True,
    verify_config: bool = True,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    gpsc_path = Path(gpsc_path)

    if not gpsc_path.exists():
        raise FileNotFoundError(f"GPSC file not found: {gpsc_path}")

    if verify_config and verbose:
        config = verify_parallel_config()
        for warning in config["warnings"]:
            logger.warning(warning)

    active_rename_map = rename_map if rename_map is not None else DEFAULT_RENAME_MAP
    channels = parse_gpsc(gpsc_path)
    montage = create_montage_from_gpsc(channels)

    if subject_ids is None:
        subject_ids = [
            _parse_subject_from_stem(Path(f).stem) or f"sub-{i+1:02d}"
            for i, f in enumerate(input_files)
        ]
    if len(subject_ids) != len(input_files):
        raise ValueError("Number of subject_ids must match input_files")

    output_files = [
        _resolve_output_path(Path(f), output_dir_path, sid, output_suffix, output_structure)
        for f, sid in zip(input_files, subject_ids)
    ]

    if verbose:
        logger.info("Processing %d subjects with n_jobs=%d", len(input_files), n_jobs)

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10 if verbose else 0)(
        delayed(_run_bel_worker)(
            in_f, out_f, sid, montage, active_rename_map,
            meegkit_params, icalabel_params, preload, overwrite, verbose
        )
        for in_f, out_f, sid in zip(input_files, output_files, subject_ids)
    )

    if verbose:
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        total_time = sum(r["processing_time"] for r in results)
        avg_time = total_time / len(results) if results else 0
        logger.info("SUMMARY: %d/%d subjects processed successfully", len(successful), len(results))
        logger.info("Average processing time: %.1fs per subject", avg_time)
        if failed:
            logger.error("Failed subjects (%d):", len(failed))
            for r in failed:
                logger.error("  - %s: %s", r["subject_id"], r["error"][:80])

    return results


def _run_bel_worker(
    input_file: Path,
    output_file: Path,
    subject_id: str,
    montage: Any,
    rename_map: Dict[str, str],
    meegkit_params: Dict[str, Any],
    icalabel_params: Dict[str, Any],
    preload: bool,
    overwrite: bool,
    verbose: bool
) -> Dict[str, Any]:
    log_path = output_file.parent / f"{output_file.stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    subject_logger = logging.getLogger(f"xeeg_kit.subject.{subject_id}")
    subject_logger.handlers.clear()
    subject_logger.setLevel(logging.INFO if verbose else logging.WARNING)

    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(formatter)
    subject_logger.addHandler(fh)

    start_time = time.time()
    result: Dict[str, Any] = {
        "subject_id": subject_id,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "success": False,
        "error": None,
        "processing_time": 0.0,
        "log_file": str(log_path)
    }

    try:
        subject_logger.info("Starting pipeline for %s", input_file.name)
        _process_single_bel_subject(
            fif_path=input_file,
            out_path=output_file,
            montage=montage,
            rename_map=rename_map,
            meegkit_params=meegkit_params,
            icalabel_params=icalabel_params,
            preload=preload,
            overwrite=overwrite
        )
        result["success"] = True
        result["processing_time"] = time.time() - start_time
        subject_logger.info("Completed in %.1fs -> %s", result["processing_time"], output_file.name)
    except Exception as e:
        result["error"] = str(e)
        result["processing_time"] = time.time() - start_time
        subject_logger.error("Failed: %s", str(e)[:200])
    finally:
        fh.close()

    return result

