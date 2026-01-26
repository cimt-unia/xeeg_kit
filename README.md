# XTRA EEG Preprocessing  
### 

**Core Philosophy: “Repair—Not Reject”**     

We present a comprehensive framework for processing continuous high-density EEG data (280 channels) that combines state-of-the-art artifact removal techniques. The pipeline integrates MEEGKit algorithms (ASR, STAR, SNS) with ICA deep learning component classification (ICLabel).  

Our approach addresses the unique challenges of long-duration recordings while maintaining spatial precision through enhanced coregistration and coordinate normalization.

A preprocessing toolkit for high-density EEG data, optimized for the **BEL EEG System One (280-channel Geodesic HD-EEG)** but also supporting general-purpose artifact cleaning for any EEG system.

Built on [MNE-Python](https://mne.tools), `xeeg_kit` integrates:
- **MEEGKit** (ASR + STAR + SNS) for robust denoising
- **ICLabel** for automatic ICA component rejection
- BEL-specific utilities for channel renaming and montage handling
- Parallel processing for batch analysis of BEL datasets

---

## 📦 Installation

Install from GitHub (requires Python ≥ 3.9):

```bash
pip install git+https://github.com/cimt-unia/xeeg_kit.git
```

> **Dependencies**: `mne`, `meegkit`, `mne-icalabel`, `numpy`, `scipy`, `joblib`

---

## Core Features

### 1. **Generic Cleaning Pipeline** (any EEG format)  
Use `execute_meegkit` and `execute_icalabel` with **any MNE-supported format** (`.fif`, `.mff`, `.set`, etc.).

### 2. **BEL 280-Channel Support**  
- Parse `.gpsc` sensor position files  
- Standardize channel names (e.g., EGI `'1'` → BEL `'E1'`)  
- Apply correct montage with fiducials (`FidNz`, `FidT9`, `FidT10`)

### 3. **Parallel Batch Processing**  
Process hundreds of BEL subjects in parallel with per-subject logging.

---

## Usage Examples

### A. General Preprocessing (Any EEG System)

```python
import mne
from xeeg_kit import execute_meegkit, execute_icalabel

# Load data (EGI .mff, FIF, Brainstorm, etc.)
raw = mne.io.read_raw_egi("subject.mff", preload=True)

# Optional: Standardize for BEL 280 (if needed)
from xeeg_kit import bel_280
rename_map = {str(i): f'E{i}' for i in range(1, 281)}
rename_map['REF CZ'] = 'Cz'
raw.rename_channels(rename_map)

channels = bel_280.parse_gpsc("ghw280_from_egig.gpsc")
montage = bel_280.create_montage_from_gpsc(channels)
raw.set_montage(montage, on_missing="warn")

# Run cleaning
clean_1 = execute_meegkit(
    raw,
    low_pass_filter=100.0,
    notch_filter_freq=60.0,
    verbose=True
)

clean_2 = execute_icalabel(clean_1, verbose=True)

# Save
clean_2.save("final_cleaned.fif", overwrite=True)
```

### B. BEL 280 Parallel Processing

```python
from xeeg_kit import process_subjects_parallel

input_files = ["sub01.mff", "sub02.mff", "sub03.mff"]
output_dir = "cleaned_data/"

# Define cleaning parameters
meegkit_params = {
    'low_pass_filter': 100.0,
    'notch_filter_freq': 60.0,
    'mad_threshold': 4.5,
    'asr_cutoff': 3.0,
    'star_thresh': 2.0,
    'sns_neighbors': 8
}

icalabel_params = {
    'mad_threshold': 10.0,
    'n_components': 0.98,
    'random_state': 97
}

# Run parallel pipeline
results = process_subjects_parallel(
    input_files=input_files,
    output_dir=output_dir,
    gpsc_file="ghw280_from_egig.gpsc",
    channel_rename_map={str(i): f'E{i}' for i in range(1, 281)},
    meegkit_params=meegkit_params,
    icalabel_params=icalabel_params,
    n_jobs=-1,  # Use all CPU cores
    verbose=True
)

# Each subject gets a .log file with detailed processing info
```

---

## Important Notes

- **BEL-Specific Modules**:  
  The `process_subjects_parallel` function is designed **only for BEL 280-channel data**. For other systems, use the generic `execute_meegkit`/`execute_icalabel` workflow.

- **Filtering**:  
  `execute_meegkit` always applies a **1.0 Hz high-pass filter** (required for ASR stability). The low-pass cutoff is configurable via `low_pass_filter`.

- **Notch Filtering**:  
  Set `notch_filter_freq=None` to skip line-noise removal.

- **File Requirements**:  
  The `.gpsc` file must be provided for BEL montage. Example: `ghw280_from_egig.gpsc`.

---

## 📁 Project Structure

```
xeeg_kit/
├── artifact_cleaning.py   # Core cleaning functions (format-agnostic)
├── bel_280.py            # BEL 280-channel utilities
├── parallel.py           # BEL-specific parallel processing
├── utils.py              # Shared helpers (bad channel detection, etc.)
└── _config.py            # BLAS thread limits (critical for parallelism)
```

---

## 📜 License

MIT License — free for academic and commercial use.

---

## 🙏 Acknowledgements

- [MNE-Python](https://mne.tools)
- [MEEGKit](https://nbara.github.io/python-meegkit/)
- [MNE-ICALabel](https://mne.tools/mne-icalabel)
```

---

This version now has:
- **Scientific depth** (your "Repair—Not Reject" philosophy)
- **Technical clarity** (working examples, clear scope)
- **Professional polish** (consistent links, clean layout)

Perfect for GitHub, lab sharing, or even supplementary material in a paper.

Well done! 🎉
