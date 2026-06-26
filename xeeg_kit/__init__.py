# xeeg_kit/__init__.py
"""
xeeg_kit: Preprocessing toolkit for high-density EEG (e.g., BEL EEG System One).
"""
from ._config import * 

from .bel_280 import parse_gpsc, create_montage_from_gpsc, BELStandardizer
from .artifact_cleaning import execute_meegkit, execute_icalabel
from .parallel import process_subjects_parallel, process_single_subject
from .utils import detect_bad_channels, find_cleanest_segment

# Analysis & Visualization Tools
from .analysis import generate_bel_channel_map, select_channels, plot_comparison
from .viz import (
    get_bel_rename_map,
    print_anatomical_summary,
    create_bel_channel_map,
    plot_headset_3d,
    plot_bad_channels_3d
)

__version__ = "0.1.0"
__author__ = "CIMT"
