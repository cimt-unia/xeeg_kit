# xeeg_kit/__init__.py

# Public API for the xeeg_kit BEL EEG preprocessing toolkit.
from ._config import *

from .bel_280 import parse_gpsc, create_montage_from_gpsc, BELStandardizer
from .bel_pipeline import preprocess_bel_trials, DEFAULT_RENAME_MAP, DEFAULT_EEG_PATTERN
from .artifact_cleaning import execute_meegkit, execute_icalabel
from .parallel import process_subjects_parallel
from .utils import detect_bad_channels, find_cleanest_segment

from .analysis import select_channels, plot_comparison, generate_bel_channel_map

from .viz import (
    get_bel_rename_map,
    print_anatomical_summary,
    load_bel_channel_map,
    plot_headset_3d,
    plot_bad_channels_3d
)

create_bel_channel_map = load_bel_channel_map

__version__ = "0.2.0"
__author__ = "CIMT"
