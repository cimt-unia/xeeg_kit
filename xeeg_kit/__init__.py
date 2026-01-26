# xeeg_kit/__init__.py
"""
xeeg_kit: Preprocessing toolkit for high-density EEG (e.g., BEL EEG System One).
"""
from ._config import *  # Set environment variables first

from .bel_280 import parse_gpsc, create_montage_from_gpsc, BELStandardizer
from .artifact_cleaning import execute_meegkit, execute_icalabel
from .parallel import process_subjects_parallel, process_single_subject
from .utils import detect_bad_channels, find_cleanest_segment

__version__ = "0.1.0"
__author__ = "CIMT"
