# xeeg_kit/analysis.py

# Analysis tools for anatomical mapping, channel selection, and comparative plotting.
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .bel_280 import parse_gpsc, create_montage_from_gpsc

logger = logging.getLogger(__name__)
mne.set_log_level('ERROR')

ORBITAL_THRESH_Y, ORBITAL_THRESH_Z = 0.065, 0.015
PFC_Z_THRESH, INFERIOR_FRONTAL_Z_THRESH = 0.015, -0.005
Y_PARIETAL_THRESH, Y_OCCIPITAL_THRESH = -0.035, -0.06
X_LATERAL_THRESH, Z_CORTEX_MIN = 0.06, -0.03
Y_FRONTAL_THRESH = 0.04

COLOR_1 = '#6A0DAD'
COLOR_2 = '#1F78B4'
HIGHLIGHT_ALPHA = 0.3

def _assign_region(row: pd.Series) -> str:
    y, x_abs, z = row['y'], abs(row['x']), row['z']
    if y > ORBITAL_THRESH_Y and z < ORBITAL_THRESH_Z: return "Orbital/Eye"
    if y > Y_FRONTAL_THRESH:
        if z >= PFC_Z_THRESH: return "Prefrontal" if x_abs < X_LATERAL_THRESH else "Fronto-Temporal"
        elif z < INFERIOR_FRONTAL_Z_THRESH: return "Inferior-Frontal/Jaw"
        else: return "Prefrontal" if x_abs < X_LATERAL_THRESH else "Fronto-Temporal"
    elif y > Y_PARIETAL_THRESH: 
        return "Lateral-Inferior/Jaw" if z < Z_CORTEX_MIN else ("Central" if x_abs < X_LATERAL_THRESH else "Temporal")
    elif y > Y_OCCIPITAL_THRESH: 
        if z < Z_CORTEX_MIN: return "Cerebellar/Neck"
        if x_abs > X_LATERAL_THRESH:
            mid_y = (Y_PARIETAL_THRESH + Y_OCCIPITAL_THRESH) / 2
            return "Occipital" if y < mid_y else "Temporal"
        return "Parietal"
    else: 
        return "Cerebellar/Neck" if z < Z_CORTEX_MIN else "Occipital"

def generate_bel_channel_map(gpsc_file: str, output_csv: str = "bel_280_channel_map.csv") -> pd.DataFrame:
    if Path(output_csv).exists():
        logger.info("Loading existing channel map: %s", output_csv)
        return pd.read_csv(output_csv)

    logger.info("Generating Anatomical Channel Map from GPSC...")
    channels = parse_gpsc(Path(gpsc_file))
    montage = create_montage_from_gpsc(channels)
    ch_pos = montage.get_positions()['ch_pos']
    
    data = [{'name': n, 'x': p[0], 'y': p[1], 'z': p[2]} for n, p in ch_pos.items() if p is not None and not np.any(np.isnan(p))]
    df = pd.DataFrame(data)
    if df.empty: 
        raise ValueError("No valid positions found.")

    df['region'] = df.apply(_assign_region, axis=1)
    df.to_csv(output_csv, index=False)
    logger.info("Channel map saved to %s", output_csv)
    return df

def select_channels(
    raw_data: mne.io.Raw, 
    map_df: Optional[pd.DataFrame], 
    method: str = "region", 
    region_name: Optional[str] = None, 
    manual_list: Optional[List[str]] = None
) -> List[str]:
    available_chs = set(raw_data.ch_names)
    
    if method == "manual":
        if not manual_list: raise ValueError("Manual list is empty.")
        selected = [ch for ch in manual_list if ch in available_chs]
        missing = set(manual_list) - set(selected)
        if missing: logger.warning("Channels not found in data: %s", missing)
    elif method == "region":
        if not region_name: raise ValueError("Region name not specified.")
        assert map_df is not None, "map_df required for region selection."
        region_chs = map_df[map_df['region'] == region_name]['name'].tolist()
        selected = [ch for ch in region_chs if ch in available_chs]
        if not selected: raise ValueError(f"No channels found for region '{region_name}'.")
    else:
        raise ValueError("Method must be 'region' or 'manual'.")
        
    logger.info("Selected %d channels using method '%s'.", len(selected), method)
    return sorted(selected)

def _extract_snippet(data_full: np.ndarray, start: float, dur: float, sfreq: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_samp = int(start * sfreq)
    end_samp = int((start + dur) * sfreq)
    max_samp = data_full.shape[1]
    if start_samp >= max_samp: raise ValueError(f"Start time {start}s is beyond data duration.")
    end_samp = min(end_samp, max_samp)
    data_ts = data_full[:, start_samp:end_samp] * 1e6
    t = np.linspace(0, (end_samp - start_samp)/sfreq, data_ts.shape[1])
    return data_ts, t, data_full[:, start_samp:end_samp]

def _compute_psd(data: np.ndarray, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
    psds, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=sfreq, fmin=1, fmax=40, 
        n_fft=min(2048, data.shape[1]), average='mean', verbose=False
    )
    return np.mean(psds, axis=0), freqs

def plot_comparison(
    eeg_input_1: mne.io.Raw, eeg_input_2: mne.io.Raw, ch_names: List[str], 
    start_1: float, dur_1: float, start_2: float, dur_2: float, 
    subject_id: str, filename: str = "comparison.png",
    output_dir: str = "./lecture_plots_alpha_ratio",
    freq_band: List[float] = [8, 13], band_color: str = 'yellow', band_alpha: float = 0.15,
    save_plot: bool = True, label_1: str = "Condition 1", label_2: str = "Condition 2"
) -> None:
    sfreq = eeg_input_1.info['sfreq']
    common_chs = [ch for ch in ch_names if ch in eeg_input_1.ch_names and ch in eeg_input_2.ch_names]
    if not common_chs: raise ValueError("No common channels found.")

    data_1_full = eeg_input_1.get_data(picks=common_chs)
    data_2_full = eeg_input_2.get_data(picks=common_chs)
    
    data_1_ts, t_1, data_1_psd_input = _extract_snippet(data_1_full, start_1, dur_1, sfreq)
    data_2_ts, t_2, data_2_psd_input = _extract_snippet(data_2_full, start_2, dur_2, sfreq)
    
    mean_psd_1, freqs = _compute_psd(data_1_psd_input, sfreq)
    mean_psd_2, _ = _compute_psd(data_2_psd_input, sfreq)

    mean_signal_1_full = np.mean(data_1_full, axis=0) * 1e6
    t_1_full = np.arange(len(mean_signal_1_full)) / sfreq
    mean_signal_2_full = np.mean(data_2_full, axis=0) * 1e6
    t_2_full = np.arange(len(mean_signal_2_full)) / sfreq

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3, height_ratios=[0.8, 1.5, 1.2], width_ratios=[1, 1])

    for ax, t, sig, start, dur, color, label in [
        (fig.add_subplot(gs[0, 0]), t_1_full, mean_signal_1_full, start_1, dur_1, COLOR_1, label_1),
        (fig.add_subplot(gs[0, 1]), t_2_full, mean_signal_2_full, start_2, dur_2, COLOR_2, label_2)
    ]:
        ax.plot(t, sig, color=color, linewidth=0.5, alpha=0.7)
        ax.set_title(f"{label}: Full Recording", fontsize=10, weight='bold', color=color)
        ax.set_ylabel('Avg Amp (uV)', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', labelbottom=False)
        rect = patches.Rectangle((start, min(sig)), dur, max(sig) - min(sig), linewidth=1, edgecolor=color, facecolor=color, alpha=HIGHLIGHT_ALPHA)
        ax.add_patch(rect)
        ax.set_xlim(0, t[-1])

    for gs_col, t, data_ts, start, dur, color, label in [
        (gs[1, 0], t_1, data_1_ts, start_1, dur_1, COLOR_1, label_1),
        (gs[1, 1], t_2, data_2_ts, start_2, dur_2, COLOR_2, label_2)
    ]:
        gs_ts = gs_col.subgridspec(len(common_chs), 1, hspace=0.05)
        axes = []
        for i, ch in enumerate(common_chs):
            ax = fig.add_subplot(gs_ts[i, 0], sharex=axes[0] if i > 0 else None)
            axes.append(ax)
            ax.plot(t, data_ts[i, :], color=color, linewidth=0.8)
            ax.set_ylabel(f'{ch}', fontsize=7, rotation=0, ha='right', va='center', labelpad=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            if i < len(common_chs) - 1: ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axes[0].set_title(f"Selected Window ({start}s - {start + dur}s)", fontsize=12, weight='bold', color=color)
        axes[-1].set_xlabel('Time (s)', fontsize=10)

    ax_psd = fig.add_subplot(gs[2, :])
    ax_psd.plot(freqs, 10 * np.log10(mean_psd_1), color=COLOR_1, linewidth=2.5, label=label_1)
    ax_psd.plot(freqs, 10 * np.log10(mean_psd_2), color=COLOR_2, linewidth=2.5, label=label_2, linestyle='--')
    ax_psd.set_xlim(1, 40)
    ax_psd.set_xlabel('Frequency (Hz)', fontsize=13)
    ax_psd.set_ylabel('Power Spectral Density (dB/Hz)', fontsize=13)
    ax_psd.set_title(f"Mean PSD: Highlighting {freq_band[0]}-{freq_band[1]} Hz", fontsize=14, weight='bold')
    ax_psd.grid(True, alpha=0.3, linestyle='--')
    ax_psd.legend(loc='upper right', fontsize=11)
    ax_psd.axvspan(freq_band[0], freq_band[1], color=band_color, alpha=band_alpha, label=f'{freq_band[0]}-{freq_band[1]} Hz Band')

    plt.suptitle(f"Comparison Analysis: Subject {subject_id}", fontsize=16, weight='bold', y=0.98)
    
    if save_plot:
        out_path = Path(output_dir)
        out_path.mkdir(exist_ok=True)
        save_file = out_path / filename
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        logger.info("Plot saved: %s", save_file)
