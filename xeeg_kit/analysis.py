# xeeg_kit/analysis.py


import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import matplotlib.patches as patches
from .bel_280 import parse_gpsc, create_montage_from_gpsc

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level('ERROR')

# ============================================================================
# MODULE 1: ANATOMICAL MAPPING
# ============================================================================

def generate_bel_channel_map(gpsc_file, output_csv="bel_280_channel_map.csv"):
    """
    Parses GPSC file, assigns anatomical regions, saves to CSV, and returns DataFrame.
    If the CSV already exists, it loads it instead of regenerating.
    """
    if Path(output_csv).exists():
        print(f"📂 Loading existing channel map: {output_csv}")
        return pd.read_csv(output_csv)

    print("🗺️ Generating Anatomical Channel Map from GPSC...")
    channels = parse_gpsc(Path(gpsc_file))
    montage = create_montage_from_gpsc(channels)
    ch_pos = montage.get_positions()['ch_pos']
    
    data = []
    for name, pos in ch_pos.items():
        if pos is not None and not np.any(np.isnan(pos)):
            data.append({'name': name, 'x': pos[0], 'y': pos[1], 'z': pos[2]})
            
    df = pd.DataFrame(data)
    if df.empty: raise ValueError("No valid positions found.")

    # --- Anatomical Thresholds ---
    orbital_thresh_y, orbital_thresh_z = 0.065, 0.015
    pfc_z_thresh, inferior_frontal_z_thresh = 0.015, -0.005
    y_parietal_thresh, y_occipital_thresh = -0.035, -0.06
    x_lateral_thresh, z_cortex_min = 0.06, -0.03
    y_frontal_thresh = 0.04

    def get_region(row):
        y, x_abs, z = row['y'], abs(row['x']), row['z']
        if y > orbital_thresh_y and z < orbital_thresh_z: return "Orbital/Eye"
        if y > y_frontal_thresh:
            if z >= pfc_z_thresh: return "Prefrontal" if x_abs < x_lateral_thresh else "Fronto-Temporal"
            elif z < inferior_frontal_z_thresh: return "Inferior-Frontal/Jaw"
            else: return "Prefrontal" if x_abs < x_lateral_thresh else "Fronto-Temporal"
        elif y > y_parietal_thresh: 
            return "Lateral-Inferior/Jaw" if z < z_cortex_min else ("Central" if x_abs < x_lateral_thresh else "Temporal")
        elif y > y_occipital_thresh: 
            if z < z_cortex_min: return "Cerebellar/Neck"
            if x_abs > x_lateral_thresh:
                mid_y = (y_parietal_thresh + y_occipital_thresh) / 2
                return "Occipital" if y < mid_y else "Temporal"
            return "Parietal"
        else: 
            return "Cerebellar/Neck" if z < z_cortex_min else "Occipital"

    df['region'] = df.apply(get_region, axis=1)
    df.to_csv(output_csv, index=False)
    print(f"✅ Channel map saved to {output_csv}")
    return df

# ============================================================================
# MODULE 2: CHANNEL SELECTION
# ============================================================================

def select_channels(raw_data, map_df, method="region", region_name=None, manual_list=None):
    """
    Selects channels based on Region Name or Manual List.
    Ensures selected channels exist in the provided raw_data.
    """
    available_chs = set(raw_data.ch_names)
    
    if method == "manual":
        if not manual_list:
            raise ValueError("Manual list is empty.")
        selected = [ch for ch in manual_list if ch in available_chs]
        missing = set(manual_list) - set(selected)
        if missing:
            print(f"⚠️ Warning: Channels not found in data: {missing}")
        
    elif method == "region":
        if not region_name:
            raise ValueError("Region name not specified.")
        region_chs = map_df[map_df['region'] == region_name]['name'].tolist()
        selected = [ch for ch in region_chs if ch in available_chs]
        if not selected:
            raise ValueError(f"No channels found for region '{region_name}' in the provided data.")
    else:
        raise ValueError("Method must be 'region' or 'manual'.")
        
    print(f"🎯 Selected {len(selected)} channels using method '{method}'.")
    return sorted(selected)

# ============================================================================
# MODULE 3: PLOTTING
# ============================================================================
def plot_comparison(eeg_input_1, eeg_input_2, ch_names, 
                    start_1, dur_1, start_2, dur_2, 
                    subject_id, filename="comparison.png",
                    output_dir="./lecture_plots_alpha_ratio",
                    freq_band=[8, 13],       # [low_freq, high_freq] to highlight
                    band_color='yellow',     # Color of the highlight
                    band_alpha=0.15,         # Transparency of the highlight
                    save_plot=True,          # Set to False to only display
                    label_1="Condition 1",   # Label for the first input (e.g., Eyes Closed)
                    label_2="Condition 2"):  # Label for the second input (e.g., Eyes Open)
    """
    High-Quality Comparison Plot with Customizable Frequency Band Highlighting.
    
    Parameters:
    -----------
    eeg_input_1 : mne.io.Raw
        First condition data (e.g., Eyes Closed).
    eeg_input_2 : mne.io.Raw
        Second condition data (e.g., Eyes Open).
    ch_names : list
        List of channel names to plot.
    start_1, dur_1 : float
        Start time and duration for the first input window.
    start_2, dur_2 : float
        Start time and duration for the second input window.
    ...
    """
    import matplotlib.patches as patches
    
    COLOR_1 = '#6A0DAD'  # Deep Purple
    COLOR_2 = '#1F78B4'  # Strong Blue
    HIGHLIGHT_ALPHA = 0.3
    
    sfreq = eeg_input_1.info['sfreq']
    common_chs = [ch for ch in ch_names if ch in eeg_input_1.ch_names and ch in eeg_input_2.ch_names]
    
    if not common_chs:
        raise ValueError("No common channels found.")

    # Get full data for context plotting and slicing
    data_1_full = eeg_input_1.get_data(picks=common_chs)
    data_2_full = eeg_input_2.get_data(picks=common_chs)
    
    # --- Extract Time Series Snippets ---
    def get_snippet(data_full, start, dur):
        start_samp = int(start * sfreq)
        end_samp = int((start + dur) * sfreq)
        max_samp = data_full.shape[1]
        
        if start_samp >= max_samp:
            raise ValueError(f"Start time {start}s is beyond data duration.")
        end_samp = min(end_samp, max_samp)
        
        data_ts = data_full[:, start_samp:end_samp] * 1e6  # uV
        t = np.linspace(0, (end_samp - start_samp)/sfreq, data_ts.shape[1])
        return data_ts, t, data_full[:, start_samp:end_samp]

    data_1_ts, t_1, data_1_psd_input = get_snippet(data_1_full, start_1, dur_1)
    data_2_ts, t_2, data_2_psd_input = get_snippet(data_2_full, start_2, dur_2)
    
    # --- Compute PSD on Snippets ---
    psds_1, freqs = mne.time_frequency.psd_array_welch(
        data_1_psd_input, sfreq=sfreq, fmin=1, fmax=40, 
        n_fft=min(2048, data_1_psd_input.shape[1]), average='mean', verbose=False
    )
    mean_psd_1 = np.mean(psds_1, axis=0)
    
    psds_2, _ = mne.time_frequency.psd_array_welch(
        data_2_psd_input, sfreq=sfreq, fmin=1, fmax=40, 
        n_fft=min(2048, data_2_psd_input.shape[1]), average='mean', verbose=False
    )
    mean_psd_2 = np.mean(psds_2, axis=0)

    # --- Context Data (Full Recording Average) ---
    mean_signal_1_full = np.mean(data_1_full, axis=0) * 1e6
    t_1_full = np.arange(len(mean_signal_1_full)) / sfreq
    
    mean_signal_2_full = np.mean(data_2_full, axis=0) * 1e6
    t_2_full = np.arange(len(mean_signal_2_full)) / sfreq

    # =========================================================================
    # PLOTTING LAYOUT
    # =========================================================================
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3, 
                          height_ratios=[0.8, 1.5, 1.2], width_ratios=[1, 1])

    # --- A. CONTEXT PLOT: INPUT 1 (Top Left) ---
    ax_ctx_1 = fig.add_subplot(gs[0, 0])
    ax_ctx_1.plot(t_1_full, mean_signal_1_full, color=COLOR_1, linewidth=0.5, alpha=0.7)
    ax_ctx_1.set_title(f"{label_1}: Full Recording", fontsize=10, weight='bold', color=COLOR_1)
    ax_ctx_1.set_ylabel('Avg Amp (µV)', fontsize=8)
    ax_ctx_1.grid(True, alpha=0.3, linestyle='--')
    ax_ctx_1.tick_params(axis='x', labelbottom=False)
    
    rect_1 = patches.Rectangle((start_1, min(mean_signal_1_full)), 
                               dur_1, 
                               max(mean_signal_1_full) - min(mean_signal_1_full),
                               linewidth=1, edgecolor=COLOR_1, 
                               facecolor=COLOR_1, alpha=HIGHLIGHT_ALPHA)
    ax_ctx_1.add_patch(rect_1)
    ax_ctx_1.set_xlim(0, t_1_full[-1])

    # --- B. CONTEXT PLOT: INPUT 2 (Top Right) ---
    ax_ctx_2 = fig.add_subplot(gs[0, 1])
    ax_ctx_2.plot(t_2_full, mean_signal_2_full, color=COLOR_2, linewidth=0.5, alpha=0.7)
    ax_ctx_2.set_title(f"{label_2}: Full Recording", fontsize=10, weight='bold', color=COLOR_2)
    ax_ctx_2.set_ylabel('Avg Amp (µV)', fontsize=8)
    ax_ctx_2.grid(True, alpha=0.3, linestyle='--')
    ax_ctx_2.tick_params(axis='x', labelbottom=False)
    
    rect_2 = patches.Rectangle((start_2, min(mean_signal_2_full)), 
                               dur_2, 
                               max(mean_signal_2_full) - min(mean_signal_2_full),
                               linewidth=1, edgecolor=COLOR_2, 
                               facecolor=COLOR_2, alpha=HIGHLIGHT_ALPHA)
    ax_ctx_2.add_patch(rect_2)
    ax_ctx_2.set_xlim(0, t_2_full[-1])

    # --- C. DETAILED TIME SERIES: INPUT 1 (Middle Left) ---
    gs_ts_1 = gs[1, 0].subgridspec(len(common_chs), 1, hspace=0.05)
    axes_ts_1 = []
    for i, ch in enumerate(common_chs):
        if i == 0:
            ax = fig.add_subplot(gs_ts_1[i, 0])
        else:
            ax = fig.add_subplot(gs_ts_1[i, 0], sharex=axes_ts_1[0])
        axes_ts_1.append(ax)
        
        ax.plot(t_1, data_1_ts[i, :], color=COLOR_1, linewidth=0.8)
        ax.set_ylabel(f'{ch}', fontsize=7, rotation=0, ha='right', va='center', labelpad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        if i < len(common_chs) - 1:
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            
    axes_ts_1[0].set_title(f"Selected Window ({start_1}s - {start_1 + dur_1}s)", 
                           fontsize=12, weight='bold', color=COLOR_1)
    axes_ts_1[-1].set_xlabel('Time (s)', fontsize=10)

    # --- D. DETAILED TIME SERIES: INPUT 2 (Middle Right) ---
    gs_ts_2 = gs[1, 1].subgridspec(len(common_chs), 1, hspace=0.05)
    axes_ts_2 = []
    for i, ch in enumerate(common_chs):
        if i == 0:
            ax = fig.add_subplot(gs_ts_2[i, 0])
        else:
            ax = fig.add_subplot(gs_ts_2[i, 0], sharex=axes_ts_2[0])
        axes_ts_2.append(ax)
        
        ax.plot(t_2, data_2_ts[i, :], color=COLOR_2, linewidth=0.8)
        ax.set_ylabel(f'{ch}', fontsize=7, rotation=0, ha='right', va='center', labelpad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        if i < len(common_chs) - 1:
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            
    axes_ts_2[0].set_title(f"Selected Window ({start_2}s - {start_2 + dur_2}s)", 
                           fontsize=12, weight='bold', color=COLOR_2)
    axes_ts_2[-1].set_xlabel('Time (s)', fontsize=10)

    # --- E. PSD: AVERAGED (Bottom Row) ---
    ax_psd = fig.add_subplot(gs[2, :])
    
    ax_psd.plot(freqs, 10 * np.log10(mean_psd_1), color=COLOR_1, linewidth=2.5, 
                label=label_1)
    ax_psd.plot(freqs, 10 * np.log10(mean_psd_2), color=COLOR_2, linewidth=2.5, 
                label=label_2, linestyle='--')
    
    ax_psd.set_xlim(1, 40)
    ax_psd.set_xlabel('Frequency (Hz)', fontsize=13)
    ax_psd.set_ylabel('Power Spectral Density (dB/Hz)', fontsize=13)
    ax_psd.set_title(f"Mean PSD: Highlighting {freq_band[0]}-{freq_band[1]} Hz", fontsize=14, weight='bold')
    ax_psd.grid(True, alpha=0.3, linestyle='--')
    ax_psd.legend(loc='upper right', fontsize=11)
    
    # Dynamic Highlighting
    ax_psd.axvspan(freq_band[0], freq_band[1], color=band_color, alpha=band_alpha, 
                   label=f'{freq_band[0]}-{freq_band[1]} Hz Band')

    plt.suptitle(f"Comparison Analysis: Subject {subject_id}", fontsize=16, weight='bold', y=0.98)
    
    # --- Conditional Saving ---
    if save_plot:
        out_path = Path(output_dir)
        out_path.mkdir(exist_ok=True)
        save_file = out_path / filename
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved: {save_file}")
    
    plt.show()

'''
Example Usage



# 1. Setup Paths
GPSC = "/mnt/movement/users/jaizor/xtra/data/eeg/ghw280_from_egig.gpsc" 

# 2. Generate the Map (This creates the CSV)
map_df = generate_bel_channel_map(GPSC)

# 3. Inspect the CSV to choose your channels
print("\n--- Available Channels by Region ---")
# Show all unique regions
print(map_df['region'].unique())

# Example: Show all Occipital channels to pick from
print("\n--- Occipital Channels ---")
occipital_chs = map_df[map_df['region'] == 'Occipital']['name'].tolist()
print(occipital_chs)

#---------------------------------------------------------------------------



# 1. Setup Data Paths
CLOSED_FILE = "rest_off_sub-01_c_eeg_mkit_cleaned.fif"  
OPEN_FILE = "rest_off_sub-01_o_eeg_mkit_cleaned.fif"    

# 2. Load Your Data
fif_ec = mne.io.read_raw(CLOSED_FILE, preload=True, verbose=False)
fif_eo = mne.io.read_raw(OPEN_FILE, preload=True, verbose=False)


# 3. Select Specific Channels MANUALLY
# REPLACE these with the 4-5 names you found in the CSV
my_manual_channels = ['E104', 'E105', 'E106', 'E107'] 

my_chs = select_channels(
    raw_data=raw_ec_filt, 
    map_df=None, # Not needed for manual selection
    method="manual", 
    manual_list=my_manual_channels
)


# 4. Run the Comparison Plot
plot_comparison(
    eeg_input_1=fif_ec,      # First condition (e.g., Eyes Closed)
    eeg_input_2=fif_eo,      # Second condition (e.g., Eyes Open)
    ch_names=my_chs,
    start_1=1.0,                  # Start time for Input 1
    dur_1=30.0,                   # Duration for Input 1
    start_2=1.0,                  # Start time for Input 2
    dur_2=30.0,                   # Duration for Input 2
    subject_id="01",
    label_1="Eyes Closed",        # Custom label for the first input
    label_2="Eyes Open",          # Custom label for the second input
    freq_band=[8, 13],            # Highlight Alpha band
    band_color='blue',          
    band_alpha=0.15,              
)


'''
