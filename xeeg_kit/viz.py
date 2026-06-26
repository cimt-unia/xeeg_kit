# xeeg_kit/viz.py

"""
Visualization tools for BEL 280 EEG systems.
Includes 3D headset mapping, bad channel visualization, and anatomical summaries.
"""
import mne
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import importlib.resources

from .bel_280 import parse_gpsc, create_montage_from_gpsc

# ============================================================================
# RESOURCE HELPERS
# ============================================================================

def get_resource_path(filename):
    """Safely get the path to a file inside the xeeg_kit/data folder."""
    try:
        return importlib.resources.files('xeeg_kit.data').joinpath(filename)
    except AttributeError:
        import pkg_resources
        return Path(pkg_resources.resource_filename('xeeg_kit', f'data/{filename}'))

def load_default_map_df():
    """Loads the default channel map CSV bundled in the package."""
    csv_path = get_resource_path('bel_280_channel_map.csv')
    return pd.read_csv(str(csv_path))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_bel_rename_map(raw: mne.io.Raw) -> dict:
    """Create rename map for BEL system (numeric '1' -> 'E1') if needed."""
    first_ch = raw.ch_names[0]
    if first_ch.isdigit():
        return {ch: f"E{ch}" for ch in raw.ch_names if ch.isdigit()}
    return {}

def print_anatomical_summary(bad_chs: list, map_df: pd.DataFrame = None):
    """
    Print anatomical region counts and detect clustering of bad channels.
    If map_df is None, loads the default bundled map.
    """
    if map_df is None:
        map_df = load_default_map_df()

    bad_locations = map_df[map_df['name'].isin(bad_chs)]
    
    if bad_locations.empty:
        print("⚠️ No bad channels found in the channel map.")
        return

    print("\n📍 Bad Channels by Anatomical Region:")
    region_counts = bad_locations['region'].value_counts()
    print(region_counts.to_string())
    
    # Highlight clustering
    if not region_counts.empty:
        most_common = region_counts.index[0]
        count = region_counts.iloc[0]
        if count > len(bad_chs) * 0.4:
            print(f"\n⚠️ CLUSTERING DETECTED: {count}/{len(bad_chs)} bads are in '{most_common}'")
            print(f"   Likely cause: Local artifact (muscle, sweat, cap lift).")

# ============================================================================
# CHANNEL MAP GENERATION & VISUALIZATION
# ============================================================================

def create_bel_channel_map(gpsc_file=None, output_csv="bel_280_channel_map.csv"):
    """
    Parses the GPSC file, assigns anatomical regions, and saves to CSV.
    If gpsc_file is None, it uses the default file bundled in the package.
    """
    if gpsc_file is None:
        gpsc_file = get_resource_path('ghw280_from_egig.gpsc')
        print("Using default BEL 280 GPSC file from package...")
    else:
        print("Generating Anatomical Channel Map from custom GPSC...")

    gpsc_path_str = str(gpsc_file)
    channels = parse_gpsc(Path(gpsc_path_str))
    montage = create_montage_from_gpsc(channels)
    ch_pos = montage.get_positions()['ch_pos']
    
    data = []
    for name, pos in ch_pos.items():
        if pos is not None and not np.any(np.isnan(pos)):
            data.append({'name': name, 'x': pos[0], 'y': pos[1], 'z': pos[2]})
            
    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("No valid positions found in GPSC file.")

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
    df = df.sort_values(['region', 'name']).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    
    print(f"✅ Channel map saved to: {output_csv}")
    return df

def plot_headset_3d(map_df=None, output_file="bel_280_headset_layout.html"):
    """
    Renders an interactive 3D scatter plot of the full BEL 280 headset.
    If map_df is None, loads the default bundled map.
    """
    if map_df is None:
        map_df = load_default_map_df()

    color_map = {
        'Prefrontal': '#FF0000', 'Fronto-Temporal': '#FFA500',
        'Temporal': '#FFFF00', 'Central': '#008000',
        'Parietal': '#800080', 'Occipital': '#0000FF',
        'Orbital/Eye': '#FFC0CB', 'Cerebellar/Neck': '#808080',
        'Inferior-Frontal/Jaw': '#8B4513', 'Lateral-Inferior/Jaw': '#808000',
        'Unknown': '#000000'
    }
    
    fig = go.Figure()
    for region in sorted(map_df['region'].unique()):
        subset = map_df[map_df['region'] == region]
        fig.add_trace(go.Scatter3d(
            x=subset['x'], y=subset['y'], z=subset['z'],
            mode='markers+text', name=region,
            marker=dict(size=6, color=color_map.get(region, '#000'), opacity=0.9, line=dict(width=1, color='White')),
            text=subset['name'],
            textfont=dict(size=7, color='black', family="Arial"),
            hovertemplate='<b>%{text}</b><br>Region: %{customdata}<extra></extra>',
            customdata=subset['region']
        ))

    fig.update_layout(
        title='BEL 280 System: Full Headset Layout',
        scene=dict(
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5)),
            annotations=[
                dict(text="FRONTAL", showarrow=False, x=0, y=0.12, z=0.05, font=dict(size=14, color="red")),
                dict(text="OCCIPITAL", showarrow=False, x=0, y=-0.12, z=0.05, font=dict(size=14, color="blue"))
            ]
        ),
        legend_title_text="Brain Region", width=1200, height=900
    )
    
    fig.write_html(output_file)
    print(f"\n✅ Interactive 3D headset plot saved to: {output_file}")
    fig.show()

# ============================================================================
# BAD CHANNEL VISUALIZATION
# ============================================================================

def plot_bad_channels_3d(raw: mne.io.Raw, bad_chs: list, map_df: pd.DataFrame = None, output_file="bad_channels_3d_labeled.html"):
    """
    Interactive 3D scatter plot showing bad channel locations relative to good ones.
    Helps visualize if bad channels are clustered (e.g., due to a cap lift).
    If map_df is None, loads the default bundled map.
    """
    if map_df is None:
        map_df = load_default_map_df()

    montage = raw.get_montage()
    if montage is None: 
        raise ValueError("Raw object must have a montage set.")
        
    ch_pos = montage.get_positions()['ch_pos']
    
    # Build plot data efficiently
    data = []
    for ch_name in raw.ch_names:
        pos = ch_pos.get(ch_name)
        if pos is None or np.any(np.isnan(pos)):
            continue
            
        is_bad = ch_name in bad_chs
        region_row = map_df[map_df['name'] == ch_name]
        region = region_row['region'].values[0] if not region_row.empty else "Unknown"
        
        data.append({
            'name': ch_name,
            'x': pos[0], 'y': pos[1], 'z': pos[2],
            'status': 'BAD' if is_bad else 'GOOD',
            'region': region,
            'point_size': 12 if is_bad else 4,
            'label': ch_name if is_bad else ''
        })
                
    df_plot = pd.DataFrame(data)
    if df_plot.empty:
        print("No channel positions found for plotting.")
        return

    fig = px.scatter_3d(
        df_plot, x='x', y='y', z='z',
        color='status',
        color_discrete_map={'BAD': '#FF0000', 'GOOD': '#4A90E2'},
        size='point_size',
        text='label',              
        hover_data=['name', 'region'],
        title=f"Bad Channel Locations ({len(bad_chs)} detected)",
        opacity=0.9
    )
    
    fig.update_traces(
        textfont=dict(size=10, color='black', family="Arial Black"),
        marker=dict(line=dict(width=1, color='white'))
    )
    
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5)),
            annotations=[
                dict(text="FRONT", showarrow=False, x=0, y=0.12, z=0, font=dict(size=14)),
                dict(text="BACK", showarrow=False, x=0, y=-0.12, z=0, font=dict(size=14))
            ]
        ),
        width=1200, height=900,
        legend_title_text="Channel Status"
    )
    
    fig.write_html(output_file)
    print(f"\n✅ Interactive 3D plot saved to: {output_file}")
    fig.show()

'''
Example Usage:
import mne
from xeeg_kit import detect_bad_channels, plot_bad_channels_3d, print_anatomical_summary

# 1. Load your data (assuming it's already preprocessed/montaged)
raw = mne.io.read_raw("your_cleaned_file.fif", preload=True)

# 2. Detect bad channels
bad_chs = detect_bad_channels(raw, mad_threshold=20.0)

# 3. Print Summary (Checks for clustering automatically)
print_anatomical_summary(bad_chs)

# 4. Visualize in 3D (Uses default map from package)
plot_bad_channels_3d(raw, bad_chs)
'''
