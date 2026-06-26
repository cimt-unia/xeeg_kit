# xeeg_kit/viz.py
"""
Visualization tools for BEL 280 EEG systems.
"""
import mne
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import importlib.resources
import os

# ============================================================================
# RESOURCE HELPERS (The Fix)
# ============================================================================

def get_resource_path(filename):
    """
    Safely get the path to a file inside the xeeg_kit/data folder.
    Works for both editable installs and standard pip installs.
    """
    try:
        # Method 1: Standard Python 3.9+ resource loading
        return importlib.resources.files('xeeg_kit.data').joinpath(filename)
    except Exception:
        try:
            # Method 2: Fallback for older Python or specific environments
            import pkg_resources
            return Path(pkg_resources.resource_filename('xeeg_kit', f'data/{filename}'))
        except Exception:
            # Method 3: Absolute last resort - find file relative to this script
            current_dir = Path(__file__).parent
            return current_dir / 'data' / filename

def load_bel_channel_map():
    """Loads the default channel map CSV bundled in the package."""
    csv_path = get_resource_path('bel_280_channel_map.csv')
    print(f"📂 Loading bundled map from: {csv_path}")
    return pd.read_csv(str(csv_path))

def get_default_gpsc_path():
    """Returns the path to the default GPSC file bundled in the package."""
    return get_resource_path('ghw280_from_egig.gpsc')

# Alias for backward compatibility
generate_bel_channel_map = load_bel_channel_map

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_bel_rename_map(raw: mne.io.Raw) -> dict:
    first_ch = raw.ch_names[0]
    if first_ch.isdigit():
        return {ch: f"E{ch}" for ch in raw.ch_names if ch.isdigit()}
    return {}

def print_anatomical_summary(bad_chs: list, map_df: pd.DataFrame = None):
    if map_df is None:
        map_df = load_bel_channel_map()

    bad_locations = map_df[map_df['name'].isin(bad_chs)]
    if bad_locations.empty:
        print("⚠️ No bad channels found in the channel map.")
        return

    print("\n📍 Bad Channels by Anatomical Region:")
    print(bad_locations['region'].value_counts().to_string())
    
    if not bad_locations['region'].value_counts().empty:
        most_common = bad_locations['region'].value_counts().index[0]
        count = bad_locations['region'].value_counts().iloc[0]
        if count > len(bad_chs) * 0.4:
            print(f"\n⚠️ CLUSTERING DETECTED: {count}/{len(bad_chs)} bads are in '{most_common}'")

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_headset_3d(map_df=None, output_file="bel_280_headset_layout.html"):
    if map_df is None:
        map_df = load_bel_channel_map()

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

def plot_bad_channels_3d(raw: mne.io.Raw, bad_chs: list, map_df: pd.DataFrame = None, output_file="bad_channels_3d_labeled.html"):
    if map_df is None:
        map_df = load_bel_channel_map()

    montage = raw.get_montage()
    if montage is None: 
        raise ValueError("Raw object must have a montage set.")
        
    ch_pos = montage.get_positions()['ch_pos']
    data = []
    for ch_name in raw.ch_names:
        pos = ch_pos.get(ch_name)
        if pos is None or np.any(np.isnan(pos)): continue
            
        is_bad = ch_name in bad_chs
        region_row = map_df[map_df['name'] == ch_name]
        region = region_row['region'].values[0] if not region_row.empty else "Unknown"
        
        data.append({
            'name': ch_name, 'x': pos[0], 'y': pos[1], 'z': pos[2],
            'status': 'BAD' if is_bad else 'GOOD', 'region': region,
            'point_size': 12 if is_bad else 4, 'label': ch_name if is_bad else ''
        })
                
    df_plot = pd.DataFrame(data)
    if df_plot.empty: return

    fig = px.scatter_3d(
        df_plot, x='x', y='y', z='z', color='status',
        color_discrete_map={'BAD': '#FF0000', 'GOOD': '#4A90E2'},
        size='point_size', text='label',              
        hover_data=['name', 'region'], title=f"Bad Channel Locations ({len(bad_chs)} detected)", opacity=0.9
    )
    
    fig.update_traces(textfont=dict(size=10, color='black', family="Arial Black"), marker=dict(line=dict(width=1, color='white')))
    fig.update_layout(
        scene=dict(aspectmode='data', camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5))),
        width=1200, height=900, legend_title_text="Channel Status"
    )
    
    fig.write_html(output_file)
    print(f"\n✅ Interactive 3D plot saved to: {output_file}")
    fig.show()
