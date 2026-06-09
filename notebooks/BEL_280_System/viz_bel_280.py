# viz_bel_280.py


import mne
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from xeeg_kit.bel_280 import BELStandardizer
from xeeg_kit.utils import detect_bad_channels


# Helper Functions

def get_bel_rename_map(raw: mne.io.Raw) -> dict:
    """Create rename map for BEL system (numeric '1' -> 'E1') if needed."""
    first_ch = raw.ch_names[0]
    if first_ch.isdigit():
        return {ch: f"E{ch}" for ch in raw.ch_names if ch.isdigit()}
    return {}

def print_anatomical_summary(bad_chs: list, map_df: pd.DataFrame):
    """Print anatomical region counts and detect clustering."""
    bad_locations = map_df[map_df['name'].isin(bad_chs)]
    
    if bad_locations.empty:
        print("⚠️ No bad channels found in the channel map.")
        return

    print("\n📍 Bad Channels by Anatomical Region:")
    region_counts = bad_locations['region'].value_counts()
    print(region_counts.to_string())
    

# Visualize Bad Channels

def plot_bad_channels_3d(raw: mne.io.Raw, bad_chs: list, map_df: pd.DataFrame):
    """Interactive 3D scatter plot showing bad channel locations."""
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
    
    fig.write_html("bad_channels_3d_labeled.html")
    print("\n✅ Interactive 3D plot saved to: bad_channels_3d_labeled.html")
    fig.show()



# Visualize 3D Channel Map

def visualize_channel_map_3d(csv_path="bel_280_channel_map.csv"):
    df = pd.read_csv(csv_path)
    
    color_map = {
        'Prefrontal': '#FF0000',
        'Supraorbital': '#FF69B4',
        'Lateral-Brow': '#FF1493',
        'Fronto-Temporal': '#FFA500',
        'Temporal': '#FFFF00',
        'Central': '#008000',
        'Parietal': '#800080',
        'Occipital': '#0000FF',
        'Orbital/Eye': '#FFC0CB',
        'Cerebellar/Neck': '#808080',
        'Inferior-Frontal/Jaw': '#8B4513',
        'Lateral-Inferior/Jaw': '#808000'
    }

    fig = px.scatter_3d(
        df, x='x', y='y', z='z', color='region',
        color_discrete_map=color_map,
        title="BEL 280 Final Precise Map",
        labels={'x': 'X (Left-Right)', 'y': 'Y (Nasion-Inion)', 'z': 'Z (Inf-Sup)'},
        size_max=10, opacity=0.9
    )

    fig.update_layout(
        scene=dict(
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5)),
            annotations=[
                dict(text="FRONT", showarrow=False, x=0, y=0.12, z=0, font=dict(size=16, color="black")),
                dict(text="BACK", showarrow=False, x=0, y=-0.12, z=0, font=dict(size=16, color="black")),
                dict(text="EYES/LOW", showarrow=False, x=0.08, y=0.08, z=-0.02, font=dict(size=14, color="hotpink"))
            ]
        ),
        legend_title_text="Anatomical Region",
        width=1200, height=900
    )

    fig.write_html("bel_280_interactive_final.html")
    print("✅ Interactive map saved to: bel_280_interactive_final.html")
    fig.show()

