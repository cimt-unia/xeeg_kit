# xeeg_kit/viz.py

# Visualization and anatomical mapping tools for BEL 280 EEG systems.
import logging
from pathlib import Path
from typing import Optional, List, Dict
import mne
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import importlib.resources

logger = logging.getLogger(__name__)

REGION_COLORS: Dict[str, str] = {
    'Prefrontal': '#FF0000', 'Fronto-Temporal': '#FFA500',
    'Temporal': '#FFFF00', 'Central': '#008000',
    'Parietal': '#800080', 'Occipital': '#0000FF',
    'Orbital/Eye': '#FFC0CB', 'Cerebellar/Neck': '#808080',
    'Inferior-Frontal/Jaw': '#8B4513', 'Lateral-Inferior/Jaw': '#808000',
    'Unknown': '#000000'
}
CAMERA_EYE = dict(x=1.5, y=-1.5, z=0.5)
MARKER_SIZE_GOOD = 4
MARKER_SIZE_BAD = 12

def get_resource_path(filename: str) -> Path:
    try:
        return importlib.resources.files('xeeg_kit.data').joinpath(filename)
    except Exception:
        try:
            import pkg_resources
            return Path(pkg_resources.resource_filename('xeeg_kit', f'data/{filename}'))
        except Exception:
            return Path(__file__).parent / 'data' / filename

def load_bel_channel_map() -> pd.DataFrame:
    csv_path = get_resource_path('bel_280_channel_map.csv')
    logger.debug("Loading bundled channel map from: %s", csv_path)
    return pd.read_csv(str(csv_path))

def get_default_gpsc_path() -> Path:
    return get_resource_path('ghw280_from_egig.gpsc')

generate_bel_channel_map = load_bel_channel_map

def get_bel_rename_map(raw: mne.io.Raw) -> Dict[str, str]:
    first_ch = raw.ch_names[0]
    if first_ch.isdigit():
        return {ch: f"E{ch}" for ch in raw.ch_names if ch.isdigit()}
    return {}

def get_anatomical_summary(bad_chs: List[str], map_df: Optional[pd.DataFrame] = None) -> str:
    if map_df is None:
        map_df = load_bel_channel_map()

    bad_locations = map_df[map_df['name'].isin(bad_chs)]
    if bad_locations.empty:
        return "No bad channels found in the channel map."

    counts = bad_locations['region'].value_counts()
    lines = ["Bad Channels by Anatomical Region:"]
    for region, count in counts.items():
        lines.append(f"  {region}: {count}")

    if not counts.empty:
        most_common = counts.index[0]
        top_count = counts.iloc[0]
        if top_count > len(bad_chs) * 0.4:
            lines.append(f"CLUSTERING DETECTED: {top_count}/{len(bad_chs)} bads are in '{most_common}'")

    return "\n".join(lines)

def print_anatomical_summary(bad_chs: List[str], map_df: Optional[pd.DataFrame] = None) -> None:
    summary = get_anatomical_summary(bad_chs, map_df)
    logger.info(summary)

def plot_headset_3d(
    map_df: Optional[pd.DataFrame] = None, 
    output_file: str = "bel_280_headset_layout.html"
) -> None:
    if map_df is None:
        map_df = load_bel_channel_map()

    fig = go.Figure()
    for region in sorted(map_df['region'].unique()):
        subset = map_df[map_df['region'] == region]
        fig.add_trace(go.Scatter3d(
            x=subset['x'], y=subset['y'], z=subset['z'],
            mode='markers+text', name=region,
            marker=dict(size=6, color=REGION_COLORS.get(region, '#000'), opacity=0.9, line=dict(width=1, color='White')),
            text=subset['name'],
            textfont=dict(size=7, color='black', family="Arial"),
            hovertemplate='<b>%{text}</b><br>Region: %{customdata}<extra></extra>',
            customdata=subset['region']
        ))

    fig.update_layout(
        title='BEL 280 System: Full Headset Layout',
        scene=dict(
            aspectmode='data',
            camera=dict(eye=CAMERA_EYE),
            annotations=[
                dict(text="FRONTAL", showarrow=False, x=0, y=0.12, z=0.05, font=dict(size=14, color="red")),
                dict(text="OCCIPITAL", showarrow=False, x=0, y=-0.12, z=0.05, font=dict(size=14, color="blue"))
            ]
        ),
        legend_title_text="Brain Region", width=1200, height=900
    )
    
    fig.write_html(output_file)
    logger.info("Interactive 3D headset plot saved to: %s", output_file)

def plot_bad_channels_3d(
    raw: mne.io.Raw, 
    bad_chs: List[str], 
    map_df: Optional[pd.DataFrame] = None, 
    output_file: str = "bad_channels_3d_labeled.html"
) -> None:
    if map_df is None:
        map_df = load_bel_channel_map()

    montage = raw.get_montage()
    if montage is None: 
        raise ValueError("Raw object must have a montage set.")
        
    ch_pos = montage.get_positions()['ch_pos']
    data = []
    for ch_name in raw.ch_names:
        pos = ch_pos.get(ch_name)
        if pos is None or np.any(np.isnan(pos)): 
            continue
            
        is_bad = ch_name in bad_chs
        region_row = map_df[map_df['name'] == ch_name]
        region = region_row['region'].values[0] if not region_row.empty else "Unknown"
        
        data.append({
            'name': ch_name, 'x': pos[0], 'y': pos[1], 'z': pos[2],
            'status': 'BAD' if is_bad else 'GOOD', 'region': region,
            'point_size': MARKER_SIZE_BAD if is_bad else MARKER_SIZE_GOOD, 
            'label': ch_name if is_bad else ''
        })
                
    df_plot = pd.DataFrame(data)
    if df_plot.empty: 
        return

    fig = px.scatter_3d(
        df_plot, x='x', y='y', z='z', color='status',
        color_discrete_map={'BAD': '#FF0000', 'GOOD': '#4A90E2'},
        size='point_size', text='label',              
        hover_data=['name', 'region'], title=f"Bad Channel Locations ({len(bad_chs)} detected)", opacity=0.9
    )
    
    fig.update_traces(textfont=dict(size=10, color='black', family="Arial Black"), marker=dict(line=dict(width=1, color='white')))
    fig.update_layout(
        scene=dict(aspectmode='data', camera=dict(eye=CAMERA_EYE)),
        width=1200, height=900, legend_title_text="Channel Status"
    )
    
    fig.write_html(output_file)
    logger.info("Interactive 3D plot saved to: %s", output_file)
