"""
Streamlit app: Shotmap + xG visualizations
Filename: streamlit_shotmap_app.py

How to run:
    pip install -r requirements.txt
    streamlit run streamlit_shotmap_app.py

Features:
- Accept CSV upload (or use example provided file)
- Detect or compute xG (uses 'xG' column if present, otherwise computes an approximate xG from distance & angle)
- Shot map using mplsoccer
- Download Plot PNG
"""

import io
import math
import base64
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from mplsoccer import Pitch

st.set_page_config(layout="wide", page_title="Shotmap + xG dashboard")

# ----------------------------- Helpers -----------------------------

def detect_coord_scale(df):
    """Detect whether coordinates are in percentages (0-100) or normalized (0-1) or 0-120/0-80.
    Returns a function to convert columns x,y -> (x_plot, y_plot) in meters/pitch units suitable for mplsoccer default 120x80 pitch.
    """
    x_max = df['x'].max()
    y_max = df['y'].max()
    # normalized 0-1
    if x_max <= 1.01 and y_max <= 1.01:
        return lambda x, y: (x * 120, y * 80)
    # percent 0-100
    if x_max <= 100.1 and y_max <= 100.1:
        return lambda x, y: (x * 1.2, y * 0.8)
    # already 0-120 / 0-80
    return lambda x, y: (x, y)


def compute_distance_angle_to_goal(x_plot, y_plot, goal_x=120, goal_y=40):
    dx = goal_x - x_plot
    dy = goal_y - y_plot
    dist = np.sqrt(dx ** 2 + dy ** 2)
    # compute shot angle (simple): angle at goal mouth seen from shot location
    # formula: angle = arctan( (post_y - shot_y) / (goal_x - shot_x) ) - arctan( (other_post_y - shot_y) / (goal_x - shot_x) )
    post_y1 = 36.5  # left post
    post_y2 = 43.5  # right post
    # avoid divide by zero
    denom1 = (goal_x - x_plot) if (goal_x - x_plot) != 0 else 1e-6
    a1 = math.atan2(post_y1 - y_plot, denom1)
    a2 = math.atan2(post_y2 - y_plot, denom1)
    angle = abs(a2 - a1)
    return dist, angle


def approximate_xg_from_distance_angle(dist, angle):
    """A simple parametric xG approximation using distance (meters) and angle (radians).
    This is a heuristic fallback when the dataset does not include an xG model.
    It is intentionally simple and should be replaced with a trained model for production.
    """
    # logistic on distance
    # tune parameters so typical near-range ~0.4-0.3, long range ~0.01
    base = 1 / (1 + np.exp((dist - 22) / 6.0))
    # angle multiplier (0..1) where wider angle increases probability
    angle_factor = angle / math.pi  # normalize
    xg = base * (0.4 + 0.8 * angle_factor)  # scale
    return float(np.clip(xg, 0.0001, 0.9999))


def prepare_data(df):
    # --- Normalize column names (handle case differences like X vs x) ---
    df = df.rename(columns={c.strip(): c.strip() for c in df.columns})
    lower_map = {c.lower(): c for c in df.columns}

    # --- Ensure coordinates exist ---
    if 'x' not in lower_map or 'y' not in lower_map:
        raise ValueError("Input CSV must contain shot coordinate columns ('x'/'y' or 'X'/'Y').")

    x_col = lower_map['x']
    y_col = lower_map['y']

    df = df.copy()

    # --- Ensure 'minute' column ---
    if 'minute' in lower_map:
        df['minute'] = df[lower_map['minute']]
    else:
        for col in ['time', 'minute_played', 'min']:
            if col in lower_map:
                df['minute'] = df[lower_map[col]]
                break
        else:
            df['minute'] = np.arange(1, len(df) + 1)

    # --- Detect coordinate scale and convert ---
    conv = detect_coord_scale(df.rename(columns={x_col: 'x', y_col: 'y'}))
    x_plot, y_plot = zip(*[conv(x, y) for x, y in zip(df[x_col].astype(float), df[y_col].astype(float))])
    df['x_plot'] = x_plot
    df['y_plot'] = y_plot

    # --- Handle xG column (case-insensitive) ---
    if 'xg' in lower_map:
        df['xG'] = df[lower_map['xg']].astype(float)
    else:
        # Try computing approximate xG if not found
        if 'distance' in lower_map and 'angle' in lower_map:
            df['xG'] = df.apply(
                lambda r: approximate_xg_from_distance_angle(
                    float(r[lower_map['distance']]), float(r[lower_map['angle']])
                ),
                axis=1,
            )
        else:
            da = [compute_distance_angle_to_goal(x, y) for x, y in zip(df['x_plot'], df['y_plot'])]
            distances = [d for d, a in da]
            angles = [a for d, a in da]
            df['distance'] = distances
            df['angle'] = angles
            df['xG'] = [approximate_xg_from_distance_angle(d, a) for d, a in zip(distances, angles)]

    # --- Handle result/goal columns ---
    if 'result' in lower_map:
        df['result'] = df[lower_map['result']]
    elif 'goal' in lower_map:
        df['result'] = df[lower_map['goal']].apply(lambda v: 'Goal' if int(v) == 1 else 'No Goal')
    else:
        df['result'] = 'Shot'

    # --- Final cleanup ---
    df['minute'] = pd.to_numeric(df['minute'], errors='coerce').fillna(0).astype(int)

    return df



# ----------------------------- UI -----------------------------

st.title("⚽ Shotmap & xG Dashboard")

with st.sidebar:
    st.header("Upload / Options")
    uploaded = st.file_uploader("Upload an Understat-style CSV file", type=["csv"], accept_multiple_files=False)
    use_example = st.checkbox("Use example: erling_haaland_2022_understat.csv", value=False)
    min_xg_filter = st.slider("Min xG to show (filter shots)", 0.0, 1.0, 0.0, 0.01)
    cmap_choice = st.selectbox("Color by:", options=["xG", "result"], index=0)
    show_values = st.checkbox("Show summary values (goals, shots, total xG)", value=True)
    download_plots = st.checkbox("Enable plot downloads", value=True)

# load data
if uploaded is not None:
    raw_bytes = uploaded.read()
    df_input = pd.read_csv(io.BytesIO(raw_bytes))
    st.success(f"Loaded {uploaded.name} — {len(df_input)} rows")
elif use_example:
    try:
        df_input = pd.read_csv('/mnt/data/erling_haaland_2022_understat.csv')
        st.success("Loaded example dataset (erling_haaland_2022_understat.csv)")
    except Exception as e:
        st.error("Example file not found on server. Please upload your CSV.")
        st.stop()
else:
    st.info("Upload a CSV to begin, or tick the Example checkbox to load the provided Haaland file.")
    st.stop()

# process
try:
    df = prepare_data(df_input)
except Exception as e:
    st.error(f"Error preparing data: {e}")
    st.stop()

# optional player filter
player_col = None
for cand in ['player', 'Player', 'player_name']:
    if cand in df.columns:
        player_col = cand
        break

selected_player = None
if player_col:
    players = ['All'] + sorted(df[player_col].unique().tolist())
    selected_player = st.sidebar.selectbox("Filter player", players, index=0)
    if selected_player != 'All':
        df = df[df[player_col] == selected_player]

# apply xG filter
df = df[df['xG'] >= min_xg_filter]

# ----------------------------- Summary -----------------------------

if show_values:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Shots", int(len(df)))
    with col2:
        goals = int(df['result'].apply(lambda r: 1 if str(r).lower().startswith('goal') else 0).sum())
        st.metric("Goals (detected)", goals)
    with col3:
        total_xg = df['xG'].sum()
        st.metric("Total xG", f"{total_xg:.2f}")
    with col4:
        conv_rate = f"{(goals / len(df) * 100) if len(df)>0 else 0:.1f}%"
        st.metric("Conversion(xG/shot)", conv_rate)

st.markdown("---")

# ----------------------------- Shotmap -----------------------------

st.header("Shot map")
fig_pitch, ax_pitch = plt.subplots(figsize=(8, 6))
pitch = Pitch(
    pitch_type='statsbomb',
    half=True,                # only show attacking half
    pitch_color='#3f995b',    # green background
    line_color='white'
)
pitch.draw(ax=ax_pitch)

# choose color/size
if cmap_choice == 'xG':
    sc = ax_pitch.scatter(df['x_plot'], df['y_plot'], s=np.clip(df['xG'] * 4000, 20, 400),
                          c=df['xG'], cmap='viridis', alpha=0.4, edgecolors='k', linewidths=0.2)
    cbar = fig_pitch.colorbar(sc, ax=ax_pitch)
    cbar.set_label('xG')
else:
    colors = {'Goal': 'red', 'No Goal': 'blue', 'Shot': 'orange'}
    col_list = [colors.get(r, 'gray') for r in df['result']]
    sc = ax_pitch.scatter(df['x_plot'], df['y_plot'], s=60, c=col_list, alpha=0.9, edgecolors='k', linewidths=0.3)

ax_pitch.set_title('Shot map')
st.pyplot(fig_pitch)

# download plot
if download_plots and st.button('Download shot map PNG'):
    buf = io.BytesIO()
    fig_pitch.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    st.download_button(label='Download shotmap.png', data=buf, file_name='shotmap.png', mime='image/png')

# ----------------------------- Table & CSV download -----------------------------

st.header('Processed data')
st.dataframe(df.head(200))

csv_bytes = df.to_csv(index=False).encode('utf-8')
st.download_button('Download processed CSV', csv_bytes, file_name='processed_shots.csv', mime='text/csv')

# also allow user to download an aggregated summary
summary = {
    'shots': len(df),
    'goals': int(df['result'].apply(lambda r: 1 if str(r).lower().startswith('goal') else 0).sum()),
    'total_xG': float(df['xG'].sum()),
}
summary_df = pd.DataFrame([summary])
summary_csv = summary_df.to_csv(index=False).encode('utf-8')
st.download_button('Download summary CSV', summary_csv, file_name='summary.csv', mime='text/csv')

st.markdown("---")
st.markdown("*Notes:* If your CSV already contains an `xG` column it will be used directly. If not, the app computes an approximate xG from shot coordinates (this is a heuristic fallback and not a trained model).")

# footer
st.write('Simran Raina - xG Streamlit App')
