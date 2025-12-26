import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from scipy.signal import medfilt
from scipy.interpolate import griddata
import warnings
import os
import time
import glob
warnings.filterwarnings('ignore')

# Try to import tensorflow for CNN (optional)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="EEG Emotion Detection - Central Command Center",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .happy { background-color: #ffeb3b; color: #000; }
    .neutral { background-color: #9e9e9e; color: #fff; }
    .sad { background-color: #2196f3; color: #fff; }
    .anxiety { background-color: #f44336; color: #fff; }
    .status-box {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
EMOTION_LABELS = {0: 'Happy', 1: 'Neutral', 2: 'Sad', 3: 'Anxiety'}
EMOTION_COLORS = {
    'Happy': '#FFD700',
    'Neutral': '#808080',
    'Sad': '#4169E1',
    'Anxiety': '#FF4500'
}

TARGET_SENSORS = [
    'Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5',
    'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1',
    'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6',
    'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2'
]

# Standard 10–20 system electrode positions (2D projection) matching the 32 TARGET_SENSORS
ELECTRODE_POSITIONS = {
    'Fp1': (-0.3, 0.8), 'Fp2': (0.3, 0.8),
    'F7': (-0.6, 0.5), 'F3': (-0.3, 0.5), 'Fz': (0, 0.5),
    'F4': (0.3, 0.5), 'F8': (0.6, 0.5),
    'FT9': (-0.8, 0.2), 'FT10': (0.8, 0.2),
    'FC5': (-0.5, 0.25), 'FC1': (-0.15, 0.25),
    'FC2': (0.15, 0.25), 'FC6': (0.5, 0.25),
    'T7': (-0.7, 0), 'C3': (-0.3, 0), 'Cz': (0, 0),
    'C4': (0.3, 0), 'T8': (0.7, 0),
    'CP5': (-0.5, -0.25), 'CP1': (-0.15, -0.25),
    'CP2': (0.15, -0.25), 'CP6': (0.5, -0.25),
    'P7': (-0.6, -0.5), 'P3': (-0.3, -0.5), 'Pz': (0, -0.5),
    'P4': (0.3, -0.5), 'P8': (0.6, -0.5),
    'PO9': (-0.45, -0.7), 'PO10': (0.45, -0.7),
    'O1': (-0.3, -0.8), 'Oz': (0, -0.8), 'O2': (0.3, -0.8)
}

ROLLING_WINDOW_SIZE = 128
CONTEXT_WINDOW = 5  # For SVM context stacking
WINDOW_SIZE = 128
STEP_SIZE = 64    

# Dataset info (update these based on your actual data)
DATASET_INFO = {
    'total_files': 32,
    'train_files': 24,
    'test_files': 8,
    'emotions': ['Happy', 'Neutral', 'Sad', 'Anxiety'],
    'class_distribution': {'Happy': 8, 'Neutral': 8, 'Sad': 8, 'Anxiety': 8}  # Update with actual
}

def focal_loss(alpha=None, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_onehot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true_onehot * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma)
        if alpha is not None:
            weight *= tf.constant(alpha, dtype=tf.float32)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)
    return loss

@st.cache_data
def load_svm_model():
    """Load the trained SVM model"""
    try:
        model_path = os.path.join('models', 'eeg_emotion_svm_model.pkl')
        if not os.path.exists(model_path):
            model_path = os.path.join('..', 'models', 'eeg_emotion_svm_model.pkl')
        
        artifact = joblib.load(model_path)
        # Handle both direct model and artifact dict
        if isinstance(artifact, dict):
            return artifact.get('model'), artifact.get('scaler'), artifact
        return artifact, None, {}
    except Exception as e:
        return None, None, {}

@st.cache_data
def load_cnn_model():
    """Load the trained CNN model"""
    if not TF_AVAILABLE:
        return None, {}
    
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # Go up from app/ to project root
        
        # Try different possible paths
        possible_paths = [
            os.path.join(project_root, 'models', 'eeg_emotion_cnn_model.keras'),  # From app/ -> ../models/
            os.path.join('models', 'eeg_emotion_cnn_model.keras'),  # Relative to current working dir
            os.path.join('..', 'models', 'eeg_emotion_cnn_model.keras'),  # Relative from app/
            os.path.join(script_dir, '..', 'models', 'eeg_emotion_cnn_model.keras'),  # From app/ using script_dir
        ]
        
        # Remove duplicates and normalize paths
        possible_paths = [os.path.normpath(p) for p in possible_paths]
        possible_paths = list(dict.fromkeys(possible_paths))  # Remove duplicates while preserving order
        
        last_error = None
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    model = tf.keras.models.load_model(
                        path,
                        custom_objects={"loss": focal_loss(alpha=[1.0,1.0,1.3,1.0], gamma=2.0)},
                        compile=False  # Don't compile immediately, faster loading
                    )
                    # Verify model loaded correctly
                    if model is not None:
                        return model, {}
                except Exception as load_error:
                    # Store the error but try next path
                    last_error = load_error
                    continue
        
        # If we get here, model couldn't be loaded
        return None, {}
    except Exception as e:
        return None, {}


def create_context_windows(data, window_size=5):
    """Create context windows for SVM (sliding window stacking)"""
    if len(data) < window_size:
        return np.array([])
    pad_size = window_size // 2
    padded_data = np.pad(data, ((pad_size, pad_size), (0, 0)), mode='edge')
    windows = []
    for i in range(pad_size, len(padded_data) - pad_size):
        window = padded_data[i-pad_size : i+pad_size+1].flatten()
        windows.append(window)
    return np.array(windows)

def add_ratios_smart(df):
    """Add ratio features similar to preprocessing"""
    cols = df.columns
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    
    alpha_cols = [c for c in df.columns if "Alpha" in c]
    for alpha_col in alpha_cols:
        beta_col = alpha_col.replace("Alpha", "Beta")
        theta_col = alpha_col.replace("Alpha", "Theta")
        
        if '.' in alpha_col:
            sensor = alpha_col.split('.')[1]
            if beta_col in df.columns:
                df[f"Ratio.{sensor}.BetaAlpha"] = df[beta_col] / (df[alpha_col] + 1e-6)
            if theta_col in df.columns and beta_col in df.columns:
                df[f"Ratio.{sensor}.ThetaBeta"] = df[theta_col] / (df[beta_col] + 1e-6)
    
    # Global Frontal-Occipital Ratio
    frontal_beta = [c for c in df.columns if 'Beta' in c and any(x in c for x in ['F3','F4','Fp1','Fp2'])]
    occipital_alpha = [c for c in df.columns if 'Alpha' in c and any(x in c for x in ['O1','O2','Oz'])]
    if frontal_beta and occipital_alpha:
        df['Ratio.Global.FrontalOccipital'] = df[frontal_beta].sum(axis=1) / (df[occipital_alpha].sum(axis=1) + 1e-6)
    
    return df

def preprocess_data(df):
    """Preprocess data similar to training pipeline"""
    df = add_ratios_smart(df)
    
    keep_cols = []
    for col in df.columns:
        if "Ratio.Global" in col:
            keep_cols.append(col)
            continue
        if not ("Pow" in col or "EEG" in col or "Ratio" in col):
            continue
        if any(bad in col for bad in ['Counter', 'Battery', 'Clock', 'Interpolated', 'Quality', 'BufferSize']):
            continue
        
        parts = col.split('.')
        if len(parts) > 1 and parts[1] in TARGET_SENSORS:
            keep_cols.append(col)
    
    if not keep_cols:
        return None, None
    
    df_features = df[keep_cols]
    df_smooth = df_features.rolling(window=ROLLING_WINDOW_SIZE).mean().dropna()
    
    return df_smooth, keep_cols

def preprocess_for_cnn(df):
    """Preprocess data for CNN (raw time series segments)"""
    try:
        # 1. Select EEG channels in correct order matching TARGET_SENSORS
        eeg_cols = []
        for ch in TARGET_SENSORS:
            col = f"EEG.{ch}"
            if col in df.columns:
                eeg_cols.append(col)
            else:
                # Try case-insensitive search or alternative naming
                found = False
                for df_col in df.columns:
                    df_col_str = str(df_col).strip()
                    if (df_col_str.upper() == col.upper() or 
                        (ch in df_col_str and "EEG" in df_col_str.upper())):
                        eeg_cols.append(df_col)
                        found = True
                        break
                if not found:
                    return None  # Required channel not found
        
        if len(eeg_cols) != len(TARGET_SENSORS):
            return None  # Not all required channels found

        eeg_data = df[eeg_cols].values  # (samples, num_channels)
        
        if eeg_data.shape[0] < WINDOW_SIZE:
            return None  # Not enough samples

        # 2. Per-recording z-score normalization
        eeg_data = (eeg_data - eeg_data.mean(axis=0)) / (eeg_data.std(axis=0) + 1e-6)

        # 3. Sliding window segmentation
        segments = []
        for start in range(0, eeg_data.shape[0] - WINDOW_SIZE + 1, STEP_SIZE):
            seg = eeg_data[start:start + WINDOW_SIZE]
            segments.append(seg)
        
        if len(segments) == 0:
            return None

        return np.array(segments, dtype=np.float32)
    except Exception:
        return None

def create_brain_heatmap(electrode_values, title="Brain Activity Heatmap", use_zscore=True, color_scale='plasma'):
    """Create a futuristic brain heatmap visualization with smooth interpolation.
    
    Features:
    - Dark theme with transparent background
    - Smooth interpolation for continuous heatmap
    - Futuristic colormap (plasma)
    - Contour lines for depth
    - Horizontal colorbar at bottom
    """
    fig = go.Figure()
    
    # Create electrode positions and values
    x_pos = []
    y_pos = []
    raw_values = []
    labels = []
    
    for electrode, (x, y) in ELECTRODE_POSITIONS.items():
        if electrode in electrode_values:
            x_pos.append(x)
            y_pos.append(y)
            raw_val = float(electrode_values[electrode])
            raw_values.append(raw_val)
            labels.append(f"{electrode}<br>Raw: {raw_val:.2f}")
    
    if not raw_values:
        return None
    
    values = np.array(raw_values, dtype=float)
    
    # Convert to z-scores for better visualization
    if use_zscore and len(values) > 1:
        mean_val = float(np.mean(values))
        std_val = float(np.std(values)) or 1e-6
        z_values = (values - mean_val) / std_val
        values_for_color = z_values
        max_abs = float(np.max(np.abs(z_values)))
        cmin, cmax = -max_abs, max_abs
        colorbar_title = "Z-score"
    else:
        values_for_color = values
        cmin, cmax = float(np.min(values)), float(np.max(values))
        colorbar_title = "Activity Level"
    
    # Create a fine grid for smooth interpolation
    grid_resolution = 100
    x_grid = np.linspace(-1.1, 1.1, grid_resolution)
    y_grid = np.linspace(-1.1, 1.1, grid_resolution)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Interpolate values onto the grid
    points = np.column_stack((x_pos, y_pos))
    Z_grid = griddata(points, values_for_color, (X_grid, Y_grid), method='cubic', fill_value=np.nan)
    
    # Create a mask for the head shape (circle)
    head_mask = X_grid**2 + Y_grid**2 <= 1.0
    Z_grid[~head_mask] = np.nan
    
    # Add smooth heatmap surface
    fig.add_trace(go.Contour(
        x=x_grid,
        y=y_grid,
        z=Z_grid,
        colorscale=color_scale,
        showscale=True,
        colorbar=dict(
            title=dict(text=colorbar_title, font=dict(color='white', size=12)),
            tickfont=dict(color='white', size=10),
            len=0.4,
            thickness=15,
            x=0.5,
            y=-0.15,
            xanchor='center',
            yanchor='middle',
            orientation='h',
            tickmode='linear',
            tick0=cmin,
            dtick=(cmax - cmin) / 5,
            tickformat='.2f'
        ),
        contours=dict(
            showlines=True,
            start=cmin,
            end=cmax,
            size=(cmax - cmin) / 8,
            coloring='heatmap'
        ),
        line=dict(width=0.5, color='rgba(255,255,255,0.3)'),
        hovertemplate='Z-score: %{z:.2f}<extra></extra>',
        name='Activity'
    ))
    
    # Add electrode markers with labels
    key_electrodes = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'C3', 'C4', 'Cz', 'P3', 'P4', 'Pz', 'O1', 'O2', 'Oz']
    
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode='markers',
        marker=dict(
            size=12,
            color=values_for_color,
            colorscale=color_scale,
            cmin=cmin,
            cmax=cmax,
            line=dict(width=2, color='white'),
            showscale=False
        ),
        text=[label.split('<br>')[0] for label in labels],
        textposition="middle center",
        textfont=dict(size=9, color='white', family='Arial Black'),
        hovertemplate='%{text}<br>Z-score: %{marker.color:.2f}<extra></extra>',
        name='Electrodes',
        showlegend=False
    ))
    
    # Draw head outline (circle) - white for dark theme
    theta = np.linspace(0, 2*np.pi, 100)
    head_x = np.cos(theta)
    head_y = np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=head_x,
        y=head_y,
        mode='lines',
        line=dict(color='rgba(255,255,255,0.8)', width=3),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add nose - white for dark theme
    nose_x = [0, 0.1, 0, -0.1, 0]
    nose_y = [1, 0.95, 0.9, 0.95, 1]
    fig.add_trace(go.Scatter(
        x=nose_x,
        y=nose_y,
        mode='lines',
        line=dict(color='rgba(255,255,255,0.8)', width=3),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Enhanced title styling
    title_text = f"<b style='font-size:20px; color:#00ffff;'>{title}</b><br>" \
                 f"<span style='font-size:12px; color:#cccccc;'>Red indicates high-amplitude regions, Blue indicates low-amplitude regions</span>"
    
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center',
            font=dict(color='white')
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1.2, 1.2]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1.2, 1.2],
            scaleanchor="x",
            scaleratio=1
        ),
        height=650,
        autosize=True,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper
        margin=dict(l=20, r=20, t=80, b=80)
    )
    
    return fig

def create_brain_heatmap_animation(
    frames_data,
    title="Brain Activity Over Time",
    use_zscore=True,
    color_scale='RdBu_r'
):
    # ---- Prepare electrode positions ----
    electrodes = list(ELECTRODE_POSITIONS.keys())
    x_pos = [ELECTRODE_POSITIONS[e][0] for e in electrodes]
    y_pos = [ELECTRODE_POSITIONS[e][1] for e in electrodes]

    # ---- Helper to compute z-scored values ----
    def normalize(values):
        values = np.array(values, dtype=float)
        if use_zscore and len(values) > 1:
            mean = values.mean()
            std = values.std() or 1e-6
            z = (values - mean) / std
            max_abs = np.max(np.abs(z))
            return z, -max_abs, max_abs
        return values, values.min(), values.max()

    # ---- First frame ----
    first_vals = [frames_data[0].get(e, 0) for e in electrodes]
    z_vals, cmin, cmax = normalize(first_vals)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_pos,
                y=y_pos,
                mode="markers",
                marker=dict(
                    size=14,
                    color=z_vals,
                    colorscale=color_scale,
                    cmin=cmin,
                    cmax=cmax,
                    line=dict(width=2, color="white"),
                    colorbar=dict(title="Z-score")
                ),
                text=electrodes,
                hovertemplate="%{text}<br>Z: %{marker.color:.2f}<extra></extra>"
            )
        ],
        layout=go.Layout(
            title=title,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 120}, "fromcurrent": True}]
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0}}]
                    }
                ]
            }]
        ),
        frames=[
            go.Frame(
                data=[
                    go.Scatter(
                        marker=dict(
                            color=normalize([frame.get(e, 0) for e in electrodes])[0]
                        )
                    )
                ],
                name=str(i)
            )
            for i, frame in enumerate(frames_data)
        ]
    )

    return fig

def extract_electrode_data(df, frequency_band='Alpha', metric='mean'):
    """Extract electrode values for a specific frequency band.
    Extracts all available electrodes that have positions defined.

    metric: 'mean', 'max', or 'min'
    """
    electrode_values = {}
    metric = metric.lower()
    
    # Find all available electrodes from the data
    pow_cols = [c for c in df.columns if str(c).strip().startswith("POW.")]
    
    # Extract electrode names from column names
    available_electrodes = set()
    for col in pow_cols:
        parts = str(col).split('.')
        if len(parts) >= 3 and frequency_band in parts[-1]:
            # Extract electrode name (e.g., "POW.Fp1.Alpha" -> "Fp1")
            electrode = parts[1]
            if electrode in ELECTRODE_POSITIONS:
                available_electrodes.add(electrode)
    
    # Extract values for all available electrodes
    for sensor in available_electrodes:
        col_name = f"POW.{sensor}.{frequency_band}"
        series = None
        if col_name in df.columns:
            series = df[col_name]
        else:
            # Try alternative column names
            for col in pow_cols:
                if sensor in str(col) and frequency_band in str(col):
                    series = df[col]
                    break
        if series is not None:
            if metric == 'max':
                value = float(series.max())
            elif metric == 'min':
                value = float(series.min())
            else:
                value = float(series.mean())
            electrode_values[sensor] = value
    
    return electrode_values

def generate_electrode_frames(
    df,
    frequency_band="Alpha",
    metric="mean",
    window_size=50,
    step_size=10
):
    """
    Split EEG data into overlapping windows and extract electrode values per window.
    Returns a list of electrode_value dictionaries (one per frame).
    """
    frames = []

    for start in range(0, len(df) - window_size, step_size):
        df_window = df.iloc[start:start + window_size]
        values = extract_electrode_data(df_window, frequency_band, metric)
        if values:
            frames.append(values)

    return frames

# ============================================
# PAGE 1: Data Overview
# ============================================
def page_scientific_defense():
    """Dataset overview dashboard."""
    st.header("📊 Data Overview")
    st.markdown("**High-level summary of your EEG emotion dataset.**")
    
    # Top-level dataset stats (like the sample dashboard)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sessions", DATASET_INFO['total_files'])
    with col2:
        st.metric("Training Sessions", DATASET_INFO['train_files'])
    with col3:
        total = DATASET_INFO['total_files']
        test = DATASET_INFO['test_files']
        test_pct = (test / total) * 100 if total else 0
        st.metric("Test Split", f"{test} ({test_pct:.1f}%)")
    
    st.divider()
    
    # Class distribution summary
    st.subheader("Emotion Class Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            values=list(DATASET_INFO['class_distribution'].values()),
            names=list(DATASET_INFO['class_distribution'].keys()),
            title="Label Balance",
            color_discrete_map=EMOTION_COLORS
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Distribution Details")
        for emotion, count in DATASET_INFO['class_distribution'].items():
            percentage = (count / DATASET_INFO['total_files']) * 100 if DATASET_INFO['total_files'] else 0
            st.write(f"**{emotion}**: {count} samples ({percentage:.1f}%)")
    
    st.divider()
    
    # Feature descriptions table (similar to sample screenshot)
    st.subheader("📚 Feature Descriptions")
    feature_rows = [
        {
            "Feature": "Timestamp",
            "Description": "Sampling time index for each EEG frame."
        },
        {
            "Feature": "Raw EEG Channels",
            "Description": "Voltage readings from sensors (Cz, Fz, Fp1, Fp2, F3, F4, etc.)."
        },
        {
            "Feature": "Power Spectral Features (POW.*)",
            "Description": "Band power per electrode for Theta, Alpha, Beta, Gamma ranges."
        },
        {
            "Feature": "Context Windows",
            "Description": "Stacked frames (window size 5) to capture temporal context for SVM."
        },
        {
            "Feature": "Emotion Label",
            "Description": "Ground-truth class for each session (Happy, Neutral, Sad, Anxiety)."
        },
        {
            "Feature": "Metadata",
            "Description": "Any additional markers or quality indices included in the CSV."
        },
    ]
    features_df = pd.DataFrame(feature_rows)
    st.dataframe(features_df, use_container_width=True, hide_index=True)

# ============================================
# PAGE 2: Live Monitor (Creative)
# ============================================
def page_live_monitor():
    st.title("🫀 Live EEG Emotion Monitor")
    st.markdown("**Simulates a real-time data stream to demonstrate sequential analysis.**")
    
    # Load models
    svm_model, svm_scaler, svm_artifact = load_svm_model()
    cnn_model, cnn_artifact = load_cnn_model()
    if svm_model is None:
        st.error("SVM model not found. Please ensure 'models/eeg_emotion_svm_model.pkl' exists.")
        return
    
    # Show model status (only show warnings/errors, not success messages)
    if cnn_model is None:
        if not TF_AVAILABLE:
            st.warning("⚠️ CNN model: TensorFlow not installed. Install with `pip install tensorflow`")
        else:
            # Try to find the model file to give a better error message
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_path = os.path.join(project_root, 'models', 'eeg_emotion_cnn_model.keras')
            if os.path.exists(model_path):
                st.error(f"⚠️ CNN model: Model file exists but failed to load. Check console for errors.")
            else:
                st.warning(f"⚠️ CNN model: Model file not found. Expected at: {model_path}")
    
    uploaded_file = st.file_uploader("Upload EEG CSV file", type=["csv"], key="live")
    
    # Clear previous simulation results when a new file is uploaded
    if uploaded_file is not None:
        # Check if this is a new file (different from what was processed)
        current_file_name = uploaded_file.name
        if 'last_processed_file' not in st.session_state or st.session_state['last_processed_file'] != current_file_name:
            # Clear previous results
            if 'svm_result' in st.session_state:
                del st.session_state['svm_result']
            if 'cnn_result' in st.session_state:
                del st.session_state['cnn_result']
            st.session_state['last_processed_file'] = current_file_name
    
    if uploaded_file is not None:
        try:
            # Smart header detection - find the row with "POW." and "Timestamp"
            import csv
            import io
            
            # Read the file content
            content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            # Try to detect header row
            header_row_index = None
            reader = csv.reader(io.StringIO(content.decode('utf-8', errors='replace')))
            for i, row in enumerate(reader):
                row_str = ",".join(row)
                if "POW." in row_str and "Timestamp" in row_str:
                    header_row_index = i
                    break
            
            # Read CSV with detected header, fallback to header=1 if not found
            if header_row_index is not None:
                df = pd.read_csv(uploaded_file, header=header_row_index, low_memory=False)
            else:
                # Try header=1 first
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, header=1, low_memory=False)
                # If still no POW columns, try header=0
                if not any(str(c).startswith("POW.") for c in df.columns):
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, header=0, low_memory=False)
            
            # Filter for visualization columns - try common frontal electrodes first
            # Priority: AF3/AF4 > Fp1/Fp2 > F3/F4 > first available POW columns
            vis_cols = []
            
            # Get all POW columns first
            all_pow_cols = [c for c in df.columns if str(c).strip().startswith("POW.")]
            
            if not all_pow_cols:
                st.error("Could not find any POW columns in the data. Please ensure your CSV contains POW columns.")
                return
                
            # Try AF3/AF4 first (if available)
            vis_cols = [c for c in all_pow_cols if 'AF3' in str(c) or 'AF4' in str(c)]
            
            # If not found, try Fp1/Fp2 (common in user's data) - prefer Alpha band
            if not vis_cols:
                fp_cols = [c for c in all_pow_cols if ('Fp1' in str(c) or 'Fp2' in str(c)) and '.Alpha' in str(c)]
                if len(fp_cols) >= 2:
                    vis_cols = fp_cols[:2]
                elif len(fp_cols) >= 1:
                    # Get one Fp1 and one Fp2 if available
                    fp1_cols = [c for c in all_pow_cols if 'Fp1' in str(c) and '.Alpha' in str(c)]
                    fp2_cols = [c for c in all_pow_cols if 'Fp2' in str(c) and '.Alpha' in str(c)]
                    if fp1_cols and fp2_cols:
                        vis_cols = [fp1_cols[0], fp2_cols[0]]
                    else:
                        vis_cols = fp_cols[:1]
            
            # If still not found, try F3/F4 Alpha band
            if not vis_cols:
                f_cols = [c for c in all_pow_cols if ('F3' in str(c) or 'F4' in str(c)) and '.Alpha' in str(c)]
                if len(f_cols) >= 2:
                    vis_cols = f_cols[:2]
                elif len(f_cols) >= 1:
                    f3_cols = [c for c in all_pow_cols if 'F3' in str(c) and '.Alpha' in str(c)]
                    f4_cols = [c for c in all_pow_cols if 'F4' in str(c) and '.Alpha' in str(c)]
                    if f3_cols and f4_cols:
                        vis_cols = [f3_cols[0], f4_cols[0]]
                    else:
                        vis_cols = f_cols[:1]
            
            # Final fallback: use first available Alpha band POW columns
            if not vis_cols:
                st.info("Using Alpha band columns for visualization.")
                alpha_cols = [c for c in all_pow_cols if '.Alpha' in str(c)]
                if len(alpha_cols) >= 2:
                    vis_cols = alpha_cols[:2]
                elif len(alpha_cols) >= 1:
                    vis_cols = alpha_cols[:1]
                else:
                    # Last resort: any POW columns
                    vis_cols = all_pow_cols[:2] if len(all_pow_cols) >= 2 else all_pow_cols[:1] if len(all_pow_cols) >= 1 else []
            
            if not vis_cols:
                st.error("Could not find visualization columns. Please ensure your CSV contains POW columns.")
                # Debug: show what columns were found
                all_pow_debug = [c for c in df.columns if str(c).strip().startswith("POW.")]
                if all_pow_debug:
                    st.write(f"Found {len(all_pow_debug)} POW columns. First 5: {all_pow_debug[:5]}")
                return
            
            vis_data = df[vis_cols].apply(pd.to_numeric, errors='coerce').dropna()
            
            if vis_data.empty:
                st.error("Visualization columns contain no valid numeric data.")
                return
            
            # Prepare data for SVM model - match training pipeline exactly
            # Training uses raw POW columns directly, not preprocessed features
            pow_cols = [c for c in df.columns if str(c).strip().startswith("POW.")]
            if not pow_cols:
                st.error("Could not find POW columns in the data.")
                return
            
            # Extract and convert to numeric
            pow_df = df[pow_cols].apply(pd.to_numeric, errors='coerce').dropna(how='any')
            if pow_df.empty:
                st.error("POW columns contain no valid numeric data.")
                return
            
            # Create context windows (matches training pipeline)
            X_stacked = create_context_windows(pow_df.values, CONTEXT_WINDOW)
            
            if len(X_stacked) == 0:
                st.error("Not enough data for context window stacking.")
                return
                
            # Scale and predict
            if svm_scaler is not None:
                X_scaled = svm_scaler.transform(X_stacked)
            else:
                # Fallback: use RobustScaler like in training
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X_stacked)
            
            y_pred_raw = svm_model.predict(X_scaled)
            y_pred_smooth = medfilt(y_pred_raw, kernel_size=5)
            
            # Prepare CNN predictions for real-time display
            cnn_predictions_by_time = None
            cnn_available = False
            cnn_error_msg = None
            
            if cnn_model is None:
                if not TF_AVAILABLE:
                    cnn_error_msg = "TensorFlow not available. Please install tensorflow."
                else:
                    cnn_error_msg = "CNN model file not found. Please ensure 'models/eeg_emotion_cnn_model.keras' exists."
            else:
                try:
                    cnn_data = preprocess_for_cnn(df)
                    if cnn_data is None:
                        # Check what columns are available
                        available_eeg_cols = [c for c in df.columns if 'EEG' in str(c).upper()]
                        missing_channels = []
                        for ch in TARGET_SENSORS:
                            col = f"EEG.{ch}"
                            if col not in df.columns:
                                # Check case-insensitive
                                found = any(str(c).upper() == col.upper() for c in df.columns)
                                if not found:
                                    missing_channels.append(ch)
                        
                        if len(missing_channels) > 0:
                            cnn_error_msg = f"Missing EEG channels: {', '.join(missing_channels[:10])}{'...' if len(missing_channels) > 10 else ''}. Found {len(available_eeg_cols)} EEG columns."
                        elif len(available_eeg_cols) == 0:
                            cnn_error_msg = "No EEG columns found in data. CNN requires raw EEG data (EEG.Cz, EEG.Fz, etc.)."
                        else:
                            cnn_error_msg = f"Data preprocessing failed. Found {len(available_eeg_cols)} EEG columns but could not create segments."
                    elif len(cnn_data) == 0:
                        cnn_error_msg = "Not enough data samples for CNN (requires at least 128 samples)."
                    else:
                        cnn_probs = cnn_model.predict(cnn_data, verbose=0)   # (n_segments, 4)
                        cnn_predictions = np.argmax(cnn_probs, axis=1)
                        cnn_smooth = medfilt(cnn_predictions, kernel_size=5)
                        
                        # Map CNN segment predictions to time steps
                        # CNN uses sliding windows: segment i covers [i*STEP_SIZE, i*STEP_SIZE+WINDOW_SIZE-1]
                        cnn_predictions_by_time = np.zeros(len(vis_data), dtype=int)
                        for seg_idx in range(len(cnn_smooth)):
                            start_time = seg_idx * STEP_SIZE
                            end_time = min(start_time + WINDOW_SIZE, len(vis_data))
                            # Assign this segment's prediction to all time steps it covers
                            cnn_predictions_by_time[start_time:end_time] = cnn_smooth[seg_idx]
                        
                        cnn_available = True
                except Exception as e:
                    cnn_error_msg = f"CNN processing error: {str(e)}"
                    cnn_available = False
            
            # Ensure vis_data and predictions are aligned
            if len(vis_data) == 0:
                st.error("No valid visualization data after processing.")
                return
            
            st.divider()
            
            # Two separate simulation sections
            st.subheader("🤖 SVM Simulation")
            svm_col_graph, svm_col_status = st.columns([3, 1])
            
            svm_start_btn = st.button("▶ Start SVM Live Simulation", type="primary", key="svm_sim")
            
            # Display stored SVM results if they exist and simulation is not running
            if 'svm_result' in st.session_state and st.session_state['svm_result'].get('chart') is not None and not svm_start_btn:
                svm_col_graph.plotly_chart(st.session_state['svm_result']['chart'], use_container_width=True, key="svm_chart_final")
                if st.session_state['svm_result'].get('status'):
                    svm_col_status.markdown(st.session_state['svm_result']['status'], unsafe_allow_html=True)
            
            if svm_start_btn:
                svm_chart_placeholder = svm_col_graph.empty()
                svm_status_placeholder = svm_col_status.empty()
                svm_progress_bar = st.progress(0)
                
                # Simulate streaming
                step_size = max(1, len(vis_data) // 100)  # Show ~100 updates
                total_steps = len(vis_data) // step_size
                
                # Store final chart and status for persistence
                final_svm_fig = None
                final_svm_status = None
                
                for step, i in enumerate(range(0, len(vis_data), step_size)):
                    # Update chart (show last 50 points)
                    chart_data = vis_data.iloc[max(0, i-50):i+1]
                    if len(chart_data) > 0:
                        fig = go.Figure()
                        for col in chart_data.columns:
                            # Create a better label: extract electrode and frequency band
                            col_str = str(col)
                            if '.' in col_str:
                                parts = col_str.split('.')
                                if len(parts) >= 3:
                                    # Format: POW.Electrode.FrequencyBand -> "Electrode FrequencyBand"
                                    label = f"{parts[1]} {parts[2]}"
                                else:
                                    label = parts[-1]
                            else:
                                label = col_str
                            fig.add_trace(go.Scatter(
                                y=chart_data[col].values,
                                mode='lines',
                                name=label
                            ))
                        fig.update_layout(
                            title=f"Live EEG Stream - SVM (Sample {i}/{len(vis_data)})",
                            xaxis_title="Time Step",
                            yaxis_title="Power",
                            height=400
                        )
                        svm_chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"svm_chart_live_{i}")
                        final_svm_fig = fig  # Store final figure
                    
                    # Update emotion status - SVM only
                    pred_idx = min(i // step_size, len(y_pred_smooth) - 1)
                    if pred_idx >= 0:
                        current_emotion_idx = y_pred_smooth[pred_idx]
                        svm_emotion_name = EMOTION_LABELS.get(current_emotion_idx, "Unknown")
                        
                        # Dynamic colors
                        color_map = {
                            "Happy": "🟢",
                            "Neutral": "⚪",
                            "Sad": "🔵",
                            "Anxiety": "🔴"
                        }
                        svm_emoji = color_map.get(svm_emotion_name, "⚪")
                        
                        status_html = f"""
                        ### Status:
                        <h4>🤖 SVM:</h4>
                        <h2>{svm_emoji} {svm_emotion_name}</h2>
                        <hr style="margin: 1rem 0;">
                        <p><small><strong>Time Step:</strong> {i}</small></p>
                        <p><small><strong>Window:</strong> {pred_idx}</small></p>
                        """
                        svm_status_placeholder.markdown(status_html, unsafe_allow_html=True)
                        final_svm_status = status_html  # Store final status
                    
                    svm_progress_bar.progress((step + 1) / total_steps)
                    time.sleep(0.05)  # Controls playback speed
                
                svm_progress_bar.empty()
                st.success("✅ SVM Simulation complete!")
                
                # Store SVM results for comparison and display
                unique_svm, counts_svm = np.unique(y_pred_smooth, return_counts=True)
                sorted_idx = np.argsort(counts_svm)[::-1]
                top2_svm = []
                total = np.sum(counts_svm)

                for idx in sorted_idx[:2]:
                    emotion = EMOTION_LABELS[unique_svm[idx]]
                    confidence = (counts_svm[idx] / total) * 100
                    top2_svm.append((emotion, confidence))
                svm_emotion_final = top2_svm[0][0]
                svm_confidence_final = top2_svm[0][1]
                st.session_state['svm_result'] = {
                    'emotion': svm_emotion_final, 
                    'confidence': svm_confidence_final,
                    'top2': top2_svm,
                    'chart': final_svm_fig,
                    'status': final_svm_status
                }
            
            st.divider()
            
            # CNN Simulation Section
            st.subheader("🧠 CNN Simulation")
            cnn_col_graph, cnn_col_status = st.columns([3, 1])
            
            cnn_start_btn = st.button("▶ Start CNN Live Simulation", type="primary", key="cnn_sim")
            
            # Display stored CNN results if they exist and simulation is not running
            if 'cnn_result' in st.session_state and st.session_state['cnn_result'].get('chart') is not None and not cnn_start_btn:
                cnn_col_graph.plotly_chart(st.session_state['cnn_result']['chart'], use_container_width=True, key="cnn_chart_final")
                if st.session_state['cnn_result'].get('status'):
                    cnn_col_status.markdown(st.session_state['cnn_result']['status'], unsafe_allow_html=True)
            
            if cnn_start_btn:
                if not cnn_available:
                    st.error(f"CNN is not available: {cnn_error_msg if cnn_error_msg else 'Unknown error'}")
                else:
                    cnn_chart_placeholder = cnn_col_graph.empty()
                    cnn_status_placeholder = cnn_col_status.empty()
                    cnn_progress_bar = st.progress(0)
                    
                    # Simulate streaming for CNN
                    step_size = max(1, len(vis_data) // 100)  # Show ~100 updates
                    total_steps = len(vis_data) // step_size
                    
                    # Store final chart and status for persistence
                    final_cnn_fig = None
                    final_cnn_status = None
                    
                    for step, i in enumerate(range(0, len(vis_data), step_size)):
                        # Update chart (show last 50 points)
                        chart_data = vis_data.iloc[max(0, i-50):i+1]
                        if len(chart_data) > 0:
                            fig = go.Figure()
                            for col in chart_data.columns:
                                # Create a better label: extract electrode and frequency band
                                col_str = str(col)
                                if '.' in col_str:
                                    parts = col_str.split('.')
                                    if len(parts) >= 3:
                                        # Format: POW.Electrode.FrequencyBand -> "Electrode FrequencyBand"
                                        label = f"{parts[1]} {parts[2]}"
                                    else:
                                        label = parts[-1]
                                else:
                                    label = col_str
                                fig.add_trace(go.Scatter(
                                    y=chart_data[col].values,
                                    mode='lines',
                                    name=label
                                ))
                            fig.update_layout(
                                title=f"Live EEG Stream - CNN (Sample {i}/{len(vis_data)})",
                                xaxis_title="Time Step",
                                yaxis_title="Power",
                                height=400
                            )
                            cnn_chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"cnn_chart_live_{i}")
                            final_cnn_fig = fig  # Store final figure
                        
                        # Update emotion status - CNN only
                        if cnn_predictions_by_time is not None and i < len(cnn_predictions_by_time):
                            cnn_emotion_idx = cnn_predictions_by_time[i]
                            cnn_emotion_name = EMOTION_LABELS.get(cnn_emotion_idx, "Unknown")
                            
                            # Dynamic colors
                            color_map = {
                                "Happy": "🟢",
                                "Neutral": "⚪",
                                "Sad": "🔵",
                                "Anxiety": "🔴"
                            }
                            cnn_emoji = color_map.get(cnn_emotion_name, "⚪")
                            
                            # Calculate segment index for CNN
                            # CNN uses sliding windows, so segment index is based on time step
                            cnn_seg_idx = i // STEP_SIZE
                            
                            status_html = f"""
                            ### Status:
                            <h4>🧠 CNN:</h4>
                            <h2>{cnn_emoji} {cnn_emotion_name}</h2>
                            <hr style="margin: 1rem 0;">
                            <p><small><strong>Time Step:</strong> {i}</small></p>
                            <p><small><strong>Segment:</strong> {cnn_seg_idx}</small></p>
                            """
                            cnn_status_placeholder.markdown(status_html, unsafe_allow_html=True)
                            final_cnn_status = status_html  # Store final status
                        
                        cnn_progress_bar.progress((step + 1) / total_steps)
                        time.sleep(0.05)  # Controls playback speed
                    
                    cnn_progress_bar.empty()
                    st.success("✅ CNN Simulation complete!")
                    
                    # Store CNN results for comparison and display
                    unique_cnn, counts_cnn = np.unique(cnn_smooth, return_counts=True)
                    sorted_idx = np.argsort(counts_cnn)[::-1]
                    total = np.sum(counts_cnn)

                    top2_cnn = []
                    for idx in sorted_idx[:2]:
                        emotion = EMOTION_LABELS[unique_cnn[idx]]
                        confidence = (counts_cnn[idx] / total) * 100
                        top2_cnn.append((emotion, confidence))

                    cnn_emotion_final = top2_cnn[0][0]
                    cnn_confidence_final = top2_cnn[0][1]
                    
                    st.session_state['cnn_result'] = {
                        'emotion': cnn_emotion_final, 
                        'confidence': cnn_confidence_final,
                        'top2': top2_cnn,
                        'chart': final_cnn_fig,
                        'status': final_cnn_status
                    }
            
            
            # Show comparison if both simulations have been run
            if 'svm_result' in st.session_state and 'cnn_result' in st.session_state:
                st.divider()
                st.subheader("📊 Comparison: SVM vs CNN")
                col1, col2 = st.columns(2)
                with col1:
                    svm_res = st.session_state['svm_result']
                    svm_top2_text = "<br>".join(
                        [f"{emo}: {conf:.1f}%" for emo, conf in svm_res['top2']]
                    )
                    st.markdown(f"""
                    <div class="status-box" style="border-color: {EMOTION_COLORS[svm_res['emotion']]};">
                        <h3>🤖 SVM</h3>
                        <h2>{svm_res['emotion']}</h2>
                        <p>Confidence: {svm_res['confidence']:.1f}%</p>
                        <hr>
                        <p>Top-2 Emotions</strong><br>{svm_top2_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    cnn_res = st.session_state['cnn_result']
                    cnn_top2_text = ""
                    if 'top2' in cnn_res:
                        cnn_top2_text = "<br>".join(
                            [f"{emo}: {conf:.1f}%" for emo, conf in cnn_res['top2']]
                        )
                    st.markdown(f"""
                    <div class="status-box" style="border-color: {EMOTION_COLORS[cnn_res['emotion']]};">
                        <h3>🧠 CNN</h3>
                        <h2>{cnn_res['emotion']}</h2>
                        <p>Confidence: {cnn_res['confidence']:.1f}%</p>
                        <hr>
                        <p>Top-2 Emotions</strong><br>{cnn_top2_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if svm_res['emotion'] == cnn_res['emotion']:
                    st.success(f"✅ Both models agree: **{svm_res['emotion']}**")
                else:
                    st.warning(f"⚠️ Models disagree: SVM predicts **{svm_res['emotion']}**, CNN predicts **{cnn_res['emotion']}**")
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e)
        
# ============================================
# PAGE 3: Comparative Analysis (SVM vs CNN)
# ============================================
def page_head_to_head():
    st.header("Comparative Analysis: SVM vs CNN")
    
    svm_model, svm_scaler, svm_artifact = load_svm_model()
    cnn_model, cnn_artifact = load_cnn_model()
    
    if svm_model is None:
        st.error("SVM model not found.")
        return
    
    uploaded_file = st.file_uploader("Upload EEG CSV file for comparison", type=['csv'], key="head2head")
    
    if uploaded_file is not None:
        try:
            # Smart header detection - same as Live Monitor page
            import csv
            import io
            
            content = uploaded_file.read()
            uploaded_file.seek(0)
            
            header_row_index = None
            reader = csv.reader(io.StringIO(content.decode('utf-8', errors='replace')))
            for i, row in enumerate(reader):
                row_str = ",".join(row)
                if "POW." in row_str and "Timestamp" in row_str:
                    header_row_index = i
                    break
            
            if header_row_index is not None:
                df = pd.read_csv(uploaded_file, header=header_row_index, low_memory=False)
            else:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, header=1, low_memory=False)
                if not any(str(c).startswith("POW.") for c in df.columns):
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, header=0, low_memory=False)
            
            col1, col2 = st.columns(2)
            
            # SVM Predictions
            with col1:
                st.subheader("🤖 SVM Predictions (Context Stacking)")
                with st.spinner("Processing with SVM..."):
                    # Match training pipeline: use raw POW columns
                    pow_cols = [c for c in df.columns if str(c).strip().startswith("POW.")]
                    if pow_cols:
                        pow_df = df[pow_cols].apply(pd.to_numeric, errors='coerce').dropna(how='any')
                        if not pow_df.empty:
                            X_stacked = create_context_windows(pow_df.values, CONTEXT_WINDOW)
                            
                            if len(X_stacked) > 0:
                                if svm_scaler is not None:
                                    X_scaled = svm_scaler.transform(X_stacked)
                                else:
                                    from sklearn.preprocessing import RobustScaler
                                    scaler = RobustScaler()
                                    X_scaled = scaler.fit_transform(X_stacked)
                                
                                svm_predictions = svm_model.predict(X_scaled)
                                svm_smooth = medfilt(svm_predictions, kernel_size=5)
                                
                                unique, counts = np.unique(svm_smooth, return_counts=True)
                                most_common_idx = unique[np.argmax(counts)]
                                svm_emotion = EMOTION_LABELS[most_common_idx]
                                svm_confidence = (counts[np.argmax(counts)] / len(svm_smooth)) * 100
                                
                                st.markdown(f"""
                                <div class="status-box" style="border-color: {EMOTION_COLORS[svm_emotion]};">
                                    {svm_emotion}<br>
                                    Confidence: {svm_confidence:.1f}%
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Distribution
                                emotion_counts = {EMOTION_LABELS[k]: int(v) for k, v in zip(unique, counts)}
                                fig_svm = px.bar(
                                    x=list(emotion_counts.keys()),
                                    y=list(emotion_counts.values()),
                                    title="SVM Prediction Distribution",
                                    color=list(emotion_counts.keys()),
                                    color_discrete_map=EMOTION_COLORS
                                )
                                st.plotly_chart(fig_svm, use_container_width=True)
                            else:
                                st.error("Not enough data for context stacking.")
                        else:
                            st.error("Could not extract POW columns from data.")
                    else:
                        st.error("No POW columns found in the uploaded file.")
            
            # CNN Predictions
            with col2:
                st.subheader("🧠 CNN Predictions (Feature Learning)")
                if cnn_model is None:
                    st.warning("CNN model not available. Please ensure model file exists.")
                    st.info("Expected path: `models/eeg_emotion_cnn_model.h5` or `.keras`")
                else:
                    with st.spinner("Processing with CNN..."):
                        cnn_data = preprocess_for_cnn(df)
                        if cnn_data is not None:
                            cnn_predictions = np.argmax(cnn_model.predict(cnn_data, verbose=0), axis=1)
                            cnn_smooth = medfilt(cnn_predictions, kernel_size=3)
                            
                            unique, counts = np.unique(cnn_smooth, return_counts=True)
                            most_common_idx = unique[np.argmax(counts)]
                            cnn_emotion = EMOTION_LABELS[most_common_idx]
                            cnn_confidence = (counts[np.argmax(counts)] / len(cnn_smooth)) * 100
                            
                            st.markdown(f"""
                            <div class="status-box" style="border-color: {EMOTION_COLORS[cnn_emotion]};">
                                {cnn_emotion}<br>
                                Confidence: {cnn_confidence:.1f}%
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Distribution
                            emotion_counts = {EMOTION_LABELS[k]: int(v) for k, v in zip(unique, counts)}
                            fig_cnn = px.bar(
                                x=list(emotion_counts.keys()),
                                y=list(emotion_counts.values()),
                                title="CNN Prediction Distribution",
                                color=list(emotion_counts.keys()),
                                color_discrete_map=EMOTION_COLORS
                            )
                            st.plotly_chart(fig_cnn, use_container_width=True)
                        else:
                            st.error("Could not preprocess data for CNN.")
            
            st.divider()
            
            # Consensus Meter
            if svm_model is not None and cnn_model is not None:
                # Check if both predictions were successful
                if 'svm_emotion' in locals() and 'cnn_emotion' in locals():
                    st.subheader("🎯 Consensus Meter")
                    if svm_emotion == cnn_emotion:
                        consensus_color = "green"
                        consensus_text = "✅ AGREEMENT"
                        consensus_icon = "🟢"
                    else:
                        consensus_color = "red"
                        consensus_text = "❌ DISAGREEMENT"
                        consensus_icon = "🔴"
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 2rem; background-color: {consensus_color}20; border-radius: 10px; border: 3px solid {consensus_color};">
                        <h2>{consensus_icon} {consensus_text}</h2>
                        <p><strong>SVM:</strong> {svm_emotion} ({svm_confidence:.1f}%)</p>
                        <p><strong>CNN:</strong> {cnn_emotion} ({cnn_confidence:.1f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                    st.info("Please wait for both models to complete predictions.")
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e)

# ============================================
# PAGE 4: Brain Heatmap
# ============================================
def page_brain_heatmap():
    st.header("🧠 Brain Activity Heatmap")
    
    # Data source selection
    uploaded_file = st.file_uploader("Upload EEG CSV file", type=['csv'], key="heatmap")
    
    if uploaded_file is not None:
        try:
            # Smart header detection
            import csv
            import io
            
            content = uploaded_file.read()
            uploaded_file.seek(0)
            
            header_row_index = None
            reader = csv.reader(io.StringIO(content.decode('utf-8', errors='replace')))
            for i, row in enumerate(reader):
                row_str = ",".join(row)
                if "POW." in row_str and "Timestamp" in row_str:
                    header_row_index = i
                    break
            
            if header_row_index is not None:
                df = pd.read_csv(uploaded_file, header=header_row_index, low_memory=False)
            else:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, header=1, low_memory=False)
                if not any(str(c).startswith("POW.") for c in df.columns):
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, header=0, low_memory=False)
            
            # Use default settings: Alpha band, mean metric, all data
            frequency_band = "Alpha"
            metric = "mean"
            
            # Extract electrode data
            frames_data = generate_electrode_frames(
                df,
                frequency_band=frequency_band,
                metric=metric,
                window_size=50,
                step_size=10
            )

            if frames_data:
                fig = create_brain_heatmap_animation(
                    frames_data,
                    title="Simulated Real-Time Neural Topography",
                    color_scale="RdBu_r"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough data to generate animation.")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e)

# ============================================
# MAIN APP
# ============================================
def main():
    st.markdown('<div class="main-header">🧠 EEG Emotion Detection - Central Command Center</div>', unsafe_allow_html=True)
    
    # Top Navigation Bar with Tabs (custom order)
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview",
        "🧠 Brain Heatmap",
        "🫀 Live Monitor",
        "⚖️ Comparative Analysis",
    ])
    
    # Route to appropriate page based on selected tab
    with tab1:
        page_scientific_defense()
    
    with tab2:
        page_brain_heatmap()
    
    with tab3:
        page_live_monitor()
    
    with tab4:
        page_head_to_head()

if __name__ == "__main__":
    main()
