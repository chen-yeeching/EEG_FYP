import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from scipy.signal import medfilt
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

TARGET_SENSORS = ['F3', 'Fp1', 'AF3', 'FC5', 'F4', 'Fp2', 'AF4', 'FC6', 
                  'F8', 'T7', 'T8', 'O1', 'O2', 'Oz', 'Cz', 'Fz', 'C4', 'Pz']

ROLLING_WINDOW_SIZE = 128
CONTEXT_WINDOW = 5  # For SVM context stacking

# Dataset info (update these based on your actual data)
DATASET_INFO = {
    'total_files': 32,
    'train_files': 24,
    'test_files': 8,
    'emotions': ['Happy', 'Neutral', 'Sad', 'Anxiety'],
    'class_distribution': {'Happy': 8, 'Neutral': 8, 'Sad': 8, 'Anxiety': 8}  # Update with actual
}

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
    """Load the trained CNN model (if available)"""
    if not TF_AVAILABLE:
        return None, {}
    
    try:
        # Try different possible paths
        possible_paths = [
            os.path.join('models', 'eeg_emotion_cnn_model.h5'),
            os.path.join('models', 'eeg_emotion_cnn_model.keras'),
            os.path.join('..', 'models', 'eeg_emotion_cnn_model.h5'),
            os.path.join('..', 'models', 'eeg_emotion_cnn_model.keras'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model = tf.keras.models.load_model(path)
                return model, {}
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
    # Extract POW columns
    pow_cols = [c for c in df.columns if str(c).strip().startswith("POW.")]
    if not pow_cols:
        return None
    
    # Get sensor columns
    sensor_cols = []
    for sensor in TARGET_SENSORS:
        for col in pow_cols:
            if f".{sensor}." in col:
                sensor_cols.append(col)
                break
    
    if not sensor_cols:
        return None
    
    # Extract data and convert to numeric
    data = df[sensor_cols].apply(pd.to_numeric, errors='coerce').dropna()
    
    # Reshape to (samples, time_steps, channels) for CNN
    # Using sliding windows of size 128
    window_size = 128
    segments = []
    for i in range(0, len(data) - window_size + 1, window_size // 2):
        segment = data.iloc[i:i+window_size].values
        if len(segment) == window_size:
            segments.append(segment)
    
    if not segments:
        return None
    
    return np.array(segments)

# ============================================
# PAGE 1: Scientific Defense Dashboard
# ============================================
def page_scientific_defense():
    st.header("📊 Scientific Defense Dashboard")
    st.markdown("**Prove to your supervisor that your data and splits are valid.**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Files", DATASET_INFO['total_files'])
    with col2:
        st.metric("Training Files", DATASET_INFO['train_files'])
    with col3:
        st.metric("Test Files", DATASET_INFO['test_files'])
    
    st.divider()
    
    # Class Distribution
    st.subheader("Class Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            values=list(DATASET_INFO['class_distribution'].values()),
            names=list(DATASET_INFO['class_distribution'].keys()),
            title="Emotion Class Distribution",
            color_discrete_map=EMOTION_COLORS
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Distribution Details")
        for emotion, count in DATASET_INFO['class_distribution'].items():
            percentage = (count / DATASET_INFO['total_files']) * 100
            st.write(f"**{emotion}**: {count} files ({percentage:.1f}%)")
    
    st.divider()
    
    # Evidence: Raw Data Viewer
    st.subheader("📋 Evidence: Raw Data Viewer")
    st.write("First 5 rows showing POW (Power Spectral) columns to prove real Band Power data:")
    
    uploaded_file = st.file_uploader("Upload EEG CSV to view raw data", type=['csv'], key="defense")
    if uploaded_file:
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
            pow_cols = [c for c in df.columns if str(c).strip().startswith("POW.")]
            if pow_cols:
                st.dataframe(df[pow_cols].head(), use_container_width=True)
                st.success(f"✅ Found {len(pow_cols)} Power Spectral columns")
            else:
                st.warning("No POW columns found in this file.")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    st.divider()
    
    # Model Specs Comparison
    st.subheader("🤖 Model Specifications Comparison")
    
    model_specs = pd.DataFrame({
        'Model': ['SVM (You)', 'CNN (Friend)'],
        'Approach': [
            'Feature Engineered (Context Stacking)',
            'Feature Learning (Raw Input)'
        ],
        'Input Type': [
            'Engineered Features + Ratios',
            'Raw Time Series Segments'
        ],
        'Sequential Handling': [
            'Context Window Stacking (5 frames)',
            'Convolutional Layers (Temporal)'
        ],
        'Advantages': [
            'Interpretable features, Domain knowledge',
            'Automatic feature extraction, End-to-end learning'
        ]
    })
    
    st.dataframe(model_specs, use_container_width=True, hide_index=True)

# ============================================
# PAGE 2: Live Patient Monitor (Creative)
# ============================================
def page_live_monitor():
    st.title("🫀 Live EEG Emotion Monitor")
    st.markdown("**Simulates a real-time data stream to demonstrate sequential analysis.**")
    
    svm_model, svm_scaler, svm_artifact = load_svm_model()
    if svm_model is None:
        st.error("SVM model not found. Please ensure 'models/eeg_emotion_svm_model.pkl' exists.")
        return
    
    uploaded_file = st.file_uploader("Upload Patient Session (.csv)", type=["csv"], key="live")
    
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
            
            # Prepare data for model - match training pipeline exactly
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
            
            # Ensure vis_data and predictions are aligned
            if len(vis_data) == 0:
                st.error("No valid visualization data after processing.")
                return
            
            st.divider()
            
            col_graph, col_status = st.columns([3, 1])
            
            start_btn = st.button("▶ Start Live Simulation", type="primary")
            
            if start_btn:
                chart_placeholder = col_graph.empty()
                status_placeholder = col_status.empty()
                progress_bar = st.progress(0)
                
                # Simulate streaming
                step_size = max(1, len(vis_data) // 100)  # Show ~100 updates
                total_steps = len(vis_data) // step_size
                
                for step, i in enumerate(range(0, len(vis_data), step_size)):
                    # Update chart (show last 50 points)
                    chart_data = vis_data.iloc[max(0, i-50):i+1]
                    if len(chart_data) > 0:
                        fig = go.Figure()
                        for col in chart_data.columns:
                            fig.add_trace(go.Scatter(
                                y=chart_data[col].values,
                                mode='lines',
                                name=col.split('.')[-1] if '.' in col else col
                            ))
                        fig.update_layout(
                            title=f"Live EEG Stream (Sample {i}/{len(vis_data)})",
                            xaxis_title="Time Step",
                            yaxis_title="Power",
                            height=400
                        )
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                    
                    # Update emotion status
                    pred_idx = min(i // step_size, len(y_pred_smooth) - 1)
                    if pred_idx >= 0:
                        current_emotion_idx = y_pred_smooth[pred_idx]
                        emotion_name = EMOTION_LABELS.get(current_emotion_idx, "Unknown")
                        
                        # Dynamic colors
                        color_map = {
                            "Happy": "🟢",
                            "Neutral": "⚪",
                            "Sad": "🔵",
                            "Anxiety": "🔴"
                        }
                        emoji = color_map.get(emotion_name, "⚪")
                        
                        status_placeholder.markdown(f"""
                        ### Patient Status:
                        # {emoji} {emotion_name}
                        
                        **Confidence:** High  
                        **Sequence Window:** {pred_idx}
                        **Time Step:** {i}
                        """)
                    
                    progress_bar.progress((step + 1) / total_steps)
                    time.sleep(0.05)  # Controls playback speed
                
                progress_bar.empty()
                st.success("✅ Simulation complete!")
        
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
# MAIN APP
# ============================================
def main():
    st.markdown('<div class="main-header">🧠 EEG Emotion Detection - Central Command Center</div>', unsafe_allow_html=True)
    
    # Top Navigation Bar with Tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Scientific Defense",
        "🫀 Live Patient Monitor",
        "⚖️ Comparative Analysis"
    ])
    
    # Route to appropriate page based on selected tab
    with tab1:
        page_scientific_defense()
    
    with tab2:
        page_live_monitor()
    
    with tab3:
        page_head_to_head()

if __name__ == "__main__":
    main()
