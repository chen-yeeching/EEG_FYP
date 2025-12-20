import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="EEG Emotion Detection Dashboard",
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

# Standard 10-20 system electrode positions (simplified 2D projection)
ELECTRODE_POSITIONS = {
    'Fp1': (-0.3, 0.8), 'Fp2': (0.3, 0.8),
    'F7': (-0.6, 0.5), 'F3': (-0.3, 0.5), 'Fz': (0, 0.5), 'F4': (0.3, 0.5), 'F8': (0.6, 0.5),
    'T7': (-0.7, 0), 'C3': (-0.3, 0), 'Cz': (0, 0), 'C4': (0.3, 0), 'T8': (0.7, 0),
    'P7': (-0.6, -0.5), 'P3': (-0.3, -0.5), 'Pz': (0, -0.5), 'P4': (0.3, -0.5), 'P8': (0.6, -0.5),
    'O1': (-0.3, -0.8), 'Oz': (0, -0.8), 'O2': (0.3, -0.8),
    'AF3': (-0.2, 0.65), 'AF4': (0.2, 0.65),
    'FC5': (-0.5, 0.25), 'FC1': (-0.15, 0.25), 'FC2': (0.15, 0.25), 'FC6': (0.5, 0.25),
    'CP5': (-0.5, -0.25), 'CP1': (-0.15, -0.25), 'CP2': (0.15, -0.25), 'CP6': (0.5, -0.25),
    'FT9': (-0.75, 0.15), 'FT10': (0.75, 0.15),
    'PO9': (-0.5, -0.7), 'PO10': (0.5, -0.7)
}

TARGET_SENSORS = ['F3', 'Fp1', 'AF3', 'FC5', 'F4', 'Fp2', 'AF4', 'FC6', 
                  'F8', 'T7', 'T8', 'O1', 'O2', 'Oz', 'Cz', 'Fz', 'C4', 'Pz']

ROLLING_WINDOW_SIZE = 128

@st.cache_data
def load_model():
    """Load the trained SVM model"""
    try:
        model = joblib.load('emotion_svm_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

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
    # Feature Engineering
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

def create_brain_heatmap(electrode_values, title="Brain Activity Heatmap", use_zscore=True, color_scale='RdBu_r'):
    """Create a brain heatmap visualization.

    If use_zscore is True, values are converted to z-scores so that:
    - Red = stronger activation (positive z)
    - Blue = weaker activation (negative z)
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
    colorbar_title = "Activity Level"
    
    # Optionally standardise to z-scores so scale is symmetric around 0
    if use_zscore and len(values) > 1:
        mean_val = float(np.mean(values))
        std_val = float(np.std(values)) or 1e-6
        z_values = (values - mean_val) / std_val
        values_for_color = z_values
        # Symmetric colour range so 0 is neutral, red = strong, blue = weak
        max_abs = float(np.max(np.abs(z_values)))
        cmin, cmax = -max_abs, max_abs
        colorbar_title = "Z-score"
    else:
        values_for_color = values
        cmin, cmax = float(np.min(values)), float(np.max(values))
    
    # Create heatmap using scatter plot with color mapping
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode='markers+text',
        marker=dict(
            size=30,
            color=values_for_color,
            colorscale=color_scale,
            showscale=True,
            colorbar=dict(title=colorbar_title),
            cmin=cmin,
            cmax=cmax
        ),
        text=[label.split('<br>')[0] for label in labels],
        textposition="middle center",
        textfont=dict(size=8, color='white'),
        hovertemplate='%{text}<br>Value: %{marker.color:.2f}<extra></extra>'
    ))
    
    # Draw head outline (circle)
    theta = np.linspace(0, 2*np.pi, 100)
    head_x = np.cos(theta)
    head_y = np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=head_x,
        y=head_y,
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add nose
    nose_x = [0, 0.1, 0, -0.1, 0]
    nose_y = [1, 0.95, 0.9, 0.95, 1]
    fig.add_trace(go.Scatter(
        x=nose_x,
        y=nose_y,
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),
        height=600,
        showlegend=False,
        plot_bgcolor='white'
    )
    
    return fig

def extract_electrode_data(df, frequency_band='Alpha', metric='mean'):
    """Extract electrode values for a specific frequency band.

    metric: 'mean', 'max', or 'min'
    """
    electrode_values = {}
    metric = metric.lower()
    
    for sensor in TARGET_SENSORS:
        col_name = f"POW.{sensor}.{frequency_band}"
        series = None
        if col_name in df.columns:
            series = df[col_name]
        else:
            # Try alternative column names
            for col in df.columns:
                if sensor in col and frequency_band in col and 'POW' in col:
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

def get_statistics(df):
    """Calculate min, max, average for each electrode"""
    stats = {}
    
    for sensor in TARGET_SENSORS:
        sensor_data = {}
        for band in ['Theta', 'Alpha', 'BetaL', 'BetaH', 'Gamma']:
            col_name = f"POW.{sensor}.{band}"
            if col_name in df.columns:
                sensor_data[band] = {
                    'min': df[col_name].min(),
                    'max': df[col_name].max(),
                    'avg': df[col_name].mean(),
                    'std': df[col_name].std()
                }
        if sensor_data:
            stats[sensor] = sensor_data
    
    return stats

def main():
    st.markdown('<div class="main-header">🧠 EEG Emotion Detection Dashboard</div>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Model not found. Please ensure 'emotion_svm_model.pkl' is in the directory.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", 
                           ["📊 Upload & Predict", "🧠 Brain Heatmap", "📈 Statistics & Analysis", "🎭 Emotion Comparison"])
    
    # Load sample data if available
    sample_data = None
    try:
        sample_data = pd.read_csv('Emotion - Happy_FLEX2_278540_2025.11.24T16.50.07+08.00.md.bp.csv', header=1)
    except:
        pass
    
    if page == "📊 Upload & Predict":
        st.header("Upload EEG Data & Predict Emotion")
        
        uploaded_file = st.file_uploader("Upload EEG CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, header=1)
                st.success(f"File loaded successfully! Shape: {df.shape}")
                
                # Preprocess
                with st.spinner("Preprocessing data..."):
                    df_processed, feature_cols = preprocess_data(df)
                    
                    if df_processed is None or len(df_processed) == 0:
                        st.error("Could not extract features. Please check data format.")
                        return
                
                # Predict
                with st.spinner("Predicting emotion..."):
                    # Note: In production, you'd need to save/load the scaler too
                    # For now, we'll use the raw features (model expects scaled data)
                    X = df_processed.values
                    
                    # Simple scaling (ideally use the same scaler from training)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Predict on all samples
                    predictions = model.predict(X_scaled)
                    prediction_probs = model.decision_function(X_scaled) if hasattr(model, 'decision_function') else None
                    
                    # Get most common prediction
                    unique, counts = np.unique(predictions, return_counts=True)
                    most_common_idx = unique[np.argmax(counts)]
                    predicted_emotion = EMOTION_LABELS[most_common_idx]
                    confidence = (counts[np.argmax(counts)] / len(predictions)) * 100
                
                # Display prediction
                emotion_class = predicted_emotion.lower()
                st.markdown(f"""
                    <div class="emotion-box {emotion_class}">
                        Predicted Emotion: {predicted_emotion}<br>
                        Confidence: {confidence:.1f}%
                    </div>
                """, unsafe_allow_html=True)
                
                # Prediction distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    emotion_counts = {EMOTION_LABELS[k]: int(v) for k, v in zip(unique, counts)}
                    fig_pie = px.pie(
                        values=list(emotion_counts.values()),
                        names=list(emotion_counts.keys()),
                        title="Prediction Distribution",
                        color_discrete_map=EMOTION_COLORS
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    st.subheader("Prediction Statistics")
                    for emotion, count in emotion_counts.items():
                        percentage = (count / len(predictions)) * 100
                        st.write(f"**{emotion}**: {count} samples ({percentage:.1f}%)")
                
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.exception(e)
        
        elif sample_data is not None:
            st.info("💡 Tip: Upload a CSV file to predict emotions, or use the sample data below.")
            if st.button("Use Sample Data (Happy Emotion)"):
                st.session_state['use_sample'] = True
    
    elif page == "🧠 Brain Heatmap":
        st.header("Brain Activity Heatmap")
        
        # Data source selection
        data_source = st.radio("Data Source", ["Upload File", "Use Sample Data"], horizontal=True)
        
        df = None
        if data_source == "Upload File":
            uploaded_file = st.file_uploader("Upload EEG CSV", type=['csv'], key="heatmap")
            if uploaded_file:
                df = pd.read_csv(uploaded_file, header=1)
        else:
            if sample_data is not None:
                df = sample_data.copy()
            else:
                st.warning("Sample data not available. Please upload a file.")
        
        if df is not None:
            # Frequency band selection
            frequency_band = st.selectbox("Select Frequency Band", 
                                         ["Alpha", "BetaL", "BetaH", "Gamma", "Theta"],
                                         index=0)
            
            # Metric selection for min / max / average
            metric_label = st.selectbox(
                "Select Metric (what to summarise over the time window)",
                ["Average (mean)", "Maximum", "Minimum"],
                index=0,
            )
            if "Average" in metric_label:
                metric = "mean"
            elif "Maximum" in metric_label:
                metric = "max"
            else:
                metric = "min"
            
            # Time window selection
            if 'Timestamp' in df.columns:
                time_col = 'Timestamp'
            else:
                time_col = [c for c in df.columns if 'timestamp' in c.lower()][0] if any('timestamp' in c.lower() for c in df.columns) else None
            
            if time_col:
                time_range = st.slider("Select Time Window", 
                                      min_value=0, 
                                      max_value=len(df)-1,
                                      value=(0, min(1000, len(df)-1)),
                                      step=100)
                df_window = df.iloc[time_range[0]:time_range[1]]
            else:
                df_window = df.iloc[:1000] if len(df) > 1000 else df
            
            # Extract electrode data
            electrode_values = extract_electrode_data(df_window, frequency_band, metric=metric)
            
            if electrode_values:
                # Create heatmap
                fig = create_brain_heatmap(
                    electrode_values, 
                    title=f"Brain Activity - {frequency_band} Band ({metric_label})",
                    use_zscore=True,
                    color_scale='RdBu_r'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Min Activity", f"{min(electrode_values.values()):.2f}")
                with col2:
                    st.metric("Max Activity", f"{max(electrode_values.values()):.2f}")
                with col3:
                    st.metric("Average Activity", f"{np.mean(list(electrode_values.values())):.2f}")
                
                # Electrode values table
                with st.expander("View Electrode Values"):
                    df_electrodes = pd.DataFrame([
                        {"Electrode": k, "Value": v} 
                        for k, v in sorted(electrode_values.items())
                    ])
                    st.dataframe(df_electrodes, use_container_width=True)

                # Optional 4-band comparison to mimic 2x2 brain maps
                show_comparison = st.checkbox(
                    "Show 4-band comparison (Theta, Alpha, BetaL, Gamma)",
                    value=False,
                )
                if show_comparison:
                    st.subheader("4-Band Comparison (z-score, red=strong, blue=weak)")
                    bands = ["Theta", "Alpha", "BetaL", "Gamma"]
                    cols_top = st.columns(2)
                    cols_bottom = st.columns(2)
                    all_cols = [*cols_top, *cols_bottom]

                    for band, col in zip(bands, all_cols):
                        with col:
                            band_values = extract_electrode_data(df_window, band, metric=metric)
                            if band_values:
                                band_fig = create_brain_heatmap(
                                    band_values,
                                    title=f"{band} ({metric_label})",
                                    use_zscore=True,
                                    color_scale="RdBu_r",
                                )
                                if band_fig is not None:
                                    st.plotly_chart(band_fig, use_container_width=True)
            else:
                st.warning("Could not extract electrode data. Please check data format.")
    
    elif page == "📈 Statistics & Analysis":
        st.header("Statistics & Analysis")
        
        data_source = st.radio("Data Source", ["Upload File", "Use Sample Data"], horizontal=True, key="stats")
        
        df = None
        if data_source == "Upload File":
            uploaded_file = st.file_uploader("Upload EEG CSV", type=['csv'], key="stats_file")
            if uploaded_file:
                df = pd.read_csv(uploaded_file, header=1)
        else:
            if sample_data is not None:
                df = sample_data.copy()
            else:
                st.warning("Sample data not available. Please upload a file.")
        
        if df is not None:
            # Get statistics
            stats = get_statistics(df)
            
            if stats:
                # Select electrode and frequency band
                col1, col2 = st.columns(2)
                with col1:
                    selected_sensor = st.selectbox("Select Electrode", list(stats.keys()))
                with col2:
                    selected_band = st.selectbox("Select Frequency Band", 
                                                list(stats[selected_sensor].keys()) if selected_sensor in stats else [])
                
                if selected_sensor in stats and selected_band in stats[selected_sensor]:
                    sensor_stats = stats[selected_sensor][selected_band]
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Minimum", f"{sensor_stats['min']:.2f}")
                    with col2:
                        st.metric("Maximum", f"{sensor_stats['max']:.2f}")
                    with col3:
                        st.metric("Average", f"{sensor_stats['avg']:.2f}")
                    with col4:
                        st.metric("Std Deviation", f"{sensor_stats['std']:.2f}")
                    
                    # Time series plot
                    col_name = f"POW.{selected_sensor}.{selected_band}"
                    if col_name in df.columns:
                        fig = px.line(
                            df, 
                            y=col_name,
                            title=f"{selected_sensor} - {selected_band} Band Over Time"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                # All electrodes comparison
                st.subheader("All Electrodes - Average Power by Frequency Band")
                
                comparison_data = []
                for sensor, bands in stats.items():
                    for band, values in bands.items():
                        comparison_data.append({
                            'Electrode': sensor,
                            'Frequency Band': band,
                            'Average Power': values['avg']
                        })
                
                if comparison_data:
                    df_comparison = pd.DataFrame(comparison_data)
                    fig = px.bar(
                        df_comparison,
                        x='Electrode',
                        y='Average Power',
                        color='Frequency Band',
                        barmode='group',
                        title="Average Power Across All Electrodes"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not extract statistics. Please check data format.")
    
    elif page == "🎭 Emotion Comparison":
        st.header("Emotion Comparison")
        st.info("This page compares brain activity patterns across different emotions.")
        st.write("Upload multiple files or use sample data to compare different emotional states.")
        
        # This would require multiple files - for now show a conceptual view
        st.subheader("Expected Patterns by Emotion")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 😊 Happy")
            st.write("- Increased activity in **left frontal** regions (F3, Fp1)")
            st.write("- Higher **Alpha** and **Beta** in frontal areas")
            st.write("- Balanced frontal-occipital ratio")
        
        with col2:
            st.markdown("### 😢 Sad")
            st.write("- Increased activity in **right frontal** regions (F4, Fp2)")
            st.write("- Higher **Theta** activity")
            st.write("- Asymmetric frontal activity")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### 😰 Anxiety")
            st.write("- Increased activity in **temporal** regions (T7, T8)")
            st.write("- Higher **Beta** and **Gamma** activity")
            st.write("- Frontal asymmetry")
        
        with col4:
            st.markdown("### 😐 Neutral")
            st.write("- Balanced activity across regions")
            st.write("- Higher **Alpha** in **occipital** areas (O1, O2, Oz)")
            st.write("- Stable frontal-occipital ratio")
        
        # Visualization placeholder
        if sample_data is not None:
            st.subheader("Sample Visualization (Happy Emotion)")
            electrode_values = extract_electrode_data(sample_data, 'Alpha', metric='mean')
            if electrode_values:
                fig = create_brain_heatmap(
                    electrode_values,
                    title="Sample: Happy Emotion - Alpha Band (z-score)",
                    use_zscore=True,
                    color_scale='RdBu_r'
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

