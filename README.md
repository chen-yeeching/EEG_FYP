# EEG Emotion Detection Dashboard

A Streamlit web application for visualizing and analyzing EEG data to detect emotions (Happy, Neutral, Sad, Anxiety).

## Features

### 🧠 Brain Heatmap Visualization
- Interactive topographic brain map showing activity across 32 electrodes
- Color-coded visualization (red/blue) based on activity levels
- Support for different frequency bands (Alpha, Beta, Gamma, Theta)
- Min/Max/Average statistics display

### 📊 Emotion Prediction
- Upload EEG CSV files and get real-time emotion predictions
- Prediction distribution visualization
- Confidence scores

### 📈 Statistics & Analysis
- Detailed statistics (Min, Max, Average, Std Dev) for each electrode
- Time series plots for specific electrodes and frequency bands
- Comparative analysis across all electrodes

### 🎭 Emotion Comparison
- Educational content about expected brain patterns for each emotion
- Visual comparison capabilities

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. The app will open in your browser automatically.

3. Navigate through different pages using the sidebar:
   - **Upload & Predict**: Upload EEG data and get emotion predictions
   - **Brain Heatmap**: Visualize brain activity as a heatmap
   - **Statistics & Analysis**: View detailed statistics and time series
   - **Emotion Comparison**: Learn about emotion patterns

## Data Format

The app expects EEG data in CSV format with:
- Power spectral features: `POW.{Electrode}.{FrequencyBand}` (e.g., `POW.F3.Alpha`)
- EEG raw data: `EEG.{Electrode}` (e.g., `EEG.F3`)
- Timestamp column for time-based analysis

Supported electrodes: F3, Fp1, AF3, FC5, F4, Fp2, AF4, FC6, F8, T7, T8, O1, O2, Oz, Cz, Fz, C4, Pz

## Model Information

- Model Type: SVM (Support Vector Machine)
- Emotions: Happy, Neutral, Sad, Anxiety
- Features: Power spectral features, ratios (Beta/Alpha, Theta/Beta), global frontal-occipital ratio

## Notes

- The model file (`emotion_svm_model.pkl`) must be in the same directory as `app.py`
- Sample data is automatically loaded if available
- For best results, use data preprocessed with the same pipeline as training

