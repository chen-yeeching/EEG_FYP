# EEG Emotion Detection Dashboard

A comprehensive Streamlit web application for visualizing and analyzing EEG data to detect emotions using both SVM and CNN machine learning models. The application supports real-time emotion prediction, brain activity visualization, and comparative analysis.

## Features

### 📊 Data Overview
- Dataset statistics and summary information
- Emotion class distribution visualization
- Feature descriptions and documentation
- Training/test split information

### 🧠 Brain Heatmap Visualization
- Interactive topographic brain map showing activity across 32 electrodes
- Real-time animated heatmap visualization
- Color-coded visualization (red/blue) based on activity levels
- Support for Alpha frequency band visualization
- Smooth interpolation for continuous heatmap display
- Standard 10-20 electrode positioning system

### 🫀 Live Monitor
- Real-time EEG data stream simulation
- Simultaneous SVM and CNN emotion prediction
- Live emotion status updates with visual indicators
- Side-by-side model comparison
- Interactive time series visualization
- Confidence scores and prediction distributions

### ⚖️ Comparative Analysis
- Head-to-head comparison of SVM vs CNN predictions
- Consensus meter showing model agreement
- Detailed prediction distributions for both models
- Confidence scoring for each prediction

## Installation

1. Clone this repository or download the project files.

2. Install required packages:
```bash
pip install -r requirements.txt
```

**Note:** TensorFlow is optional but recommended for CNN model functionality. If you only need SVM predictions, you can skip TensorFlow installation.

## Usage

1. Ensure model files are in the `models/` directory:
   - `eeg_emotion_svm_model.pkl` (required for SVM predictions)
   - `eeg_emotion_cnn_model.keras` (required for CNN predictions)
   - `eeg_emotion_cnn_metadata.json` (optional, contains model metadata)

2. Run the Streamlit app:
```bash
streamlit run app/app.py
```

Or if running from the app directory:
```bash
cd app
streamlit run app.py
```

3. The app will open in your browser automatically (typically at `http://localhost:8501`).

4. Navigate through different tabs:
   - **📊 Data Overview**: View dataset statistics and information
   - **🧠 Brain Heatmap**: Upload EEG data and visualize brain activity as an animated heatmap
   - **🫀 Live Monitor**: Upload EEG data and run real-time emotion prediction simulations
   - **⚖️ Comparative Analysis**: Compare SVM and CNN predictions side-by-side

## Data Format

The app expects EEG data in CSV format with the following structure:

### Required Columns:
- **Power Spectral Features**: `POW.{Electrode}.{FrequencyBand}` 
  - Example: `POW.Fp1.Alpha`, `POW.Cz.Beta`, `POW.O1.Theta`
- **Raw EEG Data** (for CNN): `EEG.{Electrode}`
  - Example: `EEG.Fp1`, `EEG.Cz`, `EEG.O1`
- **Timestamp**: A timestamp column for time-based analysis

### Supported Electrodes (32 channels):
Cz, Fz, Fp1, F7, F3, FC1, C3, FC5, FT9, T7, CP5, CP1, P3, P7, PO9, O1, Pz, Oz, O2, PO10, P8, P4, CP2, CP6, T8, FT10, FC6, C4, FC2, F4, F8, Fp2

### Supported Frequency Bands:
- Theta (4-8 Hz)
- Alpha (8-13 Hz)
- Beta (13-30 Hz)
- Gamma (30-100 Hz)

### CSV Format Notes:
- The app automatically detects the header row by searching for "POW." and "Timestamp"
- Data should be numeric and properly formatted
- Missing values are handled automatically

## Dataset

The project includes sample EEG data organized by emotion class:
- **Happy**: 8 CSV files
- **Neutral**: 8 CSV files
- **Sad**: 8 CSV files
- **Anxiety**: 8 CSV files

Total: 32 EEG recording sessions, split into 24 training files and 8 test files.

## Model Information

### SVM Model (Support Vector Machine)
- **Type**: Support Vector Machine with RBF kernel
- **Features**: Power spectral features with context window stacking (window size: 5)
- **Preprocessing**: RobustScaler normalization
- **Emotions**: Happy (0), Neutral (1), Sad (2), Anxiety (3)
- **Model File**: `models/eeg_emotion_svm_model.pkl`

### CNN Model (Convolutional Neural Network)
- **Type**: EEGNet with Channel Attention mechanism
- **Architecture**: Deep learning model for raw EEG time series
- **Input**: 128-sample windows with 32 channels
- **Preprocessing**: Z-score normalization per recording, sliding window segmentation
- **Loss Function**: Focal Loss (gamma=2.0, alpha=[1.0, 1.0, 1.3, 1.0])
- **Emotions**: Happy (0), Neutral (1), Sad (2), Anxiety (3)
- **Model File**: `models/eeg_emotion_cnn_model.keras`
- **Metadata**: `models/eeg_emotion_cnn_metadata.json`

### Model Performance (from metadata):
- **CNN Validation Accuracy**: ~72.4%
- **CNN Test Accuracy**: ~78.9%
- **CNN Macro F1-Score**: ~79.2%

## Project Structure

```
EEG_FYP/
├── app/
│   └── app.py              # Main Streamlit application
├── models/
│   ├── eeg_emotion_svm_model.pkl      # Trained SVM model
│   ├── eeg_emotion_cnn_model.keras    # Trained CNN model
│   └── eeg_emotion_cnn_metadata.json  # CNN model metadata
├── data/
│   ├── happy/              # Happy emotion EEG recordings (8 files)
│   ├── neutral/            # Neutral emotion EEG recordings (8 files)
│   ├── sad/                # Sad emotion EEG recordings (8 files)
│   └── anxiety/            # Anxiety emotion EEG recordings (8 files)
├── notebooks/
│   ├── EEG_CNN_Emotion_Recognition.ipynb  # CNN training notebook
│   └── EEG_SVM_Emotion_Recognition.ipynb  # SVM training notebook
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Technical Details

### Preprocessing Pipeline

**SVM Preprocessing:**
1. Extract power spectral features (POW columns)
2. Create context windows (sliding window stacking, window size: 5)
3. Apply RobustScaler normalization
4. Predict using trained SVM model
5. Apply median filter smoothing (kernel size: 5)

**CNN Preprocessing:**
1. Extract raw EEG channels (32 channels in standard order)
2. Z-score normalization per recording
3. Sliding window segmentation (window size: 128, step size: 64)
4. Predict using trained CNN model
5. Apply median filter smoothing (kernel size: 3)

### Visualization Features
- **Brain Heatmap**: Uses cubic interpolation for smooth visualization
- **Live Monitor**: Real-time streaming simulation with configurable update rate
- **Comparative Analysis**: Side-by-side model comparison with consensus detection

## Dependencies

The project requires the following Python packages (see `requirements.txt` for versions):
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning (SVM, preprocessing)
- **plotly**: Interactive visualizations
- **joblib**: Model serialization
- **scipy**: Scientific computing (signal processing, interpolation)
- **tensorflow**: Deep learning framework (for CNN model)

## Troubleshooting

### Model Not Found Errors
- Ensure model files are in the `models/` directory relative to the project root
- Check file names match exactly: `eeg_emotion_svm_model.pkl` and `eeg_emotion_cnn_model.keras`
- Verify the models directory structure matches the project structure

### TensorFlow Issues
- If CNN model fails to load, ensure TensorFlow is properly installed: `pip install tensorflow>=2.13.0`
- The app will still function with SVM-only if TensorFlow is not available
- For GPU support, install `tensorflow-gpu` instead

### Data Format Issues
- Ensure your CSV contains `POW.` columns for SVM predictions
- For CNN predictions, ensure `EEG.` columns are present with all 32 required channels
- Check that the header row contains both "POW." and "Timestamp" for automatic detection
- Verify electrode names match the supported list exactly

### Performance Issues
- Large CSV files may take time to process
- The live heatmap animation uses optimized frame rates for smooth playback
- Consider reducing data size for faster processing during development
- For very large files, consider preprocessing or sampling the data

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility (Python 3.8+ recommended)
- If using a virtual environment, ensure it's activated

## Notes

- The models were trained on specific EEG data formats and preprocessing pipelines
- For best results, use data preprocessed with the same pipeline as training
- The app supports both models running simultaneously for comparison
- CNN model requires TensorFlow, while SVM only requires scikit-learn
- Sample data is provided in the `data/` directory for testing

## Development

### Training Models
Model training notebooks are available in the `notebooks/` directory:
- `EEG_CNN_Emotion_Recognition.ipynb`: CNN model training
- `EEG_SVM_Emotion_Recognition.ipynb`: SVM model training

These notebooks were originally developed in Google Colab and can be adapted for local execution.
