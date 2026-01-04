PROJECT: EEG Brain Signal Analysis for Emotion Recognition
FILES: Jupyter Notebooks (.ipynb) for Deep Learning and Machine Learning Approaches

-------------------------------------------------------------------------
1. ENVIRONMENT SETUP
-------------------------------------------------------------------------
These notebooks are designed to run in Google Colab. To access the EEG 
dataset, the code mounts your personal Google Drive to the Colab runtime.

Required Code Block (found at the start of notebooks):
    from google.colab import drive
    drive.mount('/content/drive')

When you run this cell, you will be prompted to authorize access to your 
Google Drive.

-------------------------------------------------------------------------
2. DATASET PATH CONFIGURATION (IMPORTANT)
-------------------------------------------------------------------------
Please note that the two notebooks (Deep Learning vs. Machine Learning) 
were developed using different directory structures. 

Although the variable names and paths differ, both notebooks utilize the 
EXACT SAME dataset consisting of 32 EEG recording files (.csv).

Depending on which notebook you are running, you must update the path 
variable to point to the location where you have saved the dataset on 
your own Google Drive.

A. Deep Learning Notebook (CNN Approach)
   Variable Name: FOLDER_PATH
   Default Path in Code: '/content/drive/MyDrive/EEG_Data/Final Data/'
   Action: Change this string to match your specific Drive folder location.

B. Machine Learning Notebook (SVM Approach)
   Variable Name: DATA_DIR
   Default Path in Code: "/content/drive/MyDrive/Latest Emotion Data/Final Data"
   Action: Change this string to match your specific Drive folder location.

-------------------------------------------------------------------------
3. DATASET CONTENTS
-------------------------------------------------------------------------
The target folder should contain the 32 raw EEG recording files (.csv) 
collected via the Emotiv Flex Gel headset. 

Ensure all .csv files are present in the specified directory before 
running the preprocessing cells.