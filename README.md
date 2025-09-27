# 🎵 Music Genre Classification with Deep Learning  

## Overview  
This project implements a deep neural network (DNN) in **Keras** to classify music tracks into genres using the **GTZAN dataset**.  
It extracts audio features (MFCCs, chroma, spectral centroid, tempo, etc.) and trains a Sequential DNN to achieve **89% accuracy**, outperforming traditional classifiers like SVM, KNN, Random Forest, and CNN.  

## Project Structure  
```
.
├── dataset_3_sec.csv        # Preprocessed features (3-sec clips)
├── dataset_30_sec.csv       # Preprocessed features (30-sec clips)
├── genres_original/         # Original audio samples (10 genres)
├── spectrograms_original/   # Spectrogram images (10 genres)
├── preprocessing.py         # Data preprocessing & feature extraction
├── model_utils.py           # Helper functions for model handling
├── train.py                 # Train the DNN model
├── evaluate.py              # Evaluate trained models
├── predict.py               # Predict genre for new audio files
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Installation  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/your-username/music-genre-classification.git
cd music-genre-classification
pip install -r requirements.txt
```

## Usage  

### 1. Preprocess Data  
If you want to regenerate datasets from audio files:  
```bash
python preprocessing.py
```

### 2. Train the Model  
```bash
python train.py
```
- Trains the model using `dataset_3_sec.csv` by default.  
- Saves trained weights for later use.  

### 3. Evaluate the Model  
```bash
python evaluate.py
```
- Evaluates accuracy, loss, and confusion matrix.  

### 4. Predict Genres for New Songs  
```bash
python predict.py --file path/to/audio.wav
```
- Extracts features and predicts genre using the trained model.  

## Dataset  
The project uses the **GTZAN dataset** with 10 genres:  
`blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock`  

- **1,000 audio samples** (30s each)  
- **Segmented 3s clips** for better training  
- Extracted features saved in `.csv` for faster experimentation  

## Model Architecture  
- Input (58 features)  
- Dense (256) + ReLU  
- Dense (128) + ReLU  
- Dense (64) + ReLU  
- Dense (10) + Softmax  
- Loss: Sparse Categorical Cross-Entropy  
- Optimizer: Adam  

## Results  

| Model          | Accuracy |  
|----------------|----------|  
| SVM            | 82%      |  
| KNN            | 76%      |  
| Random Forest  | 77%      |  
| CNN            | 80%      |  
| **DNN (this project)** | **89%** |  

✅ DNN model outperformed all traditional classifiers.  

## Future Work  
- Test alternative architectures (CNN, hybrid DNN+CNN)  
- Perform hyperparameter tuning & cross-validation  
- Expand dataset with more diverse audio tracks  
- Integrate into a **Spotify recommendation system**  
