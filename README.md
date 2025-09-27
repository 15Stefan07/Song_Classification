# Spotify Song Genre Classification 🎶

A deep learning approach for **music genre classification** using the **GTZAN dataset**.  
This project uses a **Dense Neural Network (DNN)** in Keras/TensorFlow to classify tracks into 10 genres, achieving up to **89% accuracy**.

---

## 📖 Overview
Music streaming platforms such as Spotify rely heavily on recommendation systems that require accurate **genre classification**.  
This project investigates several approaches (SVM, KNN, Random Forest, CNN) and compares them with a custom **Sequential DNN** model.  
Our best model significantly outperforms classical methods.

---

## 📂 Repository Contents
```
song_genre_classification_GTZAN_dataset.ipynb  # Main Jupyter notebook
PSV_article_Stirbu_Stefan.docx                 # Full project report (paper)
```

---

## 🛠️ Requirements
- Python 3.8+
- TensorFlow / Keras
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- Librosa

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Open the Notebook
```bash
jupyter notebook song_genre_classification_GTZAN_dataset.ipynb
```

### 2. Run Workflow
The notebook is structured into the following sections:
1. **Data loading & preprocessing** (GTZAN dataset CSV files and audio samples)  
2. **Feature extraction** (STFT, RMS, Spectral features, MFCCs, etc.)  
3. **Model definition** (Sequential DNN with Dense layers + ReLU & Softmax)  
4. **Training & validation** (100 epochs, batch size 256)  
5. **Evaluation** (accuracy, loss, confusion matrix, test predictions)  

### 3. Predict New Songs
The notebook includes utilities to:
- Extract features from user-provided audio files.  
- Classify them into one of the 10 genres.  

---

## 📊 Results
| Model          | Accuracy |
|----------------|----------|
| SVM            | 82%      |
| KNN            | 76%      |
| Random Forest  | 77%      |
| CNN            | 80%      |
| **DNN (ours)** | **89%**  |

- Achieved **89% test accuracy** on 3-second GTZAN clips.  
- Accuracy drops to ~74% on 30-second samples due to dataset variability.  

---

## 🔮 Future Work
- Experiment with CNN/RNN architectures.  
- Use cross-validation and hyperparameter tuning.  
- Incorporate transfer learning with pre-trained audio models.  
- Extend dataset for better generalization.  

---

## 📜 License
MIT License  

---

## 📚 References
- Tzanetakis, G., & Cook, P. (2002). *Musical genre classification of audio signals*.  
- Sahil Poonia et al. (2022). *Music Genre Classification using Machine Learning: A Comparative Study*.  
- [Librosa Documentation](https://librosa.org/)  

---

👨‍💻 Author: **Știrbu Ștefan-Mihai**
