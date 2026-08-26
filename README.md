# 🛡️ Spam SMS Detection Engine

A complete, production-ready Deep Learning NLP application for **Spam SMS Detection** built with **TensorFlow / Keras**, **NLTK**, and **Streamlit**.

---

## 📌 Project Overview & Specifications

- **Task**: Binary Text Classification (`0` = Legit / Ham, `1` = Spam)
- **Model Architecture**: Bidirectional LSTM with Word Embeddings
  - Embedding Layer (input_dim=5000, output_dim=64, input_length=100)
  - SpatialDropout1D (0.2)
  - Bidirectional LSTM (64 units, dropout=0.2, recurrent_dropout=0.2)
  - Dense Regularized Layer (32 units, ReLU activation, L2=0.01) + Dropout(0.3)
  - Output Dense Layer (1 unit, Sigmoid activation)
- **Dataset**: SMS Spam Collection Dataset (`v1` label, `v2` text message)
- **Preprocessing Pipeline**: Lowercasing, URL/HTML stripping, NLTK Stopword removal, WordNet Lemmatization, Keras Sequence Padding (`maxlen=100`)
- **Web UI**: Interactive Streamlit Dashboard with dynamic thresholding, token analysis, confidence gauge bar, and CSV batch processing.

---

## 📁 Repository Structure

```
c:/Users/user/Desktop/Spam SMS Detection/
├── data/
│   ├── spam.csv                 # Clean SMS dataset
│   └── sample_batch.csv         # Sample batch file for testing CSV uploads
├── models/
│   ├── spam_lstm_model.keras    # Trained Bi-LSTM model artifact
│   ├── tokenizer.pickle         # Fitted Keras Tokenizer
│   └── config.json              # Model & preprocessing configuration
├── reports/
│   ├── metrics.json             # Test set metrics (Accuracy, Precision, Recall, F1, AUC)
│   ├── confusion_matrix.png     # Heatmap visualization of confusion matrix
│   └── training_history.png     # Subplot curves for Loss and Accuracy
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Text cleaning, lemmatization, tokenization & sequence padding
│   ├── train.py                 # Bi-LSTM model compilation, class weights, callbacks & saving
│   ├── evaluate.py              # Performance metric calculations & report/chart generation
│   └── inference.py             # SpamClassifier class, thresholding, risk scoring & token highlighting
├── app.py                       # Interactive Streamlit Web Application
├── download_dataset.py          # Automatic dataset fetcher script
├── run_pipeline.py              # Full pipeline runner (Download -> Train -> Evaluate)
├── requirements.txt             # Dependency specification
└── README.md                    # Documentation & setup instructions
```

---

## ⚡ Quickstart & Execution Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete ML Pipeline (Dataset -> Training -> Evaluation)
```bash
python run_pipeline.py
```

### 3. Launch Streamlit Web UI
```bash
streamlit run app.py
```

---

## 🧪 Testing Individual Modules

- **Dataset Downloader**:
  ```bash
  python download_dataset.py
  ```
- **Model Training**:
  ```bash
  python src/train.py
  ```
- **Inference Engine Direct Test**:
  ```python
  from src.inference import SpamClassifier

  classifier = SpamClassifier()
  result = classifier.predict("WINNER! You have won $1000 cash prize. Dial 1-800-555-0199 to claim!")
  print(result)
  ```
