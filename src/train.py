import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, Dict, Any

from src.preprocessing import clean_text, encode_labels, fit_tokenizer_and_pad, save_artifacts


def build_bilstm_model(
    vocab_size: int = 5000,
    embedding_dim: int = 64,
    maxlen: int = 100,
    learning_rate: float = 0.001
) -> Sequential:
    """
    Builds and compiles the Bidirectional LSTM Model architecture.
    """
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=maxlen,
            name="embedding_layer"
        ),
        SpatialDropout1D(0.2, name="spatial_dropout"),
        Bidirectional(
            LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
            name="bilstm_layer"
        ),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01), name="dense_regularized"),
        Dropout(0.3, name="dense_dropout"),
        Dense(1, activation='sigmoid', name="output_layer")
    ])

    optimizer = Adam(learning_rate=learning_rate)
    
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=metrics
    )

    print("\n--- Model Architecture Summary ---")
    model.summary()
    return model


def train_pipeline(
    data_path: str = "data/spam.csv",
    models_dir: str = "models",
    epochs: int = 15,
    batch_size: int = 32
) -> Tuple[Sequential, tf.keras.callbacks.History, np.ndarray, np.ndarray]:
    """
    Complete training pipeline: loads data, cleans text, tokenizes, splits stratified train/test,
    computes class weights, trains Bi-LSTM with callbacks, and saves model & tokenizer artifacts.
    """
    if not os.path.exists(data_path):
        from download_dataset import download_or_generate_dataset
        download_or_generate_dataset()

    print(f"[INFO] Loading dataset from {data_path}...")
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except Exception:
        df = pd.read_csv(data_path, encoding='latin-1')

    # Detect label and text columns
    if 'v1' in df.columns and 'v2' in df.columns:
        label_col, text_col = 'v1', 'v2'
    elif 'label' in df.columns and 'text' in df.columns:
        label_col, text_col = 'label', 'text'
    else:
        label_col, text_col = df.columns[0], df.columns[1]

    # Preprocess text and labels
    print("[INFO] Cleaning text dataset...")
    df['cleaned_text'] = df[text_col].apply(clean_text)
    
    # Filter out empty texts after cleaning
    df = df[df['cleaned_text'].str.strip() != ''].reset_index(drop=True)
    
    labels = encode_labels(df[label_col])
    texts = df['cleaned_text'].tolist()

    # Fit Tokenizer and Pad Sequences
    print("[INFO] Tokenizing and padding sequences...")
    tokenizer, X_padded, config = fit_tokenizer_and_pad(texts)
    
    # Save preprocessing artifacts
    save_artifacts(tokenizer, config, models_dir=models_dir)

    # Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"[INFO] Dataset split: Train size = {len(X_train)}, Test size = {len(X_test)}")

    # Compute Class Weights for Imbalance Handling
    unique_classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
    class_weight_dict = dict(zip(unique_classes, weights))
    print(f"[INFO] Computed Class Weights: {class_weight_dict}")

    # Build Model
    model = build_bilstm_model(
        vocab_size=config['vocab_size'],
        embedding_dim=64,
        maxlen=config['maxlen']
    )

    # Callbacks setup
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "spam_lstm_model.keras")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    # Train Model
    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model explicitly
    try:
        model.save(model_path)
        print(f"[SUCCESS] Final model saved successfully to {model_path}")
    except Exception as e:
        alt_path = os.path.join(models_dir, "spam_lstm_model.h5")
        model.save(alt_path)
        print(f"[SUCCESS] Saved model using HDF5 format to {alt_path} (error: {e})")

    return model, history, X_test, y_test


if __name__ == "__main__":
    train_pipeline()
