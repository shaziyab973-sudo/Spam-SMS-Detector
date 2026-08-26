import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re
import pickle
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Dict, Any, List, Optional, Tuple

from src.preprocessing import clean_text

# High-risk spam trigger keywords for token analysis
SPAM_TRIGGER_WORDS = {
    'win', 'winner', 'cash', 'prize', 'urgent', 'claim', 'free', 'money', 'guaranteed',
    'congratulations', 'loan', 'credit', 'click', 'bank', 'account', 'verify', 'password',
    'urgent', 'alert', 'notice', 'call', 'txt', 'bonus', 'offer', 'discount', 'reward',
    'nokia', 'customer', 'service', 'selected', 'awarded', 'subscribers', 'entry', 'competition',
    'http_url', 'email_address', 'currency_token', 'http', 'https', 'bit', 'ly'
}


class SpamClassifier:
    """
    Production Inference Engine for Spam SMS Detection.
    Loads saved Bi-LSTM Keras model and tokenizer safely, validates input,
    computes spam probability, assigns risk level, and extracts suspicious token highlights.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model = None
        self.tokenizer = None
        self.config = None
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads trained model, tokenizer pickle, and config JSON with robust try-except error handling."""
        model_keras = os.path.join(self.models_dir, "spam_lstm_model.keras")
        model_h5 = os.path.join(self.models_dir, "spam_lstm_model.h5")
        tokenizer_path = os.path.join(self.models_dir, "tokenizer.pickle")
        config_path = os.path.join(self.models_dir, "config.json")

        try:
            # Load Keras Model
            if os.path.exists(model_keras):
                self.model = load_model(model_keras)
                print(f"[INFO] SpamClassifier loaded model from {model_keras}")
            elif os.path.exists(model_h5):
                self.model = load_model(model_h5)
                print(f"[INFO] SpamClassifier loaded model from {model_h5}")
            else:
                raise FileNotFoundError("Model file not found in 'models/' directory.")

            # Load Tokenizer
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, "rb") as f:
                    self.tokenizer = pickle.load(f)
                print(f"[INFO] SpamClassifier loaded tokenizer from {tokenizer_path}")
            else:
                raise FileNotFoundError("Tokenizer pickle file not found in 'models/' directory.")

            # Load Config
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            else:
                self.config = {"maxlen": 100, "padding": "post", "truncating": "post"}

            self.is_loaded = True

        except Exception as e:
            print(f"[ERROR] Failed to initialize SpamClassifier: {e}")
            self.is_loaded = False

    def validate_input(self, text: str) -> Tuple[bool, str]:
        """
        Validates user input string.
        Checks for empty strings, whitespace-only strings, and extreme length.
        """
        if not isinstance(text, str) or not text.strip():
            return False, "Input text cannot be empty."

        if len(text) > 2000:
            return False, "Input exceeds maximum allowed length of 2000 characters."

        return True, "Valid input"

    def analyze_tokens(self, original_text: str, cleaned_text: str) -> List[Dict[str, Any]]:
        """
        Analyzes individual tokens in the message and identifies suspicious spam trigger words.
        Returns token list with flags and risk scores.
        """
        words = re.findall(r'\b\w+\b', original_text.lower())
        token_analysis = []

        for word in words:
            is_suspicious = word in SPAM_TRIGGER_WORDS
            token_analysis.append({
                "token": word,
                "is_suspicious": is_suspicious,
                "reason": "Spam trigger keyword" if is_suspicious else "Normal text"
            })

        return token_analysis

    def predict(self, text: str, threshold: float = 0.90) -> Dict[str, Any]:
        """
        Predicts whether an SMS message is Spam or Ham.

        Args:
            text (str): Input SMS text string.
            threshold (float): Decision threshold for classification (default 0.90).

        Returns:
            Dict containing label ('Spam'/'Ham'), confidence percentage, raw_probability,
            risk_level ('Low'/'Medium'/'High'), cleaned text, and token highlights.
        """
        if not self.is_loaded:
            return {
                "error": "Model artifacts are not loaded. Please train the model first.",
                "label": "Unknown",
                "raw_probability": 0.0,
                "confidence": "0.0%",
                "risk_level": "Unknown",
                "cleaned_text": "",
                "token_highlights": []
            }

        # Input Validation
        is_valid, err_msg = self.validate_input(text)
        if not is_valid:
            return {
                "error": err_msg,
                "label": "Invalid",
                "raw_probability": 0.0,
                "confidence": "0.0%",
                "risk_level": "N/A",
                "cleaned_text": "",
                "token_highlights": []
            }

        cleaned = clean_text(text)
        
        # Fallback if cleaning removed everything
        if not cleaned.strip():
            cleaned = text.lower()

        # Tokenize and Pad Sequence
        maxlen = self.config.get("maxlen", 100)
        padding = self.config.get("padding", "post")
        truncating = self.config.get("truncating", "post")

        sequence = self.tokenizer.texts_to_sequences([cleaned])
        padded_seq = pad_sequences(sequence, maxlen=maxlen, padding=padding, truncating=truncating)

        # Model Inference
        raw_prob = float(self.model.predict(padded_seq, verbose=0)[0][0])
        is_spam = raw_prob >= threshold

        # Calculate Confidence Score (relative to classification decision)
        if is_spam:
            confidence = raw_prob * 100
        else:
            confidence = (1 - raw_prob) * 100

        # Determine Risk Level based on probability
        if raw_prob >= 0.75:
            risk_level = "High"
        elif raw_prob >= 0.40:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Token Analysis
        token_highlights = self.analyze_tokens(text, cleaned)

        return {
            "label": "Spam" if is_spam else "Ham",
            "confidence": f"{confidence:.2f}%",
            "confidence_score": round(confidence, 2),
            "raw_probability": round(raw_prob, 4),
            "risk_level": risk_level,
            "threshold_used": threshold,
            "cleaned_text": cleaned,
            "token_highlights": token_highlights,
            "error": None
        }
