import os
import re
import pickle
import json
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
import nltk

# Ensure NLTK resources are available
def setup_nltk():
    """Downloads necessary NLTK data quietly if missing."""
    for resource in ['stopwords', 'punkt', 'wordnet', 'omw-1.4']:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                print(f"[INFO] Downloading NLTK resource: {resource}...")
                nltk.download(resource, quiet=True)

setup_nltk()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Module Constants
VOCAB_SIZE = 5000
MAX_LEN = 100
OOV_TOKEN = "<OOV>"
PADDING_TYPE = 'post'
TRUNCATING_TYPE = 'post'

LEMMATIZER = WordNetLemmatizer()
try:
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = set()


def clean_text(text: str) -> str:
    """
    Cleans raw SMS text while preserving sequence context for Bi-LSTM neural network:
    - Lowercase conversion
    - URL replacement (replaces web links with 'http_url' token)
    - Email replacement (replaces emails with 'email_address' token)
    - Currency symbol replacement ('$', '£', '€' with 'currency_token')
    - HTML tag removal
    - Special character stripping (preserves alphanumeric words, numbers, and tokens)
    - Retains full sequence grammar and syntax for accurate RNN pattern modeling
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Convert to lower case
    text = text.lower()

    # Replace URLs with special token
    text = re.sub(r'https?://\S+|www\.\S+', ' http_url ', text)

    # Replace email addresses with special token
    text = re.sub(r'\S+@\S+', ' email_address ', text)

    # Replace currency symbols with special token
    text = re.sub(r'[\$\£\€]', ' currency_token ', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Remove non-alphanumeric characters (keep alphanumeric, underscores, and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s_]', ' ', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def encode_labels(labels: pd.Series) -> np.ndarray:
    """
    Encodes categorical labels to binary integer targets.
    'ham' -> 0, 'spam' -> 1
    """
    mapping = {'ham': 0, 'spam': 1, 0: 0, 1: 1, '0': 0, '1': 1}
    return labels.map(lambda x: mapping.get(str(x).lower(), 0)).values.astype(np.int32)


def fit_tokenizer_and_pad(
    texts: List[str],
    vocab_size: int = VOCAB_SIZE,
    maxlen: int = MAX_LEN,
    oov_token: str = OOV_TOKEN,
    padding: str = PADDING_TYPE,
    truncating: str = TRUNCATING_TYPE
) -> Tuple[Tokenizer, np.ndarray, Dict[str, Any]]:
    """
    Fits Keras Tokenizer on clean text list and returns padded sequences.
    """
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(
        sequences, maxlen=maxlen, padding=padding, truncating=truncating
    )

    config = {
        "vocab_size": vocab_size,
        "maxlen": maxlen,
        "oov_token": oov_token,
        "padding": padding,
        "truncating": truncating,
        "num_words_fitted": len(tokenizer.word_index)
    }

    return tokenizer, padded_sequences, config


def text_to_padded_sequence(
    text: str,
    tokenizer: Tokenizer,
    maxlen: int = MAX_LEN,
    padding: str = PADDING_TYPE,
    truncating: str = TRUNCATING_TYPE
) -> np.ndarray:
    """
    Cleans a single input text string and converts it into a padded numpy sequence.
    """
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=maxlen, padding=padding, truncating=truncating)
    return padded


def save_artifacts(
    tokenizer: Tokenizer,
    config: Dict[str, Any],
    models_dir: str = "models"
) -> Tuple[str, str]:
    """Saves tokenizer pickle and configuration JSON to models directory."""
    os.makedirs(models_dir, exist_ok=True)
    tokenizer_path = os.path.join(models_dir, "tokenizer.pickle")
    config_path = os.path.join(models_dir, "config.json")

    with open(tokenizer_path, "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)

    print(f"[INFO] Saved tokenizer to {tokenizer_path}")
    print(f"[INFO] Saved config to {config_path}")
    return tokenizer_path, config_path


def load_artifacts(models_dir: str = "models") -> Tuple[Optional[Tokenizer], Optional[Dict[str, Any]]]:
    """Loads saved tokenizer and configuration from disk."""
    tokenizer_path = os.path.join(models_dir, "tokenizer.pickle")
    config_path = os.path.join(models_dir, "config.json")

    tokenizer, config = None, None

    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)

    return tokenizer, config
