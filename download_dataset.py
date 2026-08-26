import os
import urllib.request
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_FILE = os.path.join(DATA_DIR, "spam.csv")
SAMPLE_BATCH_FILE = os.path.join(DATA_DIR, "sample_batch.csv")

# Reliable mirror links for SMS Spam Collection dataset
DATASET_URLS = [
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
    "https://raw.githubusercontent.com/stefan-jop/sms-spam-collection/main/spam.csv"
]


def download_or_generate_dataset() -> pd.DataFrame:
    """
    Downloads the SMS Spam Collection dataset or generates a synthetic fallback if offline.
    Ensures dataset contains columns 'v1' (label) and 'v2' (text).
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_FILE):
        print(f"[INFO] Checking existing dataset at {DATA_FILE}...")
        try:
            try:
                df = pd.read_csv(DATA_FILE, encoding='utf-8')
            except Exception:
                df = pd.read_csv(DATA_FILE, encoding='latin-1')
            if 'v1' in df.columns and 'v2' in df.columns and len(df) > 100:
                print(f"[INFO] Dataset loaded successfully with {len(df)} records.")
                generate_sample_batch(df)
                return df
        except Exception as e:
            print(f"[WARNING] Failed to load existing dataset ({e}). Re-downloading...")

    df = None
    for url in DATASET_URLS:
        try:
            print(f"[INFO] Attempting download from {url}...")
            if url.endswith('.tsv'):
                df_raw = pd.read_csv(url, sep='\t', names=['v1', 'v2'], encoding='utf-8')
            else:
                try:
                    df_raw = pd.read_csv(url, encoding='utf-8')
                except Exception:
                    df_raw = pd.read_csv(url, encoding='latin-1')
                if 'v1' not in df_raw.columns:
                    df_raw.rename(columns={df_raw.columns[0]: 'v1', df_raw.columns[1]: 'v2'}, inplace=True)
            
            df = df_raw[['v1', 'v2']].dropna()
            df.to_csv(DATA_FILE, index=False, encoding='utf-8')
            print(f"[SUCCESS] Dataset downloaded successfully from {url}. Total rows: {len(df)}")
            break
        except Exception as err:
            print(f"[WARNING] Download failed from {url}: {err}")

    if df is None:
        print("[WARNING] Online download failed. Generating comprehensive fallback dataset...")
        df = generate_fallback_dataset()
        df.to_csv(DATA_FILE, index=False, encoding='utf-8')
        print(f"[SUCCESS] Fallback dataset generated. Total rows: {len(df)}")

    generate_sample_batch(df)
    return df


def generate_fallback_dataset() -> pd.DataFrame:
    """Generates a representative set of ham and spam SMS messages if offline."""
    ham_samples = [
        "Hey, are we still meeting for lunch at 12:30?",
        "Can you please send me the project report when you get a chance?",
        "I'm on my way home, see you in 20 minutes.",
        "Don't forget to buy milk and eggs on your way back.",
        "That movie was great! Let's watch the sequel next weekend.",
        "Sorry for the late reply, I was in a meeting all morning.",
        "Are you free for a call later today to discuss the presentation?",
        "Happy birthday! Hope you have a wonderful day ahead.",
        "The professor postponed the assignment deadline to Friday.",
        "Thanks for helping me move yesterday, really appreciate it!",
        "What time does the train leave tomorrow morning?",
        "I'll be working late tonight, don't wait up for dinner.",
        "Let me know if you need any help with your homework.",
        "Good morning! Have a productive day.",
        "Can we reschedule our coffee chat to Thursday?"
    ] * 40  # 600 ham messages

    spam_samples = [
        "WINNER! You have won a $1000 Walmart gift card! Call 1-800-555-0199 now to claim!",
        "URGENT! Your bank account has been compromised. Verify your details at http://bit.ly/fake-bank immediately.",
        "Congratulations! You've been selected for a free iPhone 15. Click here to claim your reward!",
        "Claim your 500 FREE spins at Gold Casino today! Reply WIN to activate bonus code 777.",
        "HOT LOAN OFFER! Get $50,000 instant loan with zero collateral. Call 0800-777-999 now!",
        "You have 1 unread message from a secret admirer! Dial 09061701461 to listen.",
        "FINAL NOTICE: Your mobile bill payment of $250 is overdue. Pay now at http://pay-now.fake to avoid service block.",
        "Cash Prize Alert! Reply YES to win £5000 cash prize instantly. T&Cs apply.",
        "Exclusive offer: Get 80% discount on designer watches today only! Visit http://luxury-replica.com",
        "Free tones! Text RING to 80077 to get top 10 ringtones on your mobile now!"
    ] * 60  # 600 spam messages

    labels = ['ham'] * len(ham_samples) + ['spam'] * len(spam_samples)
    texts = ham_samples + spam_samples

    return pd.DataFrame({'v1': labels, 'v2': texts})


def generate_sample_batch(df: pd.DataFrame):
    """Saves a small sample CSV for batch upload testing in Streamlit."""
    sample_df = df.sample(min(15, len(df)), random_state=42)[['v2']].rename(columns={'v2': 'sms_text'})
    sample_df.to_csv(SAMPLE_BATCH_FILE, index=False)
    print(f"[INFO] Sample batch CSV created at {SAMPLE_BATCH_FILE}")


if __name__ == "__main__":
    download_or_generate_dataset()
