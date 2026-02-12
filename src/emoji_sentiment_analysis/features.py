# src/emoji_sentiment_analysis/features.py

from pathlib import Path
import pandas as pd
from loguru import logger  # Added this import

# ---------------------------------------------------------------------
# Emoji Lexicon (Synced with Notebook 2.0 counts)
# ---------------------------------------------------------------------
POSITIVE_EMOJIS = {
    "😍", # HEART-SHAPED EYES
    "😘", # FACE THROWING A KISS
    "😊", # SMILING EYES
    "😁", # GRINNING FACE
    "😉", # WINKING FACE
    "😄", # OPEN MOUTH & SMILING EYES
    "😆", # TIGHTLY-CLOSED EYES
    "😀", # GRINNING FACE
    "😛"  # STUCK-OUT TONGUE
}

NEGATIVE_EMOJIS = {
    "😭", # LOUDLY CRYING
    "😒", # UNAMUSED
    "😔", # PENSIVE
    "😢", # CRYING
    "😧", # ANGUISHED
    "😲"  # ASTONISHED (Usually negative shock in sentiment tasks)
}

# NEUTRAL_EMOJIS = {"😐"} # 'NEUTRAL FACE' - We exclude this from pos/neg counts

# ---------------------------------------------------------------------
# Text Lexicon (Added to solve the small-data noise issue)
# ---------------------------------------------------------------------
POSITIVE_WORDS = {"happy", "good", "great", "love", "nice", "awesome"}
NEGATIVE_WORDS = {"sad", "bad", "terrible", "hate", "awful", "unhappy"}

EMOJI_BOOST = 10 # Give emojis 10x the weight of a single word

def extract_emoji_polarity_features(text: str):
    """
    Extracts both emoji counts and word lexicon counts.
    Returns: (pos_emoji, neg_emoji, pos_word, neg_word)
    """
    text_lower = str(text).lower()
    
    # Emoji Counts with Boost
    # If one emoji appears, it counts as 5 'points' 
    pos_emoji = sum(EMOJI_BOOST for char in text if char in POSITIVE_EMOJIS)
    neg_emoji = sum(EMOJI_BOOST for char in text if char in NEGATIVE_EMOJIS)
    
    # Word Counts (Helps stabilize the model against small-data noise)
    pos_word = sum(1 for word in POSITIVE_WORDS if word in text_lower)
    neg_word = sum(1 for word in NEGATIVE_WORDS if word in text_lower)
    
    return pos_emoji, neg_emoji, pos_word, neg_word

def build_final_features(input_path: Path, output_path: Path):
    """
    Applies the hybrid feature extraction to the dataset and persists it.
    """
    df = pd.read_csv(input_path)

    logger.info("Extracting hybrid features (Emoji + Text Lexicons)...")
    
    # Apply extraction and expand results into 4 separate columns
    features = df["text"].apply(extract_emoji_polarity_features)
    
    df["emoji_pos_count"] = [f[0] for f in features]
    df["emoji_neg_count"] = [f[1] for f in features]
    df["word_pos_count"] = [f[2] for f in features]
    df["word_neg_count"] = [f[3] for f in features]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Hybrid features saved to {output_path}")