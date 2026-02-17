# tests/test_features.py

import pandas as pd
from pathlib import Path
from emoji_sentiment_analysis.features import (
    extract_emoji_polarity_features,
    build_final_features,
)

# Note: Expect EMOJI_BOOST = 10
BOOST = 10

# --- Unit tests for hybrid signal extraction ---

def test_positive_signals():
    # 2 emojis (20) + 1 emoticon (10) + 1 word (1) = 31
    text = "I am happy 😊😊 :)"
    pos, neg, pos_w, neg_w = extract_emoji_polarity_features(text)
    
    assert pos == 30 # (2 * BOOST) + (1 * BOOST)
    assert neg == 0
    assert pos_w == 1 # "happy"
    assert neg_w == 0

def test_negative_signals():
    # 2 emojis (20) + 1 word (1) = 21
    text = "This is bad 😭😔"
    pos, neg, pos_w, neg_w = extract_emoji_polarity_features(text)
    
    assert pos == 0
    assert neg == 20 # 2 * BOOST
    assert pos_w == 0
    assert neg_w == 1 # "bad"

def test_emoticon_detection():
    # Testing specifically the new multi-character emoticon logic
    text = "Classic faces :-) :("
    pos, neg, _, _ = extract_emoji_polarity_features(text)
    
    assert pos == 10 # :-)
    assert neg == 10 # :(

def test_mixed_signals():
    text = "Bittersweet 😊😭"
    pos, neg, _, _ = extract_emoji_polarity_features(text)
    
    assert pos == 10
    assert neg == 10

# --- Integration test for dataset generation ---

def test_build_final_features(tmp_path):
    input_file = tmp_path / "tweets_clean.csv"
    output_file = tmp_path / "features_final.csv"

    df = pd.DataFrame({
        "text": ["Happy 😊", "Sad 😭", "Neutral text"],
        "label": [1, 0, 1],
    })

    df.to_csv(input_file, index=False)

    build_final_features(input_file, output_file)

    result = pd.read_csv(output_file)

    # Check for all 4 columns now
    assert "emoji_pos_count" in result.columns
    assert "emoji_neg_count" in result.columns
    assert "word_pos_count" in result.columns
    assert "word_neg_count" in result.columns

    # Verify boosted counts
    assert result.loc[0, "emoji_pos_count"] == 10
    assert result.loc[1, "emoji_neg_count"] == 10