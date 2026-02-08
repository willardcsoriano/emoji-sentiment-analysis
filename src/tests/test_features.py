import pandas as pd
from pathlib import Path
from emoji_sentiment_analysis.features import (
    extract_emoji_polarity_features,
    build_final_features,
)

# --- Unit tests for emoji extraction ---

def test_positive_emoji_count():
    text = "I am happy 😊😊😍"
    pos, neg = extract_emoji_polarity_features(text)
    
    assert pos == 3
    assert neg == 0


def test_negative_emoji_count():
    text = "This is bad 😭😔"
    pos, neg = extract_emoji_polarity_features(text)
    
    assert pos == 0
    assert neg == 2


def test_mixed_emoji_count():
    text = "Bittersweet 😊😭"
    pos, neg = extract_emoji_polarity_features(text)
    
    assert pos == 1
    assert neg == 1


def test_no_emoji():
    text = "Just text here"
    pos, neg = extract_emoji_polarity_features(text)
    
    assert pos == 0
    assert neg == 0


# --- Integration test for dataset generation ---

def test_build_final_features(tmp_path):
    # Create temporary input dataset
    input_file = tmp_path / "tweets_clean.csv"
    output_file = tmp_path / "features_final.csv"

    df = pd.DataFrame({
        "text": ["Happy 😊", "Sad 😭", "Neutral text"],
        "label": [1, 0, 1],
    })

    df.to_csv(input_file, index=False)

    # Run feature builder
    build_final_features(input_file, output_file)

    # Load result
    result = pd.read_csv(output_file)

    assert "emoji_pos_count" in result.columns
    assert "emoji_neg_count" in result.columns

    assert result.loc[0, "emoji_pos_count"] == 1
    assert result.loc[1, "emoji_neg_count"] == 1
    assert result.loc[2, "emoji_pos_count"] == 0
