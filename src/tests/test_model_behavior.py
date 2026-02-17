# src/tests/test_model_behavior.py

import pytest
from emoji_sentiment_analysis.modeling.predict import predict_sentiment
from emoji_sentiment_analysis.services.audit_service import generate_inference_audit

# --- EXTENDED TEST DATA ---
# Add "Reasoning" to the test cases to ensure the right feature is winning
GOLDEN_BEHAVIOR_CASES = [
    # 1. Lexicon Heavy (Word List Power)
    ("i love you baby", "Positive", "word_pos_count"),
    ("i hate you baby", "Negative", "word_neg_count"),
    
    # 2. Emoji Power (Veto Power)
    ("today was okay 😊", "Positive", "emoji_pos_count"),
    ("today was okay 😭", "Negative", "emoji_neg_count"),
    
    # 3. Sarcasm (The "Veto" Check)
    # This specifically validates that your Heuristic Override is working
    ("i love having bugs 😭", "Negative", "emoji_neg_count"), 
    ("great, another meeting 😧", "Negative", "emoji_neg_count"),

    # 4. Mixed Signals (Tug of War)
    ("love hate love 😊", "Negative", "word_neg_count"), # Neg count weight (-4.1) > Pos count + emoji
    
    # 5. Pure Signal (Non-Alphanumeric)
    ("😊😊😊😊", "Positive", "emoji_pos_count"),
    ("😭😧😔", "Negative", "emoji_neg_count"),
]

# --- UNIT TESTS ---

@pytest.mark.parametrize("text,expected_sentiment,expected_driver", GOLDEN_BEHAVIOR_CASES)
def test_sentiment_and_drivers(text, expected_sentiment, expected_driver):
    """
    Validates that the model returns the correct label AND that the 
    expected feature is appearing in the top decision drivers.
    """
    result = predict_sentiment(text, run_audit=True)
    
    # 1. Check Label
    assert result["prediction"] == expected_sentiment, \
        f"Label Mismatch for '{text}': Expected {expected_sentiment}, got {result['prediction']}"
    
    # 2. Check Driver (Ensures the 'Smart Lexicon' is actually doing the work)
    driver_names = [d['token'] for d in result['top_drivers']]
    # Note: For Sarcasm veto, the driver might be 'word_pos' but the result is Negative.
    # Check if our key signals are at least being considered.
    assert any(expected_driver in d for d in driver_names), \
        f"Driver Mismatch for '{text}': Expected {expected_driver} to be a top driver. Found: {driver_names}"

def test_ambiguity_logic():
    """
    Ensures the 'High Ambiguity' flag triggers on neutral, low-signal text.
    """
    from emoji_sentiment_analysis.modeling.predict import model, tfidf
    
    neutral_text = "the table is brown and the chair is wooden"
    result = generate_inference_audit(neutral_text, model, tfidf)
    
    assert result["entropy_flag"] == "High Ambiguity", \
        f"Ambiguity not detected for neutral text: {neutral_text}"
    assert result["confidence"] < 0.65, "Neutral text should not have high confidence."

def test_audit_structure():
    """
    Structural test: Ensures the predict_sentiment service returns 
    all keys required by the UI/API.
    """
    result = predict_sentiment("test text 😊")
    
    required_keys = {
        "timestamp", "raw_text", "prediction", "confidence", 
        "entropy_flag", "top_drivers"
    }
    assert required_keys.issubset(result.keys()), \
        f"Missing keys in audit response: {required_keys - set(result.keys())}"

def test_empty_or_garbage_input():
    """
    Resilience test: Ensures the model doesn't crash on meaningless input.
    """
    garbage_text = "!!!!!!!!!!!!!!!"
    result = predict_sentiment(garbage_text)
    
    assert result["prediction"] in ["Positive", "Negative"]
    assert result["entropy_flag"] == "High Ambiguity"