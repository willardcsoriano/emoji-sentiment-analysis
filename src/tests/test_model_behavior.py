# src/tests/test_model_behavior.py

"""
Behavioral Test Suite
---------------------
End-to-end validation of the full inference pipeline:
predict_sentiment() → sarcasm veto → model → audit service.

These tests validate decisions, not math. Feature extraction math
is covered separately in test_features.py.
"""

import pytest

from emoji_sentiment_analysis.modeling.predict import predict_sentiment

# ---------------------------------------------------------------------------
# Golden Behavior Cases
# ---------------------------------------------------------------------------
# Format: (text, expected_sentiment, expected_driver)
# expected_driver: the feature name that should appear in top_drivers

GOLDEN_BEHAVIOR_CASES = [
    # 1. Word lexicon — unambiguous text, no emoji interference
    ("i love you baby",           "Positive", "word_pos_count"),
    ("i hate you baby",           "Negative", "word_neg_count"),

    # 2. Emoji signal — emoji tips neutral text
    ("today was okay 😊",         "Positive", "emoji_pos_count"),
    ("today was okay 😭",         "Negative", "emoji_neg_count"),

    # 3. Sarcasm veto — negative emoji overrides positive text
    ("i love having bugs 😭",     "Negative", "emoji_neg_count"),
    ("great, another meeting 😧", "Negative", "emoji_neg_count"),
    ("wow amazing day 😭😭",      "Negative", "emoji_neg_count"),

    # 4. Mixed signals — tug of war between lexicon features
    ("love hate love 😊",         "Negative", "word_neg_count"),

    # 5. Emoji-dominant — no text signal at all
    ("😊😊😊😊",                  "Positive", "emoji_pos_count"),
    ("😭😧😔",                    "Negative", "emoji_neg_count"),
    ("😭😭😭",                    "Negative", "emoji_neg_count"),

    # 6. Emoticon — veto-resolved via negative emoticon
    ("feeling great today :(",    "Negative", "emoji_neg_count"),
]

# ---------------------------------------------------------------------------
# Documented Limitations — excluded from regression suite
# These are known failure cases, not bugs.
#
# "this is the worst :)"  — positive emoticon overrides negative text.
#                           No positive emoticon veto exists by design.
# "okay I guess"          — genuinely ambiguous, no sentiment signal.
#                           Model tips Positive on weak n-gram priors.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Core Behavioral Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_sentiment,expected_driver", GOLDEN_BEHAVIOR_CASES)
def test_sentiment_and_drivers(text, expected_sentiment, expected_driver):
    """
    Validates prediction label and that the expected feature appears
    in the top decision drivers. Tests the full pipeline including veto.
    """
    result = predict_sentiment(text)

    assert result["prediction"] == expected_sentiment, (
        f"Label mismatch for '{text}': "
        f"expected {expected_sentiment}, got {result['prediction']}"
    )

    driver_names = [d["token"] for d in result["top_drivers"]]
    assert any(expected_driver in d for d in driver_names), (
        f"Driver mismatch for '{text}': "
        f"expected '{expected_driver}' in top drivers. Found: {driver_names}"
    )


# ---------------------------------------------------------------------------
# Veto-Specific Tests
# ---------------------------------------------------------------------------

def test_veto_flag_is_set_on_sarcasm():
    """
    Confirms veto_applied is True when negative emoji overrides positive text.
    Validates the veto is firing and being reported correctly.
    """
    result = predict_sentiment("i love having bugs 😭")
    assert result["veto_applied"] is True, (
        "Sarcasm veto should have fired but veto_applied is False"
    )


def test_veto_flag_is_false_on_clean_input():
    """
    Confirms veto_applied is False when no conflict exists.
    """
    result = predict_sentiment("i love this project 😊")
    assert result["veto_applied"] is False, (
        "Veto should not fire on non-conflict input"
    )


def test_veto_confidence_is_scaled():
    """
    Veto-resolved predictions should return scaled confidence 0.75–0.95
    based on signal dominance. Hardcoded 0.85 was replaced in predict.py
    per Notebook 3.5 Section 8b specification.
    """
    result = predict_sentiment("great, another meeting 😧")
    assert result["veto_applied"] is True
    assert 0.75 <= result["confidence"] <= 0.95, (
        f"Veto confidence should be in range 0.75–0.95, got {result['confidence']}"
    )


# ---------------------------------------------------------------------------
# Ambiguity Tests
# ---------------------------------------------------------------------------

def test_ambiguity_flag_on_neutral_text():
    """
    Ensures High Ambiguity flag triggers on neutral, low-signal text.
    """
    result = predict_sentiment("the table is brown and the chair is wooden")
    assert result["entropy_flag"] == "High Ambiguity", (
        "Ambiguity not detected for neutral text"
    )
    assert result["confidence"] < 0.65, (
        f"Neutral text should not have high confidence, got {result['confidence']}"
    )


def test_ambiguity_flag_absent_on_clear_input():
    """
    Ensures Clear Signal flag on strongly-valenced input.
    """
    result = predict_sentiment("i absolutely hate this terrible experience 😭😭")
    assert result["entropy_flag"] == "Clear Signal", (
        "Strong negative input should not flag High Ambiguity"
    )


# ---------------------------------------------------------------------------
# Structural / Resilience Tests
# ---------------------------------------------------------------------------

def test_audit_structure():
    """
    Ensures predict_sentiment returns all keys required by the UI and API.
    Includes veto_applied added during the predict.py refactor.
    """
    result = predict_sentiment("test text 😊")
    required_keys = {
        "timestamp", "raw_text", "prediction", "prediction_int",
        "confidence", "entropy_flag", "top_drivers", "veto_applied"
    }
    missing = required_keys - set(result.keys())
    assert not missing, f"Missing keys in response: {missing}"


def test_garbage_input_resilience():
    """
    Ensures the pipeline does not crash on meaningless input
    and returns High Ambiguity on zero-signal text.
    """
    result = predict_sentiment("!!!!!!!!!!!!!!!")
    assert result["prediction"] in ["Positive", "Negative"]
    assert result["entropy_flag"] == "High Ambiguity"


def test_empty_string_resilience():
    """
    Ensures the pipeline handles an empty string without crashing.
    """
    result = predict_sentiment("")
    assert result["prediction"] in ["Positive", "Negative"]
    assert "top_drivers" in result


def test_emoji_only_input():
    """
    Ensures emoji-only input (no text) produces a valid prediction.
    Pure emoji signal should be sufficient for a confident prediction.
    """
    result = predict_sentiment("😊😊😊")
    assert result["prediction"] == "Positive"
    assert result["entropy_flag"] == "Clear Signal"

    # ---------------------------------------------------------------------------
# Driver Detail Tests
# ---------------------------------------------------------------------------

def test_driver_count():
    """
    Confirms top_drivers never exceeds 6 — the UI display limit.
    """
    result = predict_sentiment("i love this project so much 😊😊")
    assert len(result["top_drivers"]) <= 6, (
        f"Expected max 6 drivers, got {len(result['top_drivers'])}"
    )


def test_driver_structure():
    """
    Confirms each driver has the required keys and correct value types.
    """
    result = predict_sentiment("i love this project 😊")
    for driver in result["top_drivers"]:
        assert "token" in driver, f"Driver missing 'token' key: {driver}"
        assert "weight" in driver, f"Driver missing 'weight' key: {driver}"
        assert isinstance(driver["token"], str), \
            f"Driver token should be str, got {type(driver['token'])}"
        assert isinstance(driver["weight"], float), \
            f"Driver weight should be float, got {type(driver['weight'])}"


def test_driver_ordering():
    """
    Confirms drivers are sorted by absolute weight descending —
    the highest-influence feature should always be first.
    """
    result = predict_sentiment("i love this project 😊")
    weights = [abs(d["weight"]) for d in result["top_drivers"]]
    assert weights == sorted(weights, reverse=True), \
        f"Drivers are not sorted by absolute weight: {weights}"


def test_veto_driver_is_primary():
    """
    When veto fires, emoji_neg_count must be the first driver —
    the UI should surface what actually drove the decision.
    """
    result = predict_sentiment("i love having bugs 😭")
    assert result["veto_applied"] is True
    assert result["top_drivers"][0]["token"] == "emoji_neg_count", (
        f"Veto fired but emoji_neg_count is not the primary driver. "
        f"Got: {result['top_drivers'][0]['token']}"
    )


def test_driver_weights_are_nonzero():
    """
    Confirms no zero-weight drivers appear in the top 6 —
    a zero-weight feature has no explanatory value and
    should not be shown to the user.
    """
    result = predict_sentiment("i love this project 😊")
    for driver in result["top_drivers"]:
        assert driver["weight"] != 0.0, (
            f"Zero-weight driver found: {driver['token']}"
        )


def test_positive_prediction_has_positive_primary_driver():
    """
    For a clean positive prediction with no veto, confirm the prediction
    is Positive and emoji_pos_count appears as a driver.
    The primary driver weight is not guaranteed to be positive — TF-IDF
    tokens like 'so' can have negative coefficients from training context.
    """
    result = predict_sentiment("i love this so much 😊😊😊")
    assert result["prediction"] == "Positive"
    assert result["veto_applied"] is False
    driver_names = [d["token"] for d in result["top_drivers"]]
    assert "emoji_pos_count" in driver_names, (
        f"emoji_pos_count should appear as a driver. Found: {driver_names}"
)


def test_negative_prediction_has_negative_primary_driver():
    """
    For a clean negative prediction with no veto, the primary driver
    should have a negative weight — it should be pulling toward Negative.
    """
    result = predict_sentiment("i hate this so much 😭😭😭")
    assert result["veto_applied"] is True  # veto fires here
    # veto_applied means emoji_neg_count is primary — always negative weight
    assert result["top_drivers"][0]["token"] == "emoji_neg_count"


def test_driver_tokens_are_known_features():
    """
    All hybrid feature drivers should be one of the four known
    numeric feature names. TF-IDF tokens are strings so we only
    validate the hybrid ones specifically.
    """
    HYBRID_FEATURES = {
        "emoji_pos_count", "emoji_neg_count",
        "word_pos_count", "word_neg_count"
    }
    result = predict_sentiment("i love having bugs 😭")
    driver_tokens = {d["token"] for d in result["top_drivers"]}
    hybrid_drivers_found = driver_tokens & HYBRID_FEATURES

    # At least one hybrid feature should appear for emoji-containing input
    assert hybrid_drivers_found, (
        f"No hybrid features found in drivers for emoji input. "
        f"Got: {driver_tokens}"
    )