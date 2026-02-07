from pathlib import Path
import pandas as pd

"""
Emoji Polarity Mapping
----------------------

Source Dataset:
    data/processed/emoji_reference_clean.csv

Selection Criteria:
    - High observed frequency in the tweet corpus (Notebook 2.0)
    - Clear sentiment polarity alignment
    - Empirically validated via modeling performance (Notebook 3.0)

Design Rationale:
    Emoji features were intentionally constrained to a small, high-signal subset
    to reduce sparsity, prevent overfitting, and maintain deployment stability.

Exclusions (Intentional):
    The following emojis exist in the reference dataset but were excluded:

        😢  CRYING FACE
        😲  ASTONISHED FACE
        😐  NEUTRAL FACE

    Reasons for exclusion:
        - Did not appear or appeared negligibly in the tweet corpus
        - Possess weak or ambiguous sentiment polarity
        - Were not validated as beneficial during modeling evaluation

    This ensures that the final feature set remains evidence-bound rather than
    semantically or intuitively expanded.

Deployment Note:
    This mapping is frozen post-evaluation and must remain stable to ensure
    feature consistency between training and inference environments.
"""

POSITIVE_EMOJIS = {
    "😊",  # SMILING FACE WITH SMILING EYES
    "😄",  # SMILING FACE WITH OPEN MOUTH AND SMILING EYES
    "😆",  # SMILING FACE WITH OPEN MOUTH AND TIGHTLY-CLOSED EYES
    "😍",  # SMILING FACE WITH HEART-SHAPED EYES
    "😘",  # FACE THROWING A KISS
}

NEGATIVE_EMOJIS = {
    "😧",  # ANGUISHED FACE
    "😔",  # PENSIVE FACE
    "😭",  # LOUDLY CRYING FACE
    "😒",  # UNAMUSED FACE
}

def extract_emoji_polarity_features(text: str):
    pos_count = 0
    neg_count = 0
    
    for char in text:
        if char in POSITIVE_EMOJIS:
            pos_count += 1
        elif char in NEGATIVE_EMOJIS:
            neg_count += 1
    
    return pos_count, neg_count


def build_final_features(
    input_path: Path,
    output_path: Path,
):
    df = pd.read_csv(input_path)

    features = df["text"].apply(extract_emoji_polarity_features)
    df["emoji_pos_count"] = [f[0] for f in features]
    df["emoji_neg_count"] = [f[1] for f in features]

    df.to_csv(output_path, index=False)
