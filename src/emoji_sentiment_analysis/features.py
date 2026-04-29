# src/emoji_sentiment_analysis/features.py

from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------
# MASTER POSITIVE LEXICON
# ---------------------------------------------------------------------
POSITIVE_EMOJIS = {
    # Faces: Happy / Affectionate / Laughing
    "😀",
    "😃",
    "😄",
    "😁",
    "😆",
    "😅",
    "🤣",
    "😂",
    "🙂",
    "🙃",
    "😉",
    "😊",
    "😇",
    "🥰",
    "😍",
    "🤩",
    "😘",
    "😗",
    "😚",
    "😙",
    "😋",
    "😛",
    "😜",
    "🤪",
    "😝",
    "🤑",
    "🤗",
    "🤭",
    "🥱",
    "😌",
    "😎",
    "🥳",
    "😏",
    # Gestures: Approval / Celebration
    "👍",
    "👏",
    "🙌",
    "👐",
    "🤲",
    "🤝",
    "👊",
    "✌️",
    "🤘",
    "👌",
    "🤙",
    "💪",
    "🙏",
    "🎉",
    "🎊",
    "🎈",
    "🧧",
    "✨",
    "🌟",
    "⭐",
    "⚡",
    "🔥",
    "💯",
    "✅",
    "✔️",
    # Symbols: Love / Growth / Success
    "❤️",
    "🧡",
    "💛",
    "💚",
    "💙",
    "💜",
    "🖤",
    "🤍",
    "🤎",
    "💖",
    "💗",
    "💓",
    "💞",
    "💕",
    "💟",
    "❣️",
    "💘",
    "💝",
    "🎯",
    "🏆",
    "💎",
    "🚀",
    "📈",
    "🌈",
    "☀️",
    "🌸",
}

# ---------------------------------------------------------------------
# MASTER NEGATIVE LEXICON
# ---------------------------------------------------------------------
NEGATIVE_EMOJIS = {
    # Faces: Sad / Angry / Distressed
    "🤨",
    "😐",
    "😑",
    "😶",
    "🙄",
    "😒",
    "🤥",
    "😌",
    "😔",
    "😪",
    "🤤",
    "😷",
    "🤒",
    "🤕",
    "🤢",
    "🤮",
    "🤧",
    "🥵",
    "🥶",
    "🥴",
    "😵",
    "🤯",
    "🤠",
    "🥳",
    "😎",
    "🤓",
    "🧐",
    "😕",
    "😟",
    "🙁",
    "☹️",
    "😮",
    "😯",
    "😲",
    "😳",
    "🥺",
    "😦",
    "😧",
    "😨",
    "😰",
    "😥",
    "😢",
    "😭",
    "😱",
    "😖",
    "😣",
    "😞",
    "😓",
    "😩",
    "😫",
    "🥱",
    "😤",
    "😡",
    "😠",
    "🤬",
    "😈",
    "👿",
    "💀",
    "☠️",
    "💩",
    "🤡",
    "👹",
    "👺",
    "👻",
    # Gestures: Disapproval / Warning
    "👎",
    "✊",
    "🤛",
    "🤜",
    "🤞",
    "🤟",
    "🤘",
    "🤏",
    "👈",
    "👉",
    "👆",
    "👇",
    "🖕",
    "🖖",
    "🖐️",
    "✋",
    "🤚",
    "👋",
    "🙅",
    "🙆",
    "🙋",
    "💁",
    "🙇",
    "🤦",
    "🤷",
    "🚫",
    "❌",
    "🛑",
    "⚠️",
    "⛔",
    "🔇",
    "🔕",
    # Symbols: Heartbreak / Failure / Loss
    "💔",
    "🥀",
    "⛈️",
    "🌑",
    "📉",
    "🗑️",
    "💣",
    "🩹",
    "⚰️",
    "🥀",
    "🍂",
    "🌧️",
    "💧",
    "☁️",
}

# ---------------------------------------------------------------------
# EMOTICON FALLBACKS (Western Style)
# ---------------------------------------------------------------------
POSITIVE_EMOTICONS = {":)", ":-)", ":D", "=D", ":P", ":-P", ";)", ";-)", "XD", "<3", "8)"}

NEGATIVE_EMOTICONS = {":(", ":-(", ":'(", "D:", "D-:", ">:(", ":|", ":/", ":-/", ":S", "-_-"}

# ---------------------------------------------------------------------
# Text Lexicon (Added to solve the small-data noise issue)
# ---------------------------------------------------------------------
POSITIVE_WORDS = {
    "happy",
    "good",
    "great",
    "love",
    "nice",
    "awesome",
    "beautiful",
    "amazing",
    "wonderful",
    "excellent",
    "perfect",
}

NEGATIVE_WORDS = {"sad", "bad", "terrible", "hate", "awful", "unhappy"}

EMOJI_BOOST = 10  # Give emojis 10x the weight of a single word; sweetest spot


def extract_emoji_polarity_features(text: str):
    """
    Extracts emoji counts, emoticon counts, and word lexicon counts.
    Returns: (pos_signal, neg_signal, pos_word, neg_word)
    """
    text_str = str(text)
    text_lower = text_str.lower()

    # 1. Emoji Signal (Character-based check)
    pos_emoji = sum(EMOJI_BOOST for char in text_str if char in POSITIVE_EMOJIS)
    neg_emoji = sum(EMOJI_BOOST for char in text_str if char in NEGATIVE_EMOJIS)

    # 2. Emoticon Signal (Substring-based check)
    # Use .count() to handle cases like "Happy :) :)"
    pos_emoticon = sum(text_str.count(emo) * EMOJI_BOOST for emo in POSITIVE_EMOTICONS)
    neg_emoticon = sum(text_str.count(emo) * EMOJI_BOOST for emo in NEGATIVE_EMOTICONS)

    # Combine Emoji and Emoticon signals into single "Polarity" features
    total_pos_signal = pos_emoji + pos_emoticon
    total_neg_signal = neg_emoji + neg_emoticon

    # 3. Word Counts
    pos_word = sum(1 for word in POSITIVE_WORDS if word in text_lower)
    neg_word = sum(1 for word in NEGATIVE_WORDS if word in text_lower)

    return total_pos_signal, total_neg_signal, pos_word, neg_word


def build_final_features(input_path: Path, output_path: Path):
    """
    Applies the hybrid feature extraction to the dataset and persists it.
    """
    import pandas as pd

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
