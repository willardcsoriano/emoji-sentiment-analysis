# src/main.py

"""
Emoji Sentiment Analysis - Master Pipeline
------------------------------------------
1. Data Cleaning (dataset.py)
2. Feature Engineering (build_features.py)
3. Model Training (train_model.py)
4. Model Health Certification (check_model_health.py)
5. Interactive Test Mode
"""

import sys

from loguru import logger

from emoji_sentiment_analysis.config import ensure_dirs, init_logging
from emoji_sentiment_analysis.dataset import main as run_data_cleaning
from emoji_sentiment_analysis.modeling.predict import predict_sentiment
from emoji_sentiment_analysis.modeling.train_model import train_pipeline
from scripts.build_features import main as run_feature_engineering
from scripts.check_model_health import run_health_check


def print_result(result: dict) -> None:
    """Pretty-print the full 8-key inference response."""
    print("\n" + "=" * 60)
    print(f"  INPUT       : {result['raw_text']}")
    print(f"  PREDICTION  : {result['prediction']} ({result['prediction_int']})")
    print(f"  CONFIDENCE  : {result['confidence']:.4f}")
    print(f"  ENTROPY     : {result['entropy_flag']}")
    print(f"  VETO APPLIED: {result['veto_applied']}")
    print(f"  TIMESTAMP   : {result['timestamp']}")
    print("  TOP DRIVERS :")
    for d in result["top_drivers"]:
        print(f"    {d['token']:<30} {d['weight']:>+.4f}")
    print("=" * 60)


def run_pipeline():
    init_logging()
    ensure_dirs()

    logger.info("🚀 Starting Emoji Sentiment Analysis Pipeline")
    print("=" * 60)

    try:
        run_data_cleaning()
        run_feature_engineering()
        train_pipeline()
        run_health_check()

        logger.success("✅ Full Pipeline Completed Successfully!")
        print("=" * 60)

        print("\n--- Model Live Test (Type 'q' to quit) ---")
        while True:
            user_input = input("\nText to analyze > ")
            if user_input.lower() in ["q", "quit", "exit"]:
                break
            if not user_input.strip():
                continue

            result = predict_sentiment(user_input)
            print_result(result)

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
