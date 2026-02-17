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
from emoji_sentiment_analysis.dataset import main as run_data_cleaning
from scripts.build_features import main as run_feature_engineering
from emoji_sentiment_analysis.modeling.train_model import train_pipeline
from emoji_sentiment_analysis.modeling.predict import predict_sentiment
from scripts.check_model_health import run_health_check
from emoji_sentiment_analysis.config import init_logging, ensure_dirs

def run_pipeline():
    init_logging()
    ensure_dirs()
    
    logger.info("🚀 Starting Emoji Sentiment Analysis Pipeline")
    print("="*60)

    try:
        run_data_cleaning()
        run_feature_engineering()
        train_pipeline()
        run_health_check() 

        logger.success("✅ Full Pipeline Completed Successfully!")
        print("="*60)

        print("\n--- Model Live Test (Type 'q' to quit) ---")
        while True:
            user_input = input("\nText to analyze > ")
            if user_input.lower() in ['q', 'quit', 'exit']: break
            if not user_input.strip(): continue

            result = predict_sentiment(user_input, run_audit=False)
            print(f"Result: {result['prediction']} | Confidence: {result['confidence']}")
            print(f"Drivers: {[f'{d['token']} ({d['weight']})' for d in result['top_drivers']]}")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()