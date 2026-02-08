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

# Pipeline Imports
from emoji_sentiment_analysis.dataset import main as run_data_cleaning
from scripts.build_features import main as run_feature_engineering
from emoji_sentiment_analysis.modeling.train_model import train_pipeline
from emoji_sentiment_analysis.modeling.predict import predict_sentiment
from scripts.check_model_health import run_health_check
from emoji_sentiment_analysis.config import init_logging, ensure_dirs

def run_pipeline():
    # Initial System Setup
    init_logging()
    ensure_dirs()
    
    logger.info("🚀 Starting Emoji Sentiment Analysis Pipeline")
    print("="*60)

    try:
        # Step 1: Cleaning & Ingestion
        logger.info("Step 1/4: Cleaning Raw Data (Standardizing Schemas)...")
        try:
            run_data_cleaning()
        except SystemExit:
            # Catch Typer's exit signal to continue the pipeline
            pass

        # Step 2: Feature Engineering
        logger.info("Step 2/4: Building Hybrid Features (Emojis + Lexicons)...")
        run_feature_engineering()

        # Step 3: Model Training
        logger.info("Step 3/4: Training & Serializing Logistic Regression...")
        train_pipeline()

        # Step 4: Health Certification (The Automated behavioral check)
        logger.info("Step 4/4: Running Model Health Certification...")
        run_health_check()

        logger.success("✅ Full Pipeline & Behavioral Certification Completed Successfully!")
        print("="*60)

        # Interactive Mode
        print("\n--- Model Live Test (Type 'q' to quit) ---")
        
        while True:
            user_input = input("\nText to analyze > ")
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            
            if not user_input.strip():
                continue

            result = predict_sentiment(user_input)
            
            if "error" in result:
                logger.error(f"Prediction failed: {result['error']}")
                continue

            sentiment = result["prediction"]
            confidence = result["confidence"]
            entropy = result["entropy_flag"]
            
            # Formatting top drivers for the terminal
            drivers_list = [f"{d['token']} ({d['weight']})" for d in result['top_drivers']]
            drivers_str = ", ".join(drivers_list)

            print(f"Result: {sentiment} (Confidence: {confidence})")
            print(f"Signal: {entropy}")
            print(f"Drivers: {drivers_str}")

    except Exception as e:
        logger.exception(f"Pipeline failed at runtime: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()