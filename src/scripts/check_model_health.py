# scripts/check_model_health.py

"""
Model Health & Behavioral Certification
---------------------------------------
Executes a suite of behavioral tests against the current model
and generates a quality report for the reports/ directory.
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from loguru import logger
from emoji_sentiment_analysis.config import REPORTS_DIR

def run_health_check():
    report_path = REPORTS_DIR / "model_health_report.json"
    logger.info("Running Model Health Certification...")

    # --- DYNAMIC PATH DISCOVERY ---
    # Path(__file__) is C:/.../src/scripts/check_model_health.py
    # .parent.parent is C:/.../src/
    BASE_DIR = Path(__file__).resolve().parent.parent
    test_file = BASE_DIR / "tests" / "test_model_behavior.py"
    
    # Check if the file actually exists
    if not test_file.exists():
        logger.error(f"Test suite missing at: {test_file}")
        return

    # Run pytest programmatically (passing absolute path as string)
    exit_code = pytest.main([str(test_file), "-q"])

    # Map exit codes to status (0 is success)
    passed = (exit_code == 0)
    
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "PASSED" if passed else "FAILED",
        "test_suite": test_file.name,
        "checks": {
            "lexicon_integrity": "OK" if passed else "CHECK REQUIRED",
            "emoji_veto_power": "OK" if passed else "CHECK REQUIRED",
            "sarcasm_detection": "OK" if passed else "CHECK REQUIRED"
        },
        "recommendation": "Ready for Deployment" if passed else "Do Not Deploy - Logic Regressed"
    }

    # Ensure reports directory exists
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the report
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    if passed:
        logger.success(f"✅ Model Certified! Report saved to {report_path}")
    else:
        logger.error(f"❌ Model Health Failed. Review behavioral tests in {test_file}")

if __name__ == "__main__":
    run_health_check()