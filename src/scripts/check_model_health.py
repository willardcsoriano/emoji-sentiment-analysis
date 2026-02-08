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

    # Define the test file to run
    test_file = "src/tests/test_model_behavior.py"
    
    # Run pytest programmatically
    # -q: quiet, --json-report would require a plugin, so we'll capture results simply
    exit_code = pytest.main([test_file, "-q"])

    # Map exit codes to status
    passed = (exit_code == 0)
    
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "PASSED" if passed else "FAILED",
        "test_suite": test_file,
        "checks": {
            "lexicon_integrity": "OK" if passed else "CHECK REQUIRED",
            "emoji_veto_power": "OK" if passed else "CHECK REQUIRED",
            "sarcasm_detection": "OK" if passed else "CHECK REQUIRED"
        },
        "recommendation": "Ready for Deployment" if passed else "Do Not Deploy - Logic Regressed"
    }

    # Save the report
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    if passed:
        logger.success(f"✅ Model Certified! Report saved to {report_path}")
    else:
        logger.error(f"❌ Model Health Failed. Review behavioral tests in {test_file}")

if __name__ == "__main__":
    run_health_check()