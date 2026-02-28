# src/scripts/check_model_health.py

"""
Model Health & Behavioral Certification
---------------------------------------
Executes a suite of behavioral tests against the current model
and generates a quality report for the reports/ directory.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest
from loguru import logger

from emoji_sentiment_analysis.config import REPORTS_DIR


class _ResultCollector:
    """Minimal pytest plugin to capture per-test outcomes."""

    def __init__(self):
        self.passed = []
        self.failed = []
        self.errors = []

    def pytest_runtest_logreport(self, report):
        if report.when != "call":
            return
        if report.passed:
            self.passed.append(report.nodeid)
        elif report.failed:
            self.failed.append(report.nodeid)

    def pytest_internalerror(self, excrepr):
        self.errors.append(str(excrepr))


def run_health_check():
    report_path = REPORTS_DIR / "model_health_report.json"
    logger.info("Running Model Health Certification...")

    BASE_DIR = Path(__file__).resolve().parent.parent
    test_file = BASE_DIR / "tests" / "test_model_behavior.py"

    if not test_file.exists():
        logger.error(f"Test suite missing at: {test_file}")
        return

    collector = _ResultCollector()
    exit_code = pytest.main([str(test_file), "-q"], plugins=[collector])

    # Interpret exit codes explicitly
    EXIT_MEANINGS = {
        0: "all tests passed",
        1: "tests failed",
        2: "interrupted",
        3: "internal error",
        4: "command line error",
        5: "no tests collected",
    }
    exit_meaning = EXIT_MEANINGS.get(int(exit_code), "unknown")
    passed = exit_code == 0

    # Categorize failures by test name prefix
    def categorize(node_ids):
        cats = {
            "lexicon_integrity": [],
            "emoji_veto": [],
            "sarcasm_detection": [],
            "confidence": [],
            "other": [],
        }
        for nid in node_ids:
            name = nid.split("::")[-1]
            if "lexicon" in name or "word" in name:
                cats["lexicon_integrity"].append(name)
            elif "veto" in name or "emoji" in name:
                cats["emoji_veto"].append(name)
            elif "sarcasm" in name:
                cats["sarcasm_detection"].append(name)
            elif "confidence" in name:
                cats["confidence"].append(name)
            else:
                cats["other"].append(name)
        return {k: v for k, v in cats.items() if v}

    failed_by_category = categorize(collector.failed)

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "PASSED" if passed else "FAILED",
        "exit_code": int(exit_code),
        "exit_meaning": exit_meaning,
        "test_suite": test_file.name,
        "summary": {
            "total": len(collector.passed) + len(collector.failed),
            "passed": len(collector.passed),
            "failed": len(collector.failed),
        },
        "failed_tests": collector.failed,
        "failed_by_category": failed_by_category,
        "internal_errors": collector.errors,
        "recommendation": (
            "Ready for Deployment"
            if passed
            else "Do Not Deploy — Behavioral Regression Detected"
            if exit_code == 1
            else f"Investigate Pipeline — {exit_meaning}"
        ),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    if passed:
        logger.success(
            f"✅ Model Certified. {report['summary']['passed']} tests passed. "
            f"Report → {report_path}"
        )
    else:
        logger.error(
            f"❌ Model Health Failed ({exit_meaning}). "
            f"{report['summary']['failed']} failed. "
            f"Review → {test_file}"
        )
        if collector.failed:
            for name in collector.failed:
                logger.error(f"   FAILED: {name}")


if __name__ == "__main__":
    run_health_check()
