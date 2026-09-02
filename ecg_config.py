"""Central configuration defaults for the ECG analysis scaffold."""

from __future__ import annotations

DEFAULT_SAMPLING_RATE: int = 250
DEFAULT_SYNTHETIC_DURATION_SECONDS: int = 10
DEFAULT_OUTPUT_DIR: str = "output"
DEFAULT_INPUT_COLUMN_INDEX: int = 0
DEFAULT_INPUT_SOURCE: str = "File Replay"
DEFAULT_FILTER_MODE: str = "Butterworth bandpass"
DEFAULT_FILTER_LOW_CUT_HZ: float = 0.5
DEFAULT_FILTER_HIGH_CUT_HZ: float = 40.0
DEFAULT_RPEAK_METHOD: str = "hamilton2002"

TEXT_REPORT_FILENAME: str = "ecg_summary_report.txt"
JSON_REPORT_FILENAME: str = "ecg_analysis_results.json"
PRE_REPORT_FILENAME: str = "ecg_pre_report.txt"

SYNTHETIC_HEART_RATE_BPM: int = 70
SYNTHETIC_RANDOM_SEED: int = 42
