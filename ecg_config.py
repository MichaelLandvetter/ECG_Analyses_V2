"""Central configuration defaults and settings persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_SAMPLING_RATE: int = 250
DEFAULT_SYNTHETIC_DURATION_SECONDS: int = 10
DEFAULT_OUTPUT_DIR: str = "output"
DEFAULT_INPUT_COLUMN_INDEX: int = 0
DEFAULT_INPUT_SOURCE: str = "File Replay"
DEFAULT_FILTER_MODE: str = "NeuroKit2 ecg_clean"
DEFAULT_FILTER_LOW_CUT_HZ: float = 0.5
DEFAULT_FILTER_HIGH_CUT_HZ: float = 40.0
DEFAULT_RPEAK_METHOD: str = "hamilton2002"
DEFAULT_POWERLINE_FREQUENCY_HZ: float = 50.0
DEFAULT_ROLLING_WINDOW_SECONDS: int = 10
DEFAULT_USB_PLACEHOLDER_TEXT: str = "USB streaming not implemented yet in clean build."

TEXT_REPORT_FILENAME: str = "ecg_summary_report.txt"
JSON_REPORT_FILENAME: str = "ecg_analysis_results.json"
PRE_REPORT_FILENAME: str = "ecg_pre_report.txt"
SETTINGS_JSON_FILENAME: str = "ecg_settings.json"

SYNTHETIC_HEART_RATE_BPM: int = 70
SYNTHETIC_RANDOM_SEED: int = 42

SETTINGS_JSON_PATH: Path = Path(__file__).resolve().with_name(SETTINGS_JSON_FILENAME)


def get_default_processing_settings() -> dict[str, Any]:
    """Return the built-in default processing settings."""
    return {
        "input_source": DEFAULT_INPUT_SOURCE,
        "sampling_rate_hz": DEFAULT_SAMPLING_RATE,
        "filter_mode": DEFAULT_FILTER_MODE,
        "filter_low_cut_hz": DEFAULT_FILTER_LOW_CUT_HZ,
        "filter_high_cut_hz": DEFAULT_FILTER_HIGH_CUT_HZ,
        "rpeak_method": DEFAULT_RPEAK_METHOD,
        "powerline_frequency_hz": DEFAULT_POWERLINE_FREQUENCY_HZ,
        "rolling_window_seconds": DEFAULT_ROLLING_WINDOW_SECONDS,
    }


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return fallback
    return value_float


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return fallback
    return value_int


def _coerce_choice(value: Any, fallback: str, allowed_values: set[str]) -> str:
    value_string = str(value)
    return value_string if value_string in allowed_values else fallback


def load_processing_settings(settings_path: Path | None = None) -> dict[str, Any]:
    """Load persisted processing settings, falling back to defaults when needed."""
    defaults = get_default_processing_settings()
    path = settings_path or SETTINGS_JSON_PATH

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return defaults
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return defaults

    if not isinstance(payload, dict):
        return defaults

    return {
        "input_source": _coerce_choice(
            payload.get("input_source", defaults["input_source"]),
            defaults["input_source"],
            {"File Replay", "USB Input"},
        ),
        "sampling_rate_hz": _coerce_int(payload.get("sampling_rate_hz"), defaults["sampling_rate_hz"]),
        "filter_mode": _coerce_choice(
            payload.get("filter_mode", defaults["filter_mode"]),
            defaults["filter_mode"],
            {"NeuroKit2 ecg_clean", "Butterworth bandpass"},
        ),
        "filter_low_cut_hz": _coerce_float(payload.get("filter_low_cut_hz"), defaults["filter_low_cut_hz"]),
        "filter_high_cut_hz": _coerce_float(payload.get("filter_high_cut_hz"), defaults["filter_high_cut_hz"]),
        "rpeak_method": _coerce_choice(
            payload.get("rpeak_method", defaults["rpeak_method"]),
            defaults["rpeak_method"],
            {"neurokit", "pantompkins1985", "engzeemod2012", "hamilton2002"},
        ),
        "powerline_frequency_hz": _coerce_float(
            payload.get("powerline_frequency_hz"),
            defaults["powerline_frequency_hz"],
        ),
        "rolling_window_seconds": _coerce_int(
            payload.get("rolling_window_seconds"),
            defaults["rolling_window_seconds"],
        ),
    }


def save_processing_settings(settings: dict[str, Any], settings_path: Path | None = None) -> Path:
    """Persist processing settings to the mirrored JSON file and return its path."""
    merged = get_default_processing_settings()
    merged.update(settings)

    path = settings_path or SETTINGS_JSON_PATH
    path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    return path
