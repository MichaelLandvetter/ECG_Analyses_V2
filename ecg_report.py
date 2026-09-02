"""Report generation utilities for ECG analysis outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from ecg_config import JSON_REPORT_FILENAME, PRE_REPORT_FILENAME, TEXT_REPORT_FILENAME


def build_text_summary(analysis_results: Dict[str, Any], source: str) -> str:
    """Create a human-readable ECG summary report."""
    metrics = analysis_results.get("metrics", {})
    lines = [
        "ECG Analysis Summary",
        "====================",
        f"Source: {source}",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Sampling rate (Hz): {metrics.get('sampling_rate_hz')}",
        f"Samples: {metrics.get('num_samples')}",
        f"Duration (s): {metrics.get('duration_seconds')}",
        f"R peaks: {metrics.get('r_peak_count')}",
        f"Mean heart rate (bpm): {metrics.get('mean_heart_rate_bpm')}",
        f"RR mean (ms): {metrics.get('rr_mean_ms')}",
        f"RR std (ms): {metrics.get('rr_std_ms')}",
        f"HRV RMSSD (ms): {metrics.get('hrv_rmssd_ms')}",
        f"HRV SDNN (ms): {metrics.get('hrv_sdnn_ms')}",
    ]
    return "\n".join(lines)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def save_reports(
    analysis_results: Dict[str, Any],
    output_directory: str,
    source: str,
    write_json: bool = True,
) -> Dict[str, Path]:
    """Save text (and optional JSON) reports to disk."""
    output_dir = Path(output_directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    text_summary = build_text_summary(analysis_results, source=source)
    text_path = output_dir / TEXT_REPORT_FILENAME
    text_path.write_text(text_summary, encoding="utf-8")

    outputs = {"text_report": text_path}

    if write_json:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "results": analysis_results,
        }
        json_path = output_dir / JSON_REPORT_FILENAME
        json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        outputs["json_report"] = json_path

    return outputs


def save_pre_report(analysis_results: Dict[str, Any], output_directory: str, source: str) -> Path:
    """Write a lightweight text pre-report and return the file path."""
    output_dir = Path(output_directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / PRE_REPORT_FILENAME
    report_path.write_text(build_text_summary(analysis_results=analysis_results, source=source), encoding="utf-8")
    return report_path
