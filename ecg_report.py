"""Report generation utilities for ECG analysis outputs."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from ecg_config import JSON_REPORT_FILENAME, PRE_REPORT_FILENAME, TEXT_REPORT_FILENAME

SUMMARY_METRIC_COLUMNS: list[str] = [
    "sampling_rate_hz",
    "num_samples",
    "duration_seconds",
    "filter_mode",
    "filter_low_cut_hz",
    "filter_high_cut_hz",
    "powerline_frequency_hz",
    "rpeak_method",
    "r_peak_count",
    "mean_heart_rate_bpm",
    "rr_mean_ms",
    "rr_std_ms",
    "hrv_rmssd_ms",
    "hrv_sdnn_ms",
]
AVERAGE_TEMPLATE_COLUMNS: list[str] = [
    "sample_offset",
    "time_offset_seconds",
    "average_amplitude",
    "beats_contributed",
]
BEAT_LANDMARK_COLUMNS: list[str] = [
    "beat_index",
    "r_peak_sample",
    "r_peak_time_seconds",
    "p_onset_sample",
    "p_peak_sample",
    "q_peak_sample",
    "qrs_onset_sample",
    "qrs_offset_sample",
    "s_peak_sample",
    "t_peak_sample",
    "t_offset_sample",
]
TIME_SERIES_COLUMNS: list[str] = [
    "sample_index",
    "time_seconds",
    "raw_ecg",
    "filtered_ecg",
    "heart_rate_bpm",
    "is_r_peak",
]


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
        f"Filter mode: {metrics.get('filter_mode')}",
        f"R-peak method: {metrics.get('rpeak_method')}",
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


def _sanitize_base_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("_") or "ecg_report"


def create_report_run_directory(input_file: str, created_at: datetime | None = None) -> Path:
    """Create and return the per-run report directory next to the selected input file."""
    input_path = Path(input_file).expanduser().resolve()
    timestamp = (created_at or datetime.now()).strftime("%Y%m%d_%H%M")
    reports_dir = input_path.parent / "Reports"
    run_directory = reports_dir / f"{_sanitize_base_name(input_path)}_{timestamp}"
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory


def _normalize_rows(rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized_rows.append({field: row.get(field) for field in fieldnames})
    return normalized_rows


def write_csv_report(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> Path:
    """Write a CSV file using stable column ordering."""
    normalized_rows = _normalize_rows(rows, fieldnames=fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)
    return path


def save_structured_reports(
    analysis_results: Dict[str, Any],
    input_file: str,
    source: str,
    write_json: bool = False,
) -> Dict[str, Path]:
    """Save export-ready ECG report files in the standard per-run folder."""
    report_directory = create_report_run_directory(input_file=input_file)
    report_tables = analysis_results.get("report_tables", {})

    outputs = {
        "summary_metrics_csv": write_csv_report(
            report_directory / "summary_metrics.csv",
            fieldnames=SUMMARY_METRIC_COLUMNS,
            rows=report_tables.get("summary_metrics", []),
        ),
        "average_template_csv": write_csv_report(
            report_directory / "average_template_wave.csv",
            fieldnames=AVERAGE_TEMPLATE_COLUMNS,
            rows=report_tables.get("average_template_wave", []),
        ),
        "beat_landmarks_csv": write_csv_report(
            report_directory / "beat_morphology_landmarks.csv",
            fieldnames=BEAT_LANDMARK_COLUMNS,
            rows=report_tables.get("beat_morphology_landmarks", []),
        ),
        "continuous_time_series_csv": write_csv_report(
            report_directory / "continuous_time_series.csv",
            fieldnames=TIME_SERIES_COLUMNS,
            rows=report_tables.get("continuous_time_series", []),
        ),
    }

    text_path = report_directory / TEXT_REPORT_FILENAME
    text_path.write_text(build_text_summary(analysis_results, source=source), encoding="utf-8")
    outputs["text_report"] = text_path

    if write_json:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "results": analysis_results,
        }
        json_path = report_directory / JSON_REPORT_FILENAME
        json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        outputs["json_report"] = json_path

    return outputs


def save_reports(
    analysis_results: Dict[str, Any],
    output_directory: str,
    source: str,
    write_json: bool = True,
) -> Dict[str, Path]:
    """Save legacy text/JSON reports to a specific directory."""
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
