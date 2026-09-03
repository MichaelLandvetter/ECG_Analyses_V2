"""Tests for report export and settings persistence helpers."""

from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from ecg_config import load_processing_settings, save_processing_settings
from ecg_report import (
    AVERAGE_TEMPLATE_COLUMNS,
    BEAT_LANDMARK_COLUMNS,
    SUMMARY_METRIC_COLUMNS,
    TIME_SERIES_COLUMNS,
    create_report_run_directory,
    save_structured_reports,
)


class TestECGReportHelpers(unittest.TestCase):
    """Validate core non-UI export behavior."""

    def test_save_and_load_processing_settings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "ecg_settings.json"
            settings = {
                "sampling_rate_hz": 500,
                "filter_mode": "Butterworth bandpass",
                "filter_low_cut_hz": 1.0,
                "filter_high_cut_hz": 35.0,
                "rpeak_method": "neurokit",
                "powerline_frequency_hz": 60.0,
                "rolling_window_seconds": 12,
            }

            save_processing_settings(settings, settings_path=settings_path)
            loaded_settings = load_processing_settings(settings_path=settings_path)

        self.assertEqual(loaded_settings["sampling_rate_hz"], 500)
        self.assertEqual(loaded_settings["filter_mode"], "Butterworth bandpass")
        self.assertEqual(loaded_settings["rpeak_method"], "neurokit")
        self.assertEqual(loaded_settings["powerline_frequency_hz"], 60.0)

    def test_load_processing_settings_falls_back_when_json_is_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "ecg_settings.json"
            settings_path.write_text("{bad json", encoding="utf-8")

            loaded_settings = load_processing_settings(settings_path=settings_path)

        self.assertIn("sampling_rate_hz", loaded_settings)
        self.assertIn("filter_mode", loaded_settings)
        self.assertIn("rpeak_method", loaded_settings)

    def test_create_report_run_directory_uses_reports_folder_and_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "example input.csv"
            input_file.write_text("1\n2\n3\n", encoding="utf-8")

            report_directory = create_report_run_directory(
                input_file=str(input_file),
                created_at=datetime(2026, 1, 2, 3, 4),
            )

        self.assertEqual(report_directory.parent.name, "Reports")
        self.assertEqual(report_directory.name, "example_input_20260102_0304")

    def test_save_structured_reports_writes_expected_csv_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "recording.csv"
            input_file.write_text("0.1\n0.2\n0.3\n", encoding="utf-8")
            analysis_results = {
                "metrics": {
                    "sampling_rate_hz": 250,
                    "num_samples": 3,
                    "duration_seconds": 0.012,
                    "filter_mode": "NeuroKit2 ecg_clean",
                    "filter_low_cut_hz": 0.5,
                    "filter_high_cut_hz": 40.0,
                    "powerline_frequency_hz": 50.0,
                    "rpeak_method": "hamilton2002",
                    "r_peak_count": 1,
                    "mean_heart_rate_bpm": 72.0,
                    "rr_mean_ms": None,
                    "rr_std_ms": None,
                    "hrv_rmssd_ms": None,
                    "hrv_sdnn_ms": None,
                },
                "report_tables": {
                    "summary_metrics": [
                        {
                            "sampling_rate_hz": 250,
                            "num_samples": 3,
                            "duration_seconds": 0.012,
                            "filter_mode": "NeuroKit2 ecg_clean",
                            "filter_low_cut_hz": 0.5,
                            "filter_high_cut_hz": 40.0,
                            "powerline_frequency_hz": 50.0,
                            "rpeak_method": "hamilton2002",
                            "r_peak_count": 1,
                            "mean_heart_rate_bpm": 72.0,
                            "rr_mean_ms": None,
                            "rr_std_ms": None,
                            "hrv_rmssd_ms": None,
                            "hrv_sdnn_ms": None,
                        }
                    ],
                    "average_template_wave": [
                        {
                            "sample_offset": -1,
                            "time_offset_seconds": -0.004,
                            "average_amplitude": 0.2,
                            "beats_contributed": 1,
                        }
                    ],
                    "beat_morphology_landmarks": [
                        {
                            "beat_index": 1,
                            "r_peak_sample": 1,
                            "r_peak_time_seconds": 0.004,
                            "p_onset_sample": None,
                            "p_peak_sample": None,
                            "q_peak_sample": None,
                            "qrs_onset_sample": None,
                            "qrs_offset_sample": None,
                            "s_peak_sample": None,
                            "t_peak_sample": None,
                            "t_offset_sample": None,
                        }
                    ],
                    "continuous_time_series": [
                        {
                            "sample_index": 0,
                            "time_seconds": 0.0,
                            "raw_ecg": 0.1,
                            "filtered_ecg": 0.1,
                            "heart_rate_bpm": 72.0,
                            "is_r_peak": 0,
                        }
                    ],
                },
            }

            output_paths = save_structured_reports(
                analysis_results=analysis_results,
                input_file=str(input_file),
                source=f"file:{input_file}",
            )

            self.assertTrue(output_paths["summary_metrics_csv"].exists())
            self.assertTrue(output_paths["average_template_csv"].exists())
            self.assertTrue(output_paths["beat_landmarks_csv"].exists())
            self.assertTrue(output_paths["continuous_time_series_csv"].exists())

            with output_paths["summary_metrics_csv"].open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, SUMMARY_METRIC_COLUMNS)

            with output_paths["average_template_csv"].open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, AVERAGE_TEMPLATE_COLUMNS)

            with output_paths["beat_landmarks_csv"].open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, BEAT_LANDMARK_COLUMNS)

            with output_paths["continuous_time_series_csv"].open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, TIME_SERIES_COLUMNS)


if __name__ == "__main__":
    unittest.main()
