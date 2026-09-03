"""Starter tests for ECG scaffold modules."""

from __future__ import annotations

import unittest

from ecg_acquisition import generate_synthetic_ecg
from ecg_analysis import NeuroKit2UnavailableError, analyze_ecg, detect_r_peaks, validate_filter_settings


class TestECGScaffold(unittest.TestCase):
    """Minimal core-flow contract tests."""

    def test_generate_synthetic_signal_length(self) -> None:
        sampling_rate = 250
        duration_seconds = 4
        signal = generate_synthetic_ecg(duration_seconds=duration_seconds, sampling_rate=sampling_rate)

        self.assertEqual(len(signal), sampling_rate * duration_seconds)

    def test_analyze_ecg_returns_metrics_contract(self) -> None:
        signal = generate_synthetic_ecg(duration_seconds=6, sampling_rate=250)

        try:
            result = analyze_ecg(signal=signal, sampling_rate=250)
        except NeuroKit2UnavailableError:
            self.skipTest("NeuroKit2 not installed in current environment")

        self.assertIn("metrics", result)
        self.assertIn("artifacts", result)
        self.assertIn("report_tables", result)
        self.assertIn("mean_heart_rate_bpm", result["metrics"])
        self.assertIn("summary_metrics", result["report_tables"])
        self.assertIn("continuous_time_series", result["report_tables"])
        self.assertIn("raw_signal", result["artifacts"])
        self.assertIn("filtered_signal", result["artifacts"])

    def test_validate_filter_settings_rejects_invalid_high_cut(self) -> None:
        with self.assertRaises(ValueError):
            validate_filter_settings(low_cut_hz=0.5, high_cut_hz=130.0, sampling_rate=250)

    def test_detect_r_peaks_rejects_unsupported_method(self) -> None:
        signal = generate_synthetic_ecg(duration_seconds=4, sampling_rate=250)
        with self.assertRaises(ValueError):
            detect_r_peaks(signal=signal, sampling_rate=250, method="unsupported_method")


if __name__ == "__main__":
    unittest.main()
