"""Starter tests for ECG scaffold modules."""

from __future__ import annotations

import unittest

from ecg_acquisition import generate_synthetic_ecg
from ecg_analysis import NeuroKit2UnavailableError, analyze_ecg


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
        self.assertIn("mean_heart_rate_bpm", result["metrics"])


if __name__ == "__main__":
    unittest.main()
