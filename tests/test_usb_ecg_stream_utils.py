"""Unit tests for standalone USB ECG stream tester utilities."""

from __future__ import annotations

import unittest

from tools.usb_ecg_stream_utils import StreamDiagnostics, estimate_dropped_samples, parse_stream_line


class TestUSBECGStreamUtils(unittest.TestCase):
    """Validate parser and dropped-sample detection logic."""

    def test_parse_stream_line_parses_valid_csv(self) -> None:
        sample = parse_stream_line("1234,56,2048")
        self.assertEqual(sample.timestamp_ms, 1234)
        self.assertEqual(sample.sample_index, 56)
        self.assertEqual(sample.raw_ecg, 2048)

    def test_parse_stream_line_rejects_invalid_shape(self) -> None:
        with self.assertRaises(ValueError):
            parse_stream_line("1234,56")

    def test_parse_stream_line_rejects_non_integer_fields(self) -> None:
        with self.assertRaises(ValueError):
            parse_stream_line("1234,abc,2048")

    def test_estimate_dropped_samples_detects_gap(self) -> None:
        self.assertEqual(estimate_dropped_samples(100, 104), 3)

    def test_estimate_dropped_samples_handles_rollover(self) -> None:
        self.assertEqual(estimate_dropped_samples((2**32) - 2, 1), 2)

    def test_stream_diagnostics_accumulates_counts(self) -> None:
        diagnostics = StreamDiagnostics(sample_rate_hz=250)
        diagnostics.observe_sample(parse_stream_line("0,0,100"), host_time=1.0)
        diagnostics.observe_sample(parse_stream_line("4,1,101"), host_time=1.004)
        diagnostics.observe_sample(parse_stream_line("12,4,102"), host_time=1.012)

        self.assertEqual(diagnostics.total_samples, 3)
        self.assertEqual(diagnostics.dropped_samples, 2)
        self.assertGreater(diagnostics.incoming_rate_hz, 0.0)
        self.assertIsNotNone(diagnostics.interval_stats)


if __name__ == "__main__":
    unittest.main()
