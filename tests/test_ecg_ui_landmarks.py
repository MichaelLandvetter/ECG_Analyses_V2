"""Tests for beat landmark point collection in pre-report plotting."""

from __future__ import annotations

import unittest

import numpy as np

from ecg_ui import _collect_per_beat_landmark_points


class TestBeatLandmarkPointCollection(unittest.TestCase):
    """Ensure per-beat landmark plotting points are aligned and robust."""

    def test_collects_points_for_valid_beat_and_skips_missing_values(self) -> None:
        beat_time = np.asarray([-0.25, 0.0, 0.25], dtype=float)
        beat_snippets = np.asarray([[0.1, 1.0, 0.2]], dtype=float)
        beat_landmarks = [
            {
                "beat_index": 1,
                "r_peak_sample": 0,
                "p_peak_sample": -1,
                "q_peak_sample": 0,
                "s_peak_sample": 1,
                "t_peak_sample": 2,
            },
            {
                "beat_index": 2,
                "r_peak_sample": 5,
                "p_peak_sample": 4,
                "q_peak_sample": 5,
                "s_peak_sample": 6,
                "t_peak_sample": float("nan"),
            },
        ]
        points = _collect_per_beat_landmark_points(
            beat_time=beat_time,
            beat_snippets=beat_snippets,
            beat_landmarks=beat_landmarks,
            r_peaks=np.asarray([0, 5], dtype=int),
            beat_sample_offsets=np.asarray([-1, 0, 1], dtype=int),
            filtered_signal_size=7,
            sampling_rate_hz=4.0,
            landmark_fields=("p_peak_sample", "q_peak_sample", "s_peak_sample", "t_peak_sample"),
        )

        self.assertEqual(points["p_peak_sample"], [(-0.25, 0.1)])
        self.assertEqual(points["q_peak_sample"], [(0.0, 1.0)])
        self.assertEqual(points["s_peak_sample"], [(0.25, 0.2)])
        self.assertEqual(points["t_peak_sample"], [])


if __name__ == "__main__":
    unittest.main()
