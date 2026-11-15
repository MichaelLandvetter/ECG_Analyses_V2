"""Tests for beat-to-beat heart-rate plotting."""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import ecg_ui


@unittest.skipUnless(ecg_ui.PYQT6_AVAILABLE and ecg_ui.PYQTGRAPH_AVAILABLE, "Qt plotting dependencies unavailable")
class TestHeartRatePlot(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = ecg_ui.QApplication.instance() or ecg_ui.QApplication([])

    def test_pre_report_draws_steps_at_peak_times(self) -> None:
        result = {
            "artifacts": {
                "time_seconds": [0, 1, 2, 3, 4],
                "raw_signal": [0] * 5,
                "filtered_signal": [0] * 5,
                "r_peaks": [0, 2, 4],
                "heart_rate_trace_bpm": [None, None, 60, 60, 120],
            },
            "metrics": {},
            "report_tables": {},
        }
        dialog = ecg_ui.PreReportReviewDialog(result, "test.csv", "test")
        try:
            dialog.show()
            self.app.processEvents()
            curve = dialog.hr_panel.plot_widget.listDataItems()[0]
            x, y = curve.getData()
            np.testing.assert_array_equal(x, [0, 1, 1, 2, 2, 3, 3, 4, 4])
            np.testing.assert_allclose(y, [np.nan, np.nan, np.nan, np.nan, 60, 60, 60, 60, 120], equal_nan=True)
            self.assertEqual(curve.opts["connect"], "finite")
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
