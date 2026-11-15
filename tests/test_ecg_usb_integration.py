"""Exercise the desktop recording lifecycle using a simulated serial device."""

import os
import tempfile
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from ecg_analysis import NeuroKit2UnavailableError, clean_ecg_signal

try:
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from ecg_ui import ECGDesktopApp
    from ecg_usb_discovery import DiscoveryResult
    from ecg_usb_stream import USBStreamError
    from tools.usb_ecg_stream_utils import StreamSample
except ImportError:
    QApplication = None


class FakeStream:
    failure = None

    def __init__(self, port, serial_factory):
        self.index = 0

    def identify(self):
        pass

    def read_samples(self):
        time.sleep(0.005)
        self.index += 1
        if self.failure == "disconnect" and self.index == 10:
            raise USBStreamError("device disconnected")
        index = self.index + (1 if self.failure == "gap" and self.index >= 10 else 0)
        return [StreamSample(index * 4, index, 10000 + index)]

    def close(self):
        pass


@unittest.skipIf(QApplication is None, "Desktop UI dependencies are not installed")
class TestUSBIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait_for(self, condition):
        deadline = time.monotonic() + 4
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.app.processEvents()
        self.assertTrue(condition(), "Timed out waiting for acquisition")

    def test_record_pause_resume_end_and_analyse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.csv"
            window = ECGDesktopApp(DiscoveryResult(("COM4",), ()))
            with (
                patch("ecg_ui.ECGSerialStream", FakeStream),
                patch.object(QFileDialog, "getSaveFileName", return_value=(str(path), "")),
                patch.object(window, "_complete_analysis_run") as analysis,
                patch.object(window, "_open_detailed_review") as review,
            ):
                window._start_usb_acquisition()
                self.assertEqual(window.usb_preview.maxlen, 250 * (window.usb_session_settings["rolling_window_seconds"] + 6))
                self.wait_for(lambda: window.usb_recording.sample_count >= 5)
                self.wait_for(lambda: window.usb_acquisition_state == "running")
                window._pause_or_resume_usb()
                count = window.usb_recording.sample_count
                time.sleep(0.08)
                self.app.processEvents()
                self.assertEqual(window.usb_recording.sample_count, count)
                self.assertFalse(window.open_file_action.isEnabled())
                window._pause_or_resume_usb()
                self.wait_for(lambda: window.usb_recording.sample_count > count + 5)
                window._end_usb_session()
                self.wait_for(lambda: window.usb_worker is None)
                analysis.assert_called_once()
                review.assert_called_once()
                self.assertIn("acquisition resumed after pause", path.read_text())
                np.testing.assert_array_equal(
                    analysis.call_args.kwargs["raw_signal"], np.loadtxt(path, delimiter=",")[:, 0]
                )
            window.close()

    def test_preview_uses_context_and_delays_both_signals(self):
        window = ECGDesktopApp(DiscoveryResult(("COM4",), ()))
        window.usb_session_settings = {
            "sampling_rate_hz": 250,
            "rolling_window_seconds": 10,
            "filter_mode": "NeuroKit2 ecg_clean",
            "filter_low_cut_hz": 0.5,
            "filter_high_cut_hz": 40.0,
            "powerline_frequency_hz": 50.0,
        }
        try:
            with (
                patch("ecg_ui.clean_ecg_signal", side_effect=lambda signal, **kwargs: signal + 10) as clean,
                patch("ecg_ui.estimate_live_heart_rate_trace", side_effect=lambda signal, **kwargs: signal + 20) as hr,
                patch.object(window, "_update_preview_plots") as plot,
            ):
                window.usb_preview = deque(range(752), maxlen=4000)
                window._refresh_usb_preview_plot()
                clean.assert_not_called()
                plot.assert_not_called()

                window.usb_preview = deque(range(4000), maxlen=4000)
                window._refresh_usb_preview_plot()
                np.testing.assert_array_equal(clean.call_args.kwargs["signal"], np.arange(4000))
                np.testing.assert_array_equal(plot.call_args.kwargs["raw_signal"], np.arange(750, 3250))
                np.testing.assert_array_equal(plot.call_args.kwargs["filtered_signal"], np.arange(750, 3250) + 10)
                np.testing.assert_array_equal(hr.call_args.kwargs["signal"], np.arange(750, 3250) + 10)
                np.testing.assert_array_equal(plot.call_args.kwargs["live_heart_rate"], np.arange(750, 3250) + 30)

                window.usb_preview = deque(range(1250), maxlen=4000)
                window._refresh_usb_preview_plot()
                np.testing.assert_array_equal(plot.call_args.kwargs["raw_signal"], np.arange(500))
                np.testing.assert_array_equal(plot.call_args.kwargs["filtered_signal"], np.arange(500) + 10)
        finally:
            window.close()

    def test_preview_filter_failure_reports_error_and_keeps_signals_aligned(self):
        window = ECGDesktopApp(DiscoveryResult(("COM4",), ()))
        window.usb_session_settings = {
            "sampling_rate_hz": 250,
            "rolling_window_seconds": 10,
            "filter_mode": "NeuroKit2 ecg_clean",
            "filter_low_cut_hz": 0.5,
            "filter_high_cut_hz": 40.0,
            "powerline_frequency_hz": 50.0,
        }
        window.usb_preview = deque(range(1250), maxlen=4000)
        try:
            with (
                patch("ecg_ui.clean_ecg_signal", side_effect=RuntimeError("test error")),
                patch("ecg_ui.estimate_live_heart_rate_trace", return_value=np.full(500, np.nan)),
                patch.object(window, "_update_preview_plots") as plot,
            ):
                window._refresh_usb_preview_plot()
                self.assertIn("Live filter unavailable: test error", window.status_label.text())
                np.testing.assert_array_equal(plot.call_args.kwargs["raw_signal"], np.arange(500))
                np.testing.assert_array_equal(plot.call_args.kwargs["filtered_signal"], np.arange(500))
        finally:
            window.close()

    def test_preview_renders_neurokit_clean_with_delayed_aligned_samples(self):
        window = ECGDesktopApp(DiscoveryResult(("COM4",), ()))
        window.usb_session_settings = {
            "sampling_rate_hz": 250,
            "rolling_window_seconds": 3,
            "filter_mode": "NeuroKit2 ecg_clean",
            "filter_low_cut_hz": 0.5,
            "filter_high_cut_hz": 40.0,
            "powerline_frequency_hz": 50.0,
        }
        time_axis = np.arange(2250) / 250
        raw = 10000 + 1000 * np.sin(2 * np.pi * time_axis) + 100 * np.sin(2 * np.pi * 50 * time_axis)
        try:
            try:
                filtered = clean_ecg_signal(raw, 250, filter_mode="NeuroKit2 ecg_clean")
            except NeuroKit2UnavailableError:
                self.skipTest("NeuroKit2 not installed in current environment")
            window.usb_preview = deque(raw, maxlen=2250)
            window._refresh_usb_preview_plot()
            self.app.processEvents()
            raw_curve, filtered_curve = window.ecg_plot.listDataItems()
            x_raw, y_raw = raw_curve.getData()
            x_filtered, y_filtered = filtered_curve.getData()
            np.testing.assert_array_equal(x_raw, np.arange(750) / 250)
            np.testing.assert_array_equal(x_filtered, x_raw)
            np.testing.assert_allclose(y_raw, raw[750:1500])
            np.testing.assert_allclose(y_filtered, filtered[750:1500])
        finally:
            window.close()

    def test_live_preview_reuses_curves_and_clears_unavailable_heart_rate(self):
        window = ECGDesktopApp(DiscoveryResult(("COM4",), ()))
        raw = np.array([1.0, 2.0, 3.0])
        filtered = raw - 1
        try:
            self.assertEqual(window.usb_timer.interval(), 100)
            window._update_preview_plots(raw, filtered, None, 250, live_heart_rate=np.array([60.0, 61.0, 62.0]))
            raw_curve, filtered_curve = window.ecg_plot.listDataItems()
            hr_curve = window.hr_plot.listDataItems()[0]

            window._update_preview_plots(raw + 1, filtered + 1, None, 250, live_heart_rate=np.full(3, np.nan))
            self.assertEqual(window.ecg_plot.listDataItems(), [raw_curve, filtered_curve])
            self.assertEqual(window.hr_plot.listDataItems(), [hr_curve])
            np.testing.assert_array_equal(raw_curve.getData()[1], raw + 1)
            np.testing.assert_array_equal(filtered_curve.getData()[1], filtered + 1)
            hr_values = hr_curve.getData()[1]
            self.assertTrue(hr_values is None or hr_values.size == 0)
            self.assertEqual(window.hr_plot.plotItem.titleLabel.text, "Heart-rate live trend unavailable")

            window._update_preview_plots(raw, filtered, None, 250, live_heart_rate=np.array([60.0, 61.0, 62.0]))
            np.testing.assert_array_equal(hr_curve.getData()[1], [60.0, 61.0, 62.0])
            self.assertEqual(window.hr_plot.plotItem.titleLabel.text, "Heart-rate live trend")

            window._set_initial_plots()
            window._update_preview_plots(raw, filtered, None, 250, live_heart_rate=np.full(3, np.nan))
            self.assertIsNot(window.ecg_plot.listDataItems()[0], raw_curve)
            self.assertEqual(len(window.ecg_plot.listDataItems()), 2)
        finally:
            window.close()

    def test_discontinuity_and_disconnect_preserve_without_analysis(self):
        for failure in ("gap", "disconnect"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "capture.csv"
                window = ECGDesktopApp(DiscoveryResult(("COM4",), ()))
                FakeStream.failure = failure
                try:
                    with (
                        patch("ecg_ui.ECGSerialStream", FakeStream),
                        patch.object(QFileDialog, "getSaveFileName", return_value=(str(path), "")),
                        patch.object(window, "_complete_analysis_run") as analysis,
                        patch.object(window, "_show_error") as error,
                    ):
                        window._start_usb_acquisition()
                        self.wait_for(lambda: window.usb_worker is None)
                        self.assertTrue(path.exists())
                        self.assertGreater(path.stat().st_size, 0)
                        analysis.assert_not_called()
                        self.assertIn("preserved", error.call_args.args[1])
                finally:
                    FakeStream.failure = None
                    window.close()


if __name__ == "__main__":
    unittest.main()
