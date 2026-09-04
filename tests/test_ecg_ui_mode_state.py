"""Tests for mode-aware UI control state derivation."""

from __future__ import annotations

import unittest

from ecg_ui import _derive_action_controls, _should_process_usb_stream_tick


class TestECGUIModeState(unittest.TestCase):
    """Validate File/USB control state transitions."""

    def test_file_mode_shows_beat_plot_and_disables_usb_controls(self) -> None:
        controls = _derive_action_controls(
            is_file_mode=True,
            acquisition_running=False,
            usb_acquisition_state="idle",
            latest_result_mode=None,
        )
        self.assertTrue(controls["show_beat_plot"])
        self.assertTrue(controls["start_enabled"])
        self.assertFalse(controls["pause_enabled"])
        self.assertFalse(controls["end_enabled"])

    def test_usb_mode_hides_beat_plot_and_supports_pause_resume_labels(self) -> None:
        running_controls = _derive_action_controls(
            is_file_mode=False,
            acquisition_running=True,
            usb_acquisition_state="running",
            latest_result_mode=None,
        )
        paused_controls = _derive_action_controls(
            is_file_mode=False,
            acquisition_running=False,
            usb_acquisition_state="paused",
            latest_result_mode=None,
        )
        self.assertFalse(running_controls["show_beat_plot"])
        self.assertEqual(running_controls["pause_text"], "Pause")
        self.assertEqual(paused_controls["pause_text"], "Resume")
        self.assertTrue(running_controls["end_enabled"])
        self.assertTrue(paused_controls["end_enabled"])

    def test_usb_completed_results_keep_reports_enabled_and_allow_new_start(self) -> None:
        controls = _derive_action_controls(
            is_file_mode=False,
            acquisition_running=False,
            usb_acquisition_state="idle",
            latest_result_mode="usb",
        )
        self.assertTrue(controls["start_enabled"])
        self.assertTrue(controls["save_enabled"])
        self.assertTrue(controls["review_enabled"])

    def test_usb_stream_tick_runs_only_when_session_is_running(self) -> None:
        self.assertTrue(
            _should_process_usb_stream_tick(
                usb_acquisition_state="running",
                usb_session_settings={"sampling_rate_hz": 250},
            )
        )
        self.assertFalse(
            _should_process_usb_stream_tick(
                usb_acquisition_state="paused",
                usb_session_settings={"sampling_rate_hz": 250},
            )
        )
        self.assertFalse(
            _should_process_usb_stream_tick(
                usb_acquisition_state="running",
                usb_session_settings=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
