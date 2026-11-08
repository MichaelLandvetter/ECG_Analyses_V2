"""Tests for startup discovery and explicit source selection."""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from ecg_ui import ECGDesktopApp, StartupDiscoveryDialog
    from ecg_usb_discovery import DiscoveryResult
except ImportError:
    QApplication = None


@unittest.skipIf(QApplication is None, "Desktop UI dependencies are not installed")
class TestStartupFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_no_device_or_file_disables_start(self):
        window = ECGDesktopApp(DiscoveryResult((), ()))
        self.addCleanup(window.close)
        self.assertIn("none", window.source_status_label.text())
        self.assertFalse(window.start_button.isEnabled())
        self.assertFalse(window.switch_usb_action.isEnabled())

    def test_detected_device_file_selection_and_switch(self):
        window = ECGDesktopApp(DiscoveryResult(("COM4",), ()))
        self.addCleanup(window.close)
        self.assertEqual(window.input_mode, "USB Input")
        self.assertEqual(window.start_button.text(), "Start acquisition")
        self.assertFalse(window.start_button.isEnabled())
        with patch.object(QFileDialog, "getOpenFileName", return_value=("/tmp/example.csv", "")):
            window._select_ecg_file()
        self.assertEqual(window.input_mode, "File Analysis")
        self.assertIn("example.csv", window.source_status_label.text())
        self.assertEqual(window.start_button.text(), "Start analysis")
        self.assertTrue(window.start_button.isEnabled())
        window._switch_to_usb()
        self.assertEqual(window.input_mode, "USB Input")
        self.assertFalse(window.start_button.isEnabled())

    def test_rescan_does_not_override_open_file(self):
        window = ECGDesktopApp(DiscoveryResult((), ()))
        self.addCleanup(window.close)
        with patch.object(QFileDialog, "getOpenFileName", return_value=("/tmp/example.csv", "")):
            window._select_ecg_file()
        window._usb_discovery_result(DiscoveryResult(("COM4",), ()))
        self.assertEqual(window.input_mode, "File Analysis")
        self.assertTrue(window.start_button.isEnabled())
        self.assertTrue(window.switch_usb_action.isEnabled())

    def test_multiple_units_require_manual_selection(self):
        window = ECGDesktopApp(DiscoveryResult(("COM4", "COM5"), ()))
        self.addCleanup(window.close)
        self.assertFalse(window.switch_usb_action.isEnabled())
        self.assertIn("Multiple", window.usb_device_label.text())
        window._usb_discovery_result(DiscoveryResult(("COM5",), ()))
        self.assertEqual(window.input_mode, "USB Input")

    def test_startup_result_and_failure_messages(self):
        dialog = StartupDiscoveryDialog()
        self.addCleanup(dialog.close)
        dialog._show_result(DiscoveryResult(("COM4",), ()))
        self.assertIn("COM4", dialog.message.text())
        dialog._show_error("Missing pyserial")
        self.assertEqual(dialog.result.ports, ())
        self.assertIn("Missing pyserial", dialog.message.text())


if __name__ == "__main__":
    unittest.main()
