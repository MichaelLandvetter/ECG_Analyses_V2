"""Tests for optional desktop UI dependency handling."""

from __future__ import annotations

import unittest
from unittest.mock import patch


class TestECGUIOptionalDependencies(unittest.TestCase):
    """Ensure UI dependency failures are explicit and import-safe."""

    def _import_ui_module(self):
        try:
            import ecg_ui
        except ModuleNotFoundError as exc:
            self.skipTest(f"UI module dependencies unavailable in test environment: {exc}")
        return ecg_ui

    def test_missing_dependency_message_lists_required_packages(self) -> None:
        ecg_ui = self._import_ui_module()
        with patch.object(ecg_ui, "PYQT6_AVAILABLE", False), patch.object(ecg_ui, "PYQTGRAPH_AVAILABLE", False):
            message = ecg_ui._get_missing_ui_dependency_message()

        self.assertIsNotNone(message)
        self.assertIn("PyQt6", message)
        self.assertIn("pyqtgraph", message)

    def test_launch_ui_raises_runtime_error_when_pyqtgraph_missing(self) -> None:
        ecg_ui = self._import_ui_module()
        with patch.object(ecg_ui, "PYQT6_AVAILABLE", True), patch.object(ecg_ui, "PYQTGRAPH_AVAILABLE", False):
            with self.assertRaises(RuntimeError) as context:
                ecg_ui.launch_ecg_ui()

        self.assertIn("pyqtgraph", str(context.exception))


if __name__ == "__main__":
    unittest.main()
