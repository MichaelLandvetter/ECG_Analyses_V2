"""PyQt6 desktop UI for ECG acquisition and analysis flow."""

from __future__ import annotations

import sys
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np

from ecg_acquisition import acquire_ecg_signal
from ecg_analysis import (
    NeuroKit2UnavailableError,
    analyze_ecg,
    apply_butterworth_bandpass,
    validate_filter_settings,
)
from ecg_config import (
    DEFAULT_FILTER_HIGH_CUT_HZ,
    DEFAULT_FILTER_LOW_CUT_HZ,
    DEFAULT_FILTER_MODE,
    DEFAULT_INPUT_SOURCE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RPEAK_METHOD,
    DEFAULT_SAMPLING_RATE,
    DEFAULT_SYNTHETIC_DURATION_SECONDS,
)
from ecg_report import save_pre_report

PYQT6_IMPORT_ERROR: Exception | None = None
PYQTGRAPH_IMPORT_ERROR: Exception | None = None

try:
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    PYQT6_AVAILABLE = True
except Exception as exc:  # pragma: no cover - env dependent
    PYQT6_IMPORT_ERROR = exc
    PYQT6_AVAILABLE = False

try:
    import pyqtgraph as pg

    PYQTGRAPH_AVAILABLE = True
except Exception as exc:  # pragma: no cover - env dependent
    PYQTGRAPH_IMPORT_ERROR = exc
    PYQTGRAPH_AVAILABLE = False


def _get_missing_ui_dependency_message() -> str | None:
    """Return a user-facing message when optional UI dependencies are unavailable."""
    missing: list[str] = []
    details: list[str] = []

    if not PYQT6_AVAILABLE:
        missing.append("PyQt6")
        if PYQT6_IMPORT_ERROR is not None:
            details.append(f"PyQt6 import error: {PYQT6_IMPORT_ERROR}")

    if not PYQTGRAPH_AVAILABLE:
        missing.append("pyqtgraph")
        if PYQTGRAPH_IMPORT_ERROR is not None:
            details.append(f"pyqtgraph import error: {PYQTGRAPH_IMPORT_ERROR}")

    if not missing:
        return None

    message = (
        "Desktop UI dependencies are missing: "
        + ", ".join(missing)
        + ". Install them with `pip install PyQt6 pyqtgraph`."
    )
    if details:
        message += "\n" + "\n".join(details)
    return message


if PYQT6_AVAILABLE and PYQTGRAPH_AVAILABLE:

    class ECGDesktopApp(QMainWindow):
        """Desktop UI for running the clean ECG analysis skeleton."""

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("ECG Analysis")
            self.resize(1300, 820)

            self.selected_file: str | None = None
            self.acquisition_running = False
            self.filter_armed = False
            self.latest_results: dict[str, Any] | None = None
            self.latest_source = ""

            central_widget = QWidget(self)
            self.setCentralWidget(central_widget)
            self.root_layout = QVBoxLayout(central_widget)

            self._build_layout()
            self._set_initial_plots()
            self._on_source_mode_changed()

        def _build_layout(self) -> None:
            controls_container = QWidget()
            controls_layout = QGridLayout(controls_container)
            controls_layout.setContentsMargins(0, 0, 0, 0)

            self._build_source_controls(controls_layout, 0)
            self._build_file_controls(controls_layout, 1)
            self._build_filter_controls(controls_layout, 2)
            self._build_rpeak_controls(controls_layout, 3)
            self._build_action_controls(controls_layout, 4)

            self.root_layout.addWidget(controls_container)

            self.tabs = QTabWidget()
            analysis_tab = QWidget()
            analysis_layout = QVBoxLayout(analysis_tab)
            self._build_analysis_tab(analysis_layout)

            settings_tab = QWidget()
            settings_layout = QVBoxLayout(settings_tab)
            self._build_settings_tab(settings_layout)

            self.tabs.addTab(analysis_tab, "Analysis View")
            self.tabs.addTab(settings_tab, "ECG Processing Settings")
            self.root_layout.addWidget(self.tabs)

            self.status_label = QLabel("Ready")
            self.root_layout.addWidget(self.status_label)

        def _build_source_controls(self, parent: QGridLayout, column: int) -> None:
            frame = QGroupBox("File or USB Input")
            layout = QVBoxLayout(frame)

            self.source_selector = QComboBox()
            self.source_selector.addItems(["File Replay", "USB Input"])
            self.source_selector.setCurrentText(DEFAULT_INPUT_SOURCE)
            self.source_selector.currentTextChanged.connect(self._on_source_mode_changed)
            layout.addWidget(self.source_selector)

            parent.addWidget(frame, 0, column)

        def _build_file_controls(self, parent: QGridLayout, column: int) -> None:
            frame = QGroupBox("ECG Data File")
            layout = QVBoxLayout(frame)

            self.file_status_label = QLabel("No file selected")
            self.file_status_label.setWordWrap(True)
            layout.addWidget(self.file_status_label)

            self.file_button = QPushButton("Choose ECG File")
            self.file_button.clicked.connect(self._select_ecg_file)
            layout.addWidget(self.file_button)

            parent.addWidget(frame, 0, column)

        def _build_filter_controls(self, parent: QGridLayout, column: int) -> None:
            frame = QGroupBox("ECG Filter Settings")
            layout = QGridLayout(frame)

            self.filter_mode_selector = QComboBox()
            self.filter_mode_selector.addItems(["Butterworth bandpass"])
            self.filter_mode_selector.setCurrentText(DEFAULT_FILTER_MODE)
            layout.addWidget(self.filter_mode_selector, 0, 0, 1, 2)

            layout.addWidget(QLabel("Low cut (Hz):"), 1, 0)
            self.low_cut_input = QLineEdit(str(DEFAULT_FILTER_LOW_CUT_HZ))
            layout.addWidget(self.low_cut_input, 1, 1)

            layout.addWidget(QLabel("High cut (Hz):"), 2, 0)
            self.high_cut_input = QLineEdit(str(DEFAULT_FILTER_HIGH_CUT_HZ))
            layout.addWidget(self.high_cut_input, 2, 1)

            apply_button = QPushButton("Apply Filter")
            apply_button.clicked.connect(self._arm_filter)
            layout.addWidget(apply_button, 3, 0, 1, 2)

            parent.addWidget(frame, 0, column)

        def _build_rpeak_controls(self, parent: QGridLayout, column: int) -> None:
            frame = QGroupBox("R-peak Detection")
            layout = QVBoxLayout(frame)

            self.rpeak_method_selector = QComboBox()
            self.rpeak_method_selector.addItems(["neurokit", "pantompkins1985", "engzeemod2012", "hamilton2002"])
            self.rpeak_method_selector.setCurrentText(DEFAULT_RPEAK_METHOD)
            layout.addWidget(self.rpeak_method_selector)

            parent.addWidget(frame, 0, column)

        def _build_action_controls(self, parent: QGridLayout, column: int) -> None:
            frame = QGroupBox("Controls")
            layout = QVBoxLayout(frame)

            start_button = QPushButton("Start")
            start_button.clicked.connect(self._start_analysis)
            layout.addWidget(start_button)

            stop_button = QPushButton("Stop")
            stop_button.clicked.connect(self._stop_analysis)
            layout.addWidget(stop_button)

            reset_button = QPushButton("Reset")
            reset_button.clicked.connect(self._reset_ui)
            layout.addWidget(reset_button)

            pre_report_button = QPushButton("Open Pre-report")
            pre_report_button.clicked.connect(self._open_pre_report)
            layout.addWidget(pre_report_button)

            parent.addWidget(frame, 0, column)

        def _build_analysis_tab(self, parent: QVBoxLayout) -> None:
            self.ecg_plot = pg.PlotWidget(title="Raw ECG + Filtered ECG")
            self.ecg_plot.setLabel("left", "Amplitude")
            self.ecg_plot.setLabel("bottom", "Sample")
            self.ecg_plot.showGrid(x=True, y=True, alpha=0.3)
            self.ecg_plot.addLegend()

            self.hr_plot = pg.PlotWidget(title="Heart Rate — R-peak tachometer")
            self.hr_plot.setLabel("left", "BPM")
            self.hr_plot.setLabel("bottom", "Sample")
            self.hr_plot.showGrid(x=True, y=True, alpha=0.3)
            self.hr_plot.addLegend()

            parent.addWidget(self.ecg_plot)
            parent.addWidget(self.hr_plot)

        def _build_settings_tab(self, parent: QVBoxLayout) -> None:
            message = (
                "This tab is reserved for additional acquisition and processing options. "
                "Current controls are available in the top section for a clean, incremental build."
            )
            label = QLabel(message)
            label.setWordWrap(True)
            parent.addWidget(label)

        def _set_initial_plots(self) -> None:
            self.ecg_plot.clear()
            self.hr_plot.clear()
            self.ecg_plot.setTitle("Raw ECG + Filtered ECG")
            self.hr_plot.setTitle("Heart Rate — R-peak tachometer")

        def _set_status(self, status: str) -> None:
            self.status_label.setText(status)

        def _show_error(self, title: str, message: str) -> None:
            QMessageBox.critical(self, title, message)

        def _show_info(self, title: str, message: str) -> None:
            QMessageBox.information(self, title, message)

        def _on_source_mode_changed(self) -> None:
            is_file_mode = self.source_selector.currentText() == "File Replay"
            self.file_button.setEnabled(is_file_mode)

            if not is_file_mode:
                self.file_status_label.setText("USB Input selected (placeholder)")
            elif self.selected_file:
                self.file_status_label.setText(Path(self.selected_file).name)
            else:
                self.file_status_label.setText("No file selected")

        def _select_ecg_file(self) -> None:
            file_path, _filter = QFileDialog.getOpenFileName(
                self,
                "Select ECG data file",
                "",
                "ECG files (*.txt *.csv *.npy);;All files (*.*)",
            )
            if not file_path:
                return

            self.selected_file = file_path
            self.file_status_label.setText(Path(file_path).name)
            self._set_status(f"Selected file: {file_path}")

        def _arm_filter(self) -> None:
            try:
                low_cut_hz = float(self.low_cut_input.text())
                high_cut_hz = float(self.high_cut_input.text())
                validate_filter_settings(
                    low_cut_hz=low_cut_hz,
                    high_cut_hz=high_cut_hz,
                    sampling_rate=DEFAULT_SAMPLING_RATE,
                )
            except ValueError as exc:
                self._show_error("Invalid filter settings", str(exc))
                self.filter_armed = False
                return

            self.filter_armed = True
            self._set_status("Filter settings armed and will be applied on Start")

        def _start_analysis(self) -> None:
            source_mode = self.source_selector.currentText()
            if source_mode == "USB Input":
                self._show_info("USB Input", "USB Input is not implemented yet.")
                self._set_status("USB Input placeholder selected")
                return

            if not self.selected_file:
                self._show_error("Missing ECG file", "Select an ECG data file before starting analysis.")
                return

            self.acquisition_running = True

            try:
                signal, source = acquire_ecg_signal(
                    input_file=self.selected_file,
                    sampling_rate=DEFAULT_SAMPLING_RATE,
                    duration_seconds=DEFAULT_SYNTHETIC_DURATION_SECONDS,
                )
                raw_signal = np.asarray(signal, dtype=float).flatten()
                analysis_signal = raw_signal

                if self.filter_armed:
                    low_cut_hz = float(self.low_cut_input.text())
                    high_cut_hz = float(self.high_cut_input.text())
                    analysis_signal = apply_butterworth_bandpass(
                        signal=raw_signal,
                        sampling_rate=DEFAULT_SAMPLING_RATE,
                        low_cut_hz=low_cut_hz,
                        high_cut_hz=high_cut_hz,
                    )

                results = analyze_ecg(
                    signal=analysis_signal,
                    sampling_rate=DEFAULT_SAMPLING_RATE,
                    rpeak_method=self.rpeak_method_selector.currentText(),
                )

                self.latest_results = results
                self.latest_source = source
                self._update_plots(raw_signal=raw_signal, filtered_signal=analysis_signal, results=results)

                metrics = results.get("metrics", {})
                self._set_status(
                    "Analysis completed: "
                    f"R peaks={metrics.get('r_peak_count')}, "
                    f"Mean HR={metrics.get('mean_heart_rate_bpm')} bpm"
                )
            except NeuroKit2UnavailableError as exc:
                self._show_error("Missing dependency", str(exc))
                self._set_status("Analysis failed: NeuroKit2 is required")
            except Exception as exc:
                self._show_error("Analysis error", str(exc))
                self._set_status(f"Analysis failed: {exc}")
            finally:
                self.acquisition_running = False

        def _stop_analysis(self) -> None:
            if not self.acquisition_running:
                self._set_status("Stop requested: no active acquisition loop")
                return

            self.acquisition_running = False
            self._set_status("Acquisition loop stopped")

        def _reset_ui(self) -> None:
            self.acquisition_running = False
            self.filter_armed = False
            self.latest_results = None
            self.latest_source = ""
            self.selected_file = None

            self.source_selector.setCurrentText(DEFAULT_INPUT_SOURCE)
            self.low_cut_input.setText(str(DEFAULT_FILTER_LOW_CUT_HZ))
            self.high_cut_input.setText(str(DEFAULT_FILTER_HIGH_CUT_HZ))
            self.rpeak_method_selector.setCurrentText(DEFAULT_RPEAK_METHOD)

            self._on_source_mode_changed()
            self._set_initial_plots()
            self._set_status("State reset")

        def _open_pre_report(self) -> None:
            if not self.latest_results:
                self._show_info("Pre-report", "Run Start first to generate a pre-report.")
                return

            report_path = save_pre_report(
                analysis_results=self.latest_results,
                output_directory=DEFAULT_OUTPUT_DIR,
                source=self.latest_source or "unknown",
            )
            try:
                webbrowser.open(report_path.as_uri())
            except Exception:
                pass

            self._set_status(f"Pre-report saved to {report_path}")

        def _update_plots(self, raw_signal: np.ndarray, filtered_signal: np.ndarray, results: dict[str, Any]) -> None:
            self.ecg_plot.clear()
            self.ecg_plot.addLegend()
            self.ecg_plot.plot(raw_signal, pen=pg.mkPen("#4C72B0", width=1.2), name="Raw ECG")
            self.ecg_plot.plot(filtered_signal, pen=pg.mkPen("#55A868", width=1.4), name="Filtered ECG")
            self.ecg_plot.setTitle("Raw ECG + Filtered ECG")

            hr_trace = np.asarray(results.get("artifacts", {}).get("heart_rate_trace_bpm", []), dtype=float)
            r_peaks = np.asarray(results.get("artifacts", {}).get("r_peaks", []), dtype=int)

            self.hr_plot.clear()
            self.hr_plot.addLegend()
            if hr_trace.size:
                self.hr_plot.plot(hr_trace, pen=pg.mkPen("#8172B3", width=1.4), name="Heart Rate (BPM)")
            if hr_trace.size and r_peaks.size:
                valid_peaks = r_peaks[r_peaks < hr_trace.size]
                if valid_peaks.size:
                    scatter = pg.ScatterPlotItem(
                        x=valid_peaks,
                        y=hr_trace[valid_peaks],
                        pen=None,
                        brush=pg.mkBrush("#C44E52"),
                        size=8,
                        name="R peaks",
                    )
                    self.hr_plot.addItem(scatter)
            self.hr_plot.setTitle("Heart Rate — R-peak tachometer")


def launch_ecg_ui() -> int:
    """Launch the desktop ECG UI."""
    dependency_error = _get_missing_ui_dependency_message()
    if dependency_error:
        raise RuntimeError(dependency_error)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = ECGDesktopApp()
    window.show()
    return app.exec()
