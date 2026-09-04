"""PyQt6 desktop UI for offline ECG file analysis and report review."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ecg_acquisition import acquire_ecg_signal
from ecg_analysis import (
    NeuroKit2UnavailableError,
    SUPPORTED_FILTER_MODES,
    SUPPORTED_RPEAK_METHODS,
    analyze_ecg,
    clean_ecg_signal,
    estimate_live_heart_rate_trace,
    validate_filter_mode,
    validate_filter_settings,
    validate_powerline_frequency,
)
from ecg_config import (
    DEFAULT_FILTER_HIGH_CUT_HZ,
    DEFAULT_FILTER_LOW_CUT_HZ,
    DEFAULT_FILTER_MODE,
    DEFAULT_INPUT_SOURCE,
    DEFAULT_POWERLINE_FREQUENCY_HZ,
    DEFAULT_ROLLING_WINDOW_SECONDS,
    DEFAULT_RPEAK_METHOD,
    DEFAULT_SAMPLING_RATE,
    DEFAULT_SYNTHETIC_DURATION_SECONDS,
    load_processing_settings,
    normalize_input_source,
    save_processing_settings,
)
from ecg_report import save_structured_reports

PYQT6_IMPORT_ERROR: Exception | None = None
PYQTGRAPH_IMPORT_ERROR: Exception | None = None

try:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QDialog,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressDialog,
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


def _collect_per_beat_landmark_points(
    *,
    beat_time: np.ndarray,
    beat_snippets: np.ndarray,
    beat_landmarks: list[dict[str, Any]],
    r_peaks: np.ndarray,
    beat_sample_offsets: np.ndarray,
    filtered_signal_size: int,
    sampling_rate_hz: float,
    landmark_fields: tuple[str, ...],
) -> dict[str, list[tuple[float, float]]]:
    """Collect per-beat landmark x/y points for plotting on beat snippets."""
    points_by_field: dict[str, list[tuple[float, float]]] = {field: [] for field in landmark_fields}
    if sampling_rate_hz <= 0 or beat_snippets.ndim != 2 or not beat_snippets.size:
        return points_by_field

    beat_index_to_row: dict[int, dict[str, Any]] = {}
    for row in beat_landmarks:
        beat_index = row.get("beat_index")
        try:
            beat_key = int(beat_index)
        except (TypeError, ValueError):
            continue
        beat_index_to_row[beat_key] = row

    if beat_sample_offsets.size:
        pre_samples = max(0, int(abs(np.min(beat_sample_offsets))))
        post_samples = max(0, int(np.max(beat_sample_offsets)))
    else:
        finite_time = np.isfinite(beat_time)
        if not finite_time.any():
            return points_by_field
        min_time = float(np.nanmin(beat_time[finite_time]))
        max_time = float(np.nanmax(beat_time[finite_time]))
        pre_samples = max(0, int(round(-min_time * sampling_rate_hz)))
        post_samples = max(0, int(round(max_time * sampling_rate_hz)))

    snippet_index = 0
    for beat_index, r_peak_sample in enumerate(r_peaks, start=1):
        start = int(r_peak_sample) - pre_samples
        stop = int(r_peak_sample) + post_samples + 1
        if start < 0 or stop > filtered_signal_size:
            continue
        if snippet_index >= beat_snippets.shape[0]:
            break

        row = beat_index_to_row.get(beat_index)
        beat_trace = beat_snippets[snippet_index]
        snippet_index += 1
        if row is None:
            continue

        finite_beat = np.isfinite(beat_time) & np.isfinite(beat_trace)
        if not finite_beat.any():
            continue
        x_min = float(np.nanmin(beat_time[finite_beat]))
        x_max = float(np.nanmax(beat_time[finite_beat]))

        r_value = row.get("r_peak_sample")
        try:
            r_peak_float = float(r_value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(r_peak_float):
            continue

        for field in landmark_fields:
            sample_value = row.get(field)
            try:
                sample_float = float(sample_value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(sample_float):
                continue
            marker_x = (sample_float - r_peak_float) / sampling_rate_hz
            if not np.isfinite(marker_x) or marker_x < x_min or marker_x > x_max:
                continue
            marker_y = float(np.interp(marker_x, beat_time[finite_beat], beat_trace[finite_beat]))
            points_by_field[field].append((float(marker_x), marker_y))

    return points_by_field


def _derive_action_controls(
    *,
    is_file_mode: bool,
    acquisition_running: bool,
    usb_acquisition_state: str,
    latest_result_mode: str | None,
) -> dict[str, Any]:
    """Compute mode-aware button state and labels."""
    has_results = latest_result_mode == ("file" if is_file_mode else "usb")
    controls: dict[str, Any] = {
        "start_text": "Start",
        "start_enabled": False,
        "pause_text": "Pause",
        "pause_enabled": False,
        "end_enabled": False,
        "save_enabled": has_results,
        "review_enabled": has_results,
        "show_beat_plot": is_file_mode,
    }

    if is_file_mode:
        controls["start_text"] = "Re-run Analysis" if has_results else "Start"
        controls["start_enabled"] = not acquisition_running
        return controls

    controls["pause_text"] = "Resume" if usb_acquisition_state == "paused" else "Pause"
    controls["pause_enabled"] = usb_acquisition_state in {"running", "paused"}
    controls["end_enabled"] = usb_acquisition_state in {"running", "paused"}
    controls["start_enabled"] = usb_acquisition_state == "idle" and not has_results
    return controls


if PYQT6_AVAILABLE and PYQTGRAPH_AVAILABLE:

    class ReviewPlotWidget(pg.PlotWidget):
        """Plot widget that opens an enlarged view on double-click."""

        def __init__(self, on_double_click: Callable[[], None] | None = None, parent: QWidget | None = None) -> None:
            super().__init__(parent=parent)
            self._on_double_click = on_double_click

        def mouseDoubleClickEvent(self, event: Any) -> None:
            if callable(self._on_double_click):
                self._on_double_click()
            event.accept()


    class ReviewPlotPanel(QWidget):
        """Reusable pre-report plot panel."""

        def __init__(
            self,
            title: str,
            render_plot: Callable[[pg.PlotWidget], None],
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self.title = title
            self._render_plot = render_plot

            layout = QVBoxLayout(self)
            title_label = QLabel(title)
            title_label.setStyleSheet("font-weight: bold;")
            layout.addWidget(title_label)

            self.plot_widget = ReviewPlotWidget(on_double_click=self._open_enlarged_view, parent=self)
            self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
            self.plot_widget.setMinimumHeight(180)
            layout.addWidget(self.plot_widget)
            self.refresh()

        def refresh(self) -> None:
            """Redraw the plot with the current renderer."""
            self.plot_widget.clear()
            self._render_plot(self.plot_widget)

        def _open_enlarged_view(self) -> None:
            dialog = QDialog(self)
            dialog.setWindowTitle(f"{self.title} — enlarged view")
            dialog.resize(1400, 760)

            layout = QVBoxLayout(dialog)
            expanded_plot = pg.PlotWidget()
            expanded_plot.showGrid(x=True, y=True, alpha=0.25)
            self._render_plot(expanded_plot)
            layout.addWidget(expanded_plot)
            dialog.exec()


    class PreReportReviewDialog(QDialog):
        """Detailed review window shown after analysis completes."""

        def __init__(
            self,
            analysis_results: dict[str, Any],
            input_file: str,
            source: str,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self.analysis_results = analysis_results
            self.input_file = input_file
            self.source = source

            self.setWindowTitle("ECG Detailed Review")
            self.setMinimumSize(1080, 700)
            self._fit_window_to_common_laptop_display()

            root_layout = QVBoxLayout(self)
            header = QLabel(f"Analysis complete: {Path(input_file).name}")
            header.setStyleSheet("font-size: 16px; font-weight: bold;")
            root_layout.addWidget(header)

            root_layout.addWidget(
                QLabel("Use mouse wheel to zoom and drag to pan. Double-click any plot to open a larger view.")
            )

            self.ecg_panel = ReviewPlotPanel("Full ECG: raw, filtered, and R-peaks", self._plot_ecg_overview, self)
            self.hr_panel = ReviewPlotPanel("Heart-rate over full duration", self._plot_heart_rate, self)
            self.beat_panel = ReviewPlotPanel("Beat snippets with average template overlay", self._plot_beats, self)

            root_layout.addWidget(self.ecg_panel)
            root_layout.addWidget(self.hr_panel)
            root_layout.addWidget(self.beat_panel)

            button_layout = QHBoxLayout()
            back_button = QPushButton("Back")
            back_button.clicked.connect(self.close)
            button_layout.addWidget(back_button)

            save_button = QPushButton("Save Reports")
            save_button.clicked.connect(self._save_reports)
            button_layout.addWidget(save_button)

            root_layout.addLayout(button_layout)

        def _fit_window_to_common_laptop_display(self) -> None:
            screen = self.screen() or QApplication.primaryScreen()
            if screen is None:
                self.resize(1500, 820)
                return
            available = screen.availableGeometry()
            target_width = min(1500, max(1080, available.width() - 48))
            target_height = min(820, max(700, available.height() - 64))
            self.resize(target_width, target_height)

        def _save_reports(self) -> None:
            output_paths = save_structured_reports(
                analysis_results=self.analysis_results,
                input_file=self.input_file,
                source=self.source,
                write_json=False,
            )
            folder_path = output_paths["summary_metrics_csv"].parent
            QMessageBox.information(
                self,
                "Reports saved",
                f"Saved report files to:\n{folder_path}",
            )
            self.accept()
            parent_window = self.parentWidget()
            if parent_window is not None:
                parent_window.raise_()
                parent_window.activateWindow()
                parent_window.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

        def _plot_ecg_overview(self, plot_widget: pg.PlotWidget) -> None:
            artifacts = self.analysis_results.get("artifacts", {})
            time_axis = np.asarray(artifacts.get("time_seconds", []), dtype=float)
            raw_signal = np.asarray(artifacts.get("raw_signal", []), dtype=float)
            filtered_signal = np.asarray(artifacts.get("filtered_signal", []), dtype=float)
            r_peaks = np.asarray(artifacts.get("r_peaks", []), dtype=int)

            plot_widget.setLabel("bottom", "Time (s)")
            plot_widget.setLabel("left", "Amplitude")
            plot_widget.setTitle(self.ecg_panel.title if hasattr(self, "ecg_panel") else "Full ECG")
            plot_widget.plot(time_axis, raw_signal, pen=pg.mkPen("#4C72B0", width=1.0), name="Raw ECG")
            plot_widget.plot(time_axis, filtered_signal, pen=pg.mkPen("#55A868", width=1.2), name="Filtered ECG")
            valid_r_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < filtered_signal.size)]
            if valid_r_peaks.size:
                plot_widget.plot(
                    time_axis[valid_r_peaks],
                    filtered_signal[valid_r_peaks],
                    pen=None,
                    symbol="o",
                    symbolSize=6,
                    symbolBrush=pg.mkBrush("#C44E52"),
                    symbolPen=None,
                )

        def _plot_heart_rate(self, plot_widget: pg.PlotWidget) -> None:
            artifacts = self.analysis_results.get("artifacts", {})
            time_axis = np.asarray(artifacts.get("time_seconds", []), dtype=float)
            heart_rate = np.asarray(artifacts.get("heart_rate_trace_bpm", []), dtype=float)

            plot_widget.setLabel("bottom", "Time (s)")
            plot_widget.setLabel("left", "Heart rate (BPM)")
            plot_widget.setTitle(self.hr_panel.title if hasattr(self, "hr_panel") else "Heart rate")
            plot_widget.plot(time_axis, heart_rate, pen=pg.mkPen("#8172B3", width=1.2))

        def _plot_beats(self, plot_widget: pg.PlotWidget) -> None:
            artifacts = self.analysis_results.get("artifacts", {})
            beat_time = np.asarray(artifacts.get("beat_time_offsets_seconds", []), dtype=float)
            beat_snippets = np.asarray(artifacts.get("beat_snippets", []), dtype=float)
            average_template = np.asarray(artifacts.get("average_template", []), dtype=float)
            beat_sample_offsets = np.asarray(artifacts.get("beat_sample_offsets", []), dtype=int)
            r_peaks = np.asarray(artifacts.get("r_peaks", []), dtype=int)
            filtered_signal = np.asarray(artifacts.get("filtered_signal", []), dtype=float)
            beat_landmarks = self.analysis_results.get("report_tables", {}).get("beat_morphology_landmarks", [])

            if beat_snippets.ndim == 2 and beat_snippets.size:
                for beat in beat_snippets:
                    beat_len = min(beat_time.size, beat.size)
                    if beat_len:
                        plot_widget.plot(beat_time[:beat_len], beat[:beat_len], pen=pg.mkPen("#B0B0B0", width=0.8))
            template_len = min(beat_time.size, average_template.size)
            if template_len:
                plot_widget.plot(
                    beat_time[:template_len],
                    average_template[:template_len],
                    pen=pg.mkPen("#C44E52", width=2.0),
                )
            plot_widget.setLabel("bottom", "Time from R-peak (s)")
            plot_widget.setLabel("left", "Amplitude")
            plot_widget.setTitle(self.beat_panel.title if hasattr(self, "beat_panel") else "Beat snippets")

            finite_template = np.isfinite(average_template)
            sampling_rate_hz = float(self.analysis_results.get("metrics", {}).get("sampling_rate_hz", 0) or 0)
            marker_specs = (
                ("p_peak_sample", "P", "#1F77B4", "t"),
                ("q_peak_sample", "Q", "#2CA02C", "d"),
                ("s_peak_sample", "S", "#FF7F0E", "s"),
                ("t_peak_sample", "T", "#9467BD", "o"),
            )
            per_beat_points = _collect_per_beat_landmark_points(
                beat_time=beat_time,
                beat_snippets=beat_snippets,
                beat_landmarks=beat_landmarks,
                r_peaks=r_peaks,
                beat_sample_offsets=beat_sample_offsets,
                filtered_signal_size=int(filtered_signal.size),
                sampling_rate_hz=sampling_rate_hz,
                landmark_fields=tuple(spec[0] for spec in marker_specs),
            )
            if beat_time.size == average_template.size:
                finite_wave = np.isfinite(beat_time) & finite_template
            else:
                finite_wave = np.array([], dtype=bool)
            for sample_field, label, color, symbol in marker_specs:
                points = per_beat_points.get(sample_field, [])
                if not points:
                    continue
                x_points = [point[0] for point in points]
                y_points = [point[1] for point in points]

                plot_widget.plot(
                    x_points,
                    y_points,
                    pen=None,
                    symbol=symbol,
                    symbolSize=7,
                    symbolBrush=pg.mkBrush(color),
                    symbolPen=pg.mkPen(color),
                )

                if not finite_wave.any():
                    continue
                marker_x = float(np.nanmedian(np.asarray(x_points, dtype=float)))
                marker_y = float(np.interp(marker_x, beat_time[finite_wave], average_template[finite_wave]))
                average_color = pg.mkColor(color)
                average_color.setAlpha(120)
                plot_widget.plot(
                    [marker_x],
                    [marker_y],
                    pen=None,
                    symbol=symbol,
                    symbolSize=5,
                    symbolBrush=pg.mkBrush(average_color),
                    symbolPen=pg.mkPen(average_color),
                )
                label_item = pg.TextItem(text=label, color=average_color, anchor=(0.5, 1.4))
                label_item.setPos(marker_x, marker_y)
                plot_widget.addItem(label_item)


    class ECGDesktopApp(QMainWindow):
        """Desktop UI for clean offline ECG file analysis."""

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("ECG Analysis")
            self.resize(1320, 900)

            self.selected_file: str | None = None
            self.acquisition_running = False
            self.usb_acquisition_state = "idle"
            self.usb_session_chunks: list[np.ndarray] = []
            self.usb_session_settings: dict[str, Any] | None = None
            self.usb_sample_cursor: int = 0
            self.usb_rng = np.random.default_rng()
            self.latest_report_input_file: str | None = None
            self.latest_results: dict[str, Any] | None = None
            self.latest_source = ""
            self.latest_result_mode: str | None = None
            self.latest_raw_signal: np.ndarray | None = None
            self.processing_settings = load_processing_settings()

            central_widget = QWidget(self)
            self.setCentralWidget(central_widget)
            self.root_layout = QVBoxLayout(central_widget)

            self._build_layout()
            self.usb_timer = QTimer(self)
            self.usb_timer.setInterval(500)
            self.usb_timer.timeout.connect(self._on_usb_stream_tick)
            self._apply_processing_settings(self.processing_settings)
            self._set_initial_plots()
            self._on_source_mode_changed()
            self._set_status("Ready for ECG analysis")

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
            self.source_selector.addItems(["File Analysis", "USB Input"])
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
            self.filter_mode_selector.addItems(list(SUPPORTED_FILTER_MODES))
            self.filter_mode_selector.currentTextChanged.connect(self._on_filter_mode_changed)
            layout.addWidget(self.filter_mode_selector, 0, 0, 1, 2)

            layout.addWidget(QLabel("Low cut (Hz):"), 1, 0)
            self.low_cut_input = QLineEdit(str(DEFAULT_FILTER_LOW_CUT_HZ))
            layout.addWidget(self.low_cut_input, 1, 1)

            layout.addWidget(QLabel("High cut (Hz):"), 2, 0)
            self.high_cut_input = QLineEdit(str(DEFAULT_FILTER_HIGH_CUT_HZ))
            layout.addWidget(self.high_cut_input, 2, 1)

            apply_button = QPushButton("Apply Filter")
            apply_button.clicked.connect(self._apply_filter_preview)
            layout.addWidget(apply_button, 3, 0, 1, 2)

            parent.addWidget(frame, 0, column)

        def _build_rpeak_controls(self, parent: QGridLayout, column: int) -> None:
            frame = QGroupBox("R-peak Detection")
            layout = QVBoxLayout(frame)

            self.rpeak_method_selector = QComboBox()
            self.rpeak_method_selector.addItems(list(SUPPORTED_RPEAK_METHODS))
            self.rpeak_method_selector.currentTextChanged.connect(self._sync_settings_rpeak_selector)
            layout.addWidget(self.rpeak_method_selector)

            parent.addWidget(frame, 0, column)

        def _build_action_controls(self, parent: QGridLayout, column: int) -> None:
            frame = QGroupBox("Controls")
            layout = QVBoxLayout(frame)

            self.start_button = QPushButton("Start")
            self.start_button.clicked.connect(self._start_or_rerun)
            layout.addWidget(self.start_button)

            self.pause_resume_button = QPushButton("Pause")
            self.pause_resume_button.clicked.connect(self._pause_or_resume_usb)
            layout.addWidget(self.pause_resume_button)

            self.end_button = QPushButton("End")
            self.end_button.clicked.connect(self._end_usb_session)
            layout.addWidget(self.end_button)

            self.save_reports_button = QPushButton("Save Reports")
            self.save_reports_button.clicked.connect(self._save_reports_from_main_view)
            layout.addWidget(self.save_reports_button)

            self.detailed_review_button = QPushButton("Open Pre-report")
            self.detailed_review_button.clicked.connect(self._open_detailed_review)
            layout.addWidget(self.detailed_review_button)

            reset_button = QPushButton("Reset")
            reset_button.clicked.connect(self._reset_ui)
            layout.addWidget(reset_button)

            parent.addWidget(frame, 0, column)

        def _build_analysis_tab(self, parent: QVBoxLayout) -> None:
            self.ecg_plot = pg.PlotWidget(title="Raw ECG + Filtered ECG")
            self.ecg_plot.setLabel("left", "Amplitude")
            self.ecg_plot.setLabel("bottom", "Time (s)")
            self.ecg_plot.showGrid(x=True, y=True, alpha=0.3)

            self.hr_plot = pg.PlotWidget(title="Heart Rate over full duration")
            self.hr_plot.setLabel("left", "BPM")
            self.hr_plot.setLabel("bottom", "Time (s)")
            self.hr_plot.showGrid(x=True, y=True, alpha=0.3)

            self.beat_plot = pg.PlotWidget(title="Beat snippets with average template")
            self.beat_plot.setLabel("left", "Amplitude")
            self.beat_plot.setLabel("bottom", "Time from R-peak (s)")
            self.beat_plot.showGrid(x=True, y=True, alpha=0.3)

            self.analysis_grid = QGridLayout()
            self.analysis_grid.setContentsMargins(0, 0, 0, 0)
            self.analysis_grid.setHorizontalSpacing(10)
            self.analysis_grid.setVerticalSpacing(10)
            parent.addLayout(self.analysis_grid)
            self._apply_analysis_layout(is_file_mode=True)

        def _build_settings_tab(self, parent: QVBoxLayout) -> None:
            detector_group = QGroupBox("R-peak detector method")
            detector_layout = QFormLayout(detector_group)
            self.settings_rpeak_method_selector = QComboBox()
            self.settings_rpeak_method_selector.addItems(list(SUPPORTED_RPEAK_METHODS))
            self.settings_rpeak_method_selector.currentTextChanged.connect(self._sync_top_rpeak_selector)
            detector_layout.addRow("Method:", self.settings_rpeak_method_selector)
            parent.addWidget(detector_group)

            offline_group = QGroupBox("Offline processing")
            offline_layout = QFormLayout(offline_group)
            self.sampling_rate_input = QLineEdit(str(DEFAULT_SAMPLING_RATE))
            self.powerline_input = QLineEdit(str(DEFAULT_POWERLINE_FREQUENCY_HZ))
            offline_layout.addRow("Sampling rate (Hz):", self.sampling_rate_input)
            offline_layout.addRow("Power-line notch (Hz):", self.powerline_input)
            parent.addWidget(offline_group)

            rolling_window_group = QGroupBox("Rolling window (USB only)")
            rolling_window_layout = QFormLayout(rolling_window_group)
            self.rolling_window_input = QLineEdit(str(DEFAULT_ROLLING_WINDOW_SECONDS))
            rolling_window_layout.addRow("Window length (s):", self.rolling_window_input)
            rolling_window_layout.addRow(
                QLabel("Used for USB preview buffering behavior."),
            )
            parent.addWidget(rolling_window_group)

            deferred_note = QLabel(
                "File Analysis and USB Input share the same final analysis pipeline and processing settings."
            )
            deferred_note.setWordWrap(True)
            parent.addWidget(deferred_note)

            save_button = QPushButton("Save ECG Processing Settings")
            save_button.clicked.connect(self._save_current_processing_settings)
            parent.addWidget(save_button)
            parent.addStretch(1)

        def _apply_processing_settings(self, settings: dict[str, Any]) -> None:
            self.source_selector.setCurrentText(normalize_input_source(settings.get("input_source", DEFAULT_INPUT_SOURCE)))
            self.filter_mode_selector.setCurrentText(str(settings.get("filter_mode", DEFAULT_FILTER_MODE)))
            self.low_cut_input.setText(str(settings.get("filter_low_cut_hz", DEFAULT_FILTER_LOW_CUT_HZ)))
            self.high_cut_input.setText(str(settings.get("filter_high_cut_hz", DEFAULT_FILTER_HIGH_CUT_HZ)))
            self.rpeak_method_selector.setCurrentText(str(settings.get("rpeak_method", DEFAULT_RPEAK_METHOD)))
            self.settings_rpeak_method_selector.setCurrentText(str(settings.get("rpeak_method", DEFAULT_RPEAK_METHOD)))
            self.sampling_rate_input.setText(str(settings.get("sampling_rate_hz", DEFAULT_SAMPLING_RATE)))
            self.powerline_input.setText(str(settings.get("powerline_frequency_hz", DEFAULT_POWERLINE_FREQUENCY_HZ)))
            self.rolling_window_input.setText(
                str(settings.get("rolling_window_seconds", DEFAULT_ROLLING_WINDOW_SECONDS))
            )
            self._on_filter_mode_changed()
            self._update_action_controls()

        def _set_initial_plots(self) -> None:
            self.ecg_plot.clear()
            self.hr_plot.clear()
            self.beat_plot.clear()
            self.ecg_plot.setTitle("Raw ECG + Filtered ECG preview")
            self.hr_plot.setTitle("Heart-rate plot")
            self.beat_plot.setTitle("Beat snippets with average template")

        def _set_status(self, status: str) -> None:
            self.status_label.setText(status)

        def _show_error(self, title: str, message: str) -> None:
            QMessageBox.critical(self, title, message)

        def _show_info(self, title: str, message: str) -> None:
            QMessageBox.information(self, title, message)

        def _sync_settings_rpeak_selector(self, method: str) -> None:
            if self.settings_rpeak_method_selector.currentText() == method:
                return
            self.settings_rpeak_method_selector.blockSignals(True)
            self.settings_rpeak_method_selector.setCurrentText(method)
            self.settings_rpeak_method_selector.blockSignals(False)

        def _sync_top_rpeak_selector(self, method: str) -> None:
            if self.rpeak_method_selector.currentText() == method:
                return
            self.rpeak_method_selector.blockSignals(True)
            self.rpeak_method_selector.setCurrentText(method)
            self.rpeak_method_selector.blockSignals(False)

        def _on_filter_mode_changed(self) -> None:
            is_butterworth = self.filter_mode_selector.currentText() == "Butterworth bandpass"
            self.low_cut_input.setEnabled(is_butterworth)
            self.high_cut_input.setEnabled(is_butterworth)

        def _apply_analysis_layout(self, is_file_mode: bool) -> None:
            self.analysis_grid.addWidget(self.ecg_plot, 0, 0, 1, 2 if not is_file_mode else 1)
            self.analysis_grid.addWidget(self.hr_plot, 1, 0, 1, 2 if not is_file_mode else 1)
            self.analysis_grid.addWidget(self.beat_plot, 0, 1, 2, 1)
            self.beat_plot.setVisible(is_file_mode)
            self.analysis_grid.setColumnStretch(0, 3)
            self.analysis_grid.setColumnStretch(1, 2 if is_file_mode else 0)
            self.analysis_grid.setRowStretch(0, 1)
            self.analysis_grid.setRowStretch(1, 1)

        def _on_source_mode_changed(self) -> None:
            is_file_mode = self.source_selector.currentText() == "File Analysis"
            self.file_button.setEnabled(is_file_mode)
            self._apply_analysis_layout(is_file_mode=is_file_mode)

            if not is_file_mode:
                self.file_status_label.setText(
                    "USB Input active. Use Start / Pause / Resume / End controls."
                )
                if self.usb_acquisition_state == "idle":
                    if self.latest_result_mode == "usb":
                        self._set_status("USB results ready. Save reports or Reset before starting a new session.")
                    else:
                        self._set_status("USB Input ready. Click Start to begin acquisition.")
            elif self.selected_file:
                self.file_status_label.setText(Path(self.selected_file).name)
            else:
                self.file_status_label.setText("No file selected")
            self._update_action_controls()

        def _update_action_controls(self) -> None:
            is_file_mode = self.source_selector.currentText() == "File Analysis"
            controls = _derive_action_controls(
                is_file_mode=is_file_mode,
                acquisition_running=self.acquisition_running,
                usb_acquisition_state=self.usb_acquisition_state,
                latest_result_mode=self.latest_result_mode,
            )
            self._apply_analysis_layout(is_file_mode=controls["show_beat_plot"])
            self.start_button.setText(str(controls["start_text"]))
            self.start_button.setEnabled(bool(controls["start_enabled"]))
            self.pause_resume_button.setText(str(controls["pause_text"]))
            self.pause_resume_button.setEnabled(bool(controls["pause_enabled"]))
            self.end_button.setEnabled(bool(controls["end_enabled"]))
            self.save_reports_button.setEnabled(bool(controls["save_enabled"]))
            self.detailed_review_button.setEnabled(bool(controls["review_enabled"]))

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

        def _get_current_processing_settings(self) -> dict[str, Any]:
            sampling_rate = int(self.sampling_rate_input.text())
            filter_mode = validate_filter_mode(self.filter_mode_selector.currentText())
            low_cut_hz = float(self.low_cut_input.text())
            high_cut_hz = float(self.high_cut_input.text())
            powerline_frequency_hz = float(self.powerline_input.text())
            rolling_window_seconds = int(self.rolling_window_input.text())
            rpeak_method = self.rpeak_method_selector.currentText().strip().lower()

            if sampling_rate <= 0:
                raise ValueError("Sampling rate must be greater than 0.")
            validate_powerline_frequency(powerline_frequency_hz, sampling_rate=sampling_rate)
            if filter_mode == "Butterworth bandpass":
                validate_filter_settings(
                    low_cut_hz=low_cut_hz,
                    high_cut_hz=high_cut_hz,
                    sampling_rate=sampling_rate,
                )
            if rolling_window_seconds <= 0:
                raise ValueError("Rolling window must be greater than 0 seconds.")

            return {
                "input_source": normalize_input_source(self.source_selector.currentText()),
                "sampling_rate_hz": sampling_rate,
                "filter_mode": filter_mode,
                "filter_low_cut_hz": low_cut_hz,
                "filter_high_cut_hz": high_cut_hz,
                "rpeak_method": rpeak_method,
                "powerline_frequency_hz": powerline_frequency_hz,
                "rolling_window_seconds": rolling_window_seconds,
            }

        def _load_selected_signal(self, sampling_rate: int) -> tuple[np.ndarray, str]:
            signal, source = acquire_ecg_signal(
                input_file=self.selected_file,
                sampling_rate=sampling_rate,
                duration_seconds=DEFAULT_SYNTHETIC_DURATION_SECONDS,
            )
            return np.asarray(signal, dtype=float).flatten(), source

        def _apply_filter_preview(self) -> None:
            try:
                settings = self._get_current_processing_settings()
            except ValueError as exc:
                self._show_error("Invalid processing settings", str(exc))
                return

            if not self.selected_file:
                self.processing_settings = settings
                self._set_status("Filter settings saved in the current session. Select a file to preview them.")
                return

            try:
                raw_signal, _source = self._load_selected_signal(settings["sampling_rate_hz"])
                filtered_signal = clean_ecg_signal(
                    signal=raw_signal,
                    sampling_rate=settings["sampling_rate_hz"],
                    filter_mode=settings["filter_mode"],
                    low_cut_hz=settings["filter_low_cut_hz"],
                    high_cut_hz=settings["filter_high_cut_hz"],
                    powerline_frequency_hz=settings["powerline_frequency_hz"],
                )
            except (FileNotFoundError, NeuroKit2UnavailableError, RuntimeError, ValueError) as exc:
                self._show_error("Filter preview error", str(exc))
                return

            self.processing_settings = settings
            self.latest_raw_signal = raw_signal
            self._update_preview_plots(raw_signal=raw_signal, filtered_signal=filtered_signal, results=None, sampling_rate=settings["sampling_rate_hz"])
            self._set_status(f"Preview updated with {settings['filter_mode']}")
            self._update_action_controls()

        def _start_or_rerun(self) -> None:
            if self.source_selector.currentText() == "USB Input":
                self._start_usb_acquisition()
                return
            self._start_file_analysis()

        def _start_file_analysis(self) -> None:
            if self.acquisition_running:
                self._set_status("Analysis already running.")
                return

            if not self.selected_file:
                self._show_error("Missing ECG file", "Select an ECG data file before starting analysis.")
                return

            try:
                settings = self._get_current_processing_settings()
            except ValueError as exc:
                self._show_error("Invalid processing settings", str(exc))
                return

            progress_dialog = QProgressDialog(
                "Processing offline ECG file and computing full analysis. Please wait...",
                None,
                0,
                0,
                self,
            )
            progress_dialog.setWindowTitle("Processing ECG")
            progress_dialog.setCancelButton(None)
            progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            progress_dialog.show()
            QApplication.processEvents()

            self.acquisition_running = True
            self._update_action_controls()
            self._set_status("Processing offline ECG file...")

            try:
                raw_signal, source = self._load_selected_signal(settings["sampling_rate_hz"])
                self._complete_analysis_run(
                    raw_signal=raw_signal,
                    source=source,
                    report_input_file=self.selected_file,
                    settings=settings,
                    completion_status="File analysis complete. Results updated in Analysis View.",
                    result_mode="file",
                )
            except (FileNotFoundError, NeuroKit2UnavailableError, RuntimeError, ValueError) as exc:
                self._show_error("Analysis error", str(exc))
                self._set_status(f"Analysis failed: {exc}")
                return
            finally:
                self.acquisition_running = False
                progress_dialog.close()
                self._update_action_controls()

        def _start_usb_acquisition(self) -> None:
            if self.usb_acquisition_state in {"running", "paused"}:
                self._set_status("USB acquisition already started.")
                return

            try:
                settings = self._get_current_processing_settings()
            except ValueError as exc:
                self._show_error("Invalid processing settings", str(exc))
                return

            self.usb_session_chunks = []
            self.usb_session_settings = settings
            self.usb_sample_cursor = 0
            self.usb_rng = np.random.default_rng()
            self.usb_acquisition_state = "running"
            self.acquisition_running = True
            self.latest_results = None
            self.latest_report_input_file = None
            self.latest_source = ""
            self.latest_result_mode = None
            self.latest_raw_signal = None
            self._set_initial_plots()
            self._append_usb_chunk(
                sampling_rate=settings["sampling_rate_hz"],
            )
            self.usb_timer.start()
            self._set_status("USB acquisition started.")
            self._refresh_usb_preview_plot()
            self._update_action_controls()

        def _pause_or_resume_usb(self) -> None:
            if self.usb_acquisition_state == "running":
                self.usb_acquisition_state = "paused"
                self.acquisition_running = False
                self.usb_timer.stop()
                self._set_status("USB acquisition paused.")
                self._update_action_controls()
                return

            if self.usb_acquisition_state == "paused":
                if self.usb_session_settings is None:
                    self._show_error("USB Input", "USB session settings are missing. Reset and start again.")
                    return
                self.usb_acquisition_state = "running"
                self.acquisition_running = True
                self.usb_timer.start()
                self._set_status("USB acquisition resumed.")
                self._refresh_usb_preview_plot()
                self._update_action_controls()

        def _end_usb_session(self) -> None:
            if self.usb_acquisition_state not in {"running", "paused"}:
                self._set_status("No active USB acquisition to end.")
                return

            previous_state = self.usb_acquisition_state
            session_file: Path | None = None
            self.acquisition_running = True
            self.usb_acquisition_state = "ending"
            self.usb_timer.stop()
            self._update_action_controls()
            self._set_status("Finalizing USB session and running analysis...")
            QApplication.processEvents()

            try:
                if self.usb_session_settings is None:
                    raise ValueError("USB session settings are missing. Reset and start a new session.")
                settings = self.usb_session_settings
                raw_signal = self._get_usb_session_signal()
                if raw_signal.size < 3:
                    raise ValueError("USB session is too short. Acquire more data before ending.")
                session_file = self._save_usb_session_to_file(raw_signal)
                reloaded_signal, _reloaded_source = acquire_ecg_signal(
                    input_file=str(session_file),
                    sampling_rate=settings["sampling_rate_hz"],
                    duration_seconds=DEFAULT_SYNTHETIC_DURATION_SECONDS,
                )
                source = f"usb_saved:{session_file}"
                self._complete_analysis_run(
                    raw_signal=np.asarray(reloaded_signal, dtype=float).flatten(),
                    source=source,
                    report_input_file=str(session_file),
                    settings=settings,
                    completion_status="USB session ended. Analysis results are ready in Analysis View.",
                    result_mode="usb",
                )
                self.usb_acquisition_state = "idle"
                self.acquisition_running = False
                self.usb_session_chunks = []
                self.usb_session_settings = None
                self.usb_sample_cursor = 0
                self._open_detailed_review()
            except (NeuroKit2UnavailableError, RuntimeError, ValueError, OSError) as exc:
                if session_file is not None:
                    self.usb_acquisition_state = "idle"
                    self.usb_session_chunks = []
                    self.usb_session_settings = None
                    self.usb_sample_cursor = 0
                    self._set_status(f"USB capture saved to {session_file}, but analysis failed: {exc}")
                else:
                    self.usb_acquisition_state = previous_state
                    self._set_status(f"USB processing failed: {exc}")
                self.acquisition_running = False
                self._show_error("USB processing error", str(exc))
            finally:
                self._update_action_controls()

        def _on_usb_stream_tick(self) -> None:
            if self.usb_acquisition_state != "running" or self.usb_session_settings is None:
                return
            self._append_usb_chunk(sampling_rate=self.usb_session_settings["sampling_rate_hz"])
            self._refresh_usb_preview_plot()

        def _generate_usb_chunk(self, sampling_rate: int) -> np.ndarray:
            chunk_size = max(5, int(round(0.5 * sampling_rate)))
            sample_index = np.arange(self.usb_sample_cursor, self.usb_sample_cursor + chunk_size, dtype=float)
            time_axis = sample_index / float(sampling_rate)
            self.usb_sample_cursor += chunk_size
            baseline = 0.8 * np.sin(2 * np.pi * 1.2 * time_axis)
            high_freq = 0.08 * np.sin(2 * np.pi * 23.0 * time_axis)
            noise = 0.03 * self.usb_rng.standard_normal(chunk_size)
            return baseline + high_freq + noise

        def _append_usb_chunk(self, sampling_rate: int) -> None:
            chunk = self._generate_usb_chunk(sampling_rate=sampling_rate)
            self.usb_session_chunks.append(np.asarray(chunk, dtype=float))

        def _get_usb_session_signal(self) -> np.ndarray:
            if not self.usb_session_chunks:
                return np.array([], dtype=float)
            return np.concatenate(self.usb_session_chunks)

        def _refresh_usb_preview_plot(self) -> None:
            settings = self.usb_session_settings
            if settings is None:
                return

            raw_signal = self._get_usb_session_signal()
            if raw_signal.size < 3:
                return

            try:
                filtered_signal = clean_ecg_signal(
                    signal=raw_signal,
                    sampling_rate=settings["sampling_rate_hz"],
                    filter_mode=settings["filter_mode"],
                    low_cut_hz=settings["filter_low_cut_hz"],
                    high_cut_hz=settings["filter_high_cut_hz"],
                    powerline_frequency_hz=settings["powerline_frequency_hz"],
                )
            except (NeuroKit2UnavailableError, RuntimeError, ValueError):
                filtered_signal = raw_signal.copy()

            window_samples = max(
                3,
                int(settings["sampling_rate_hz"] * max(1, int(settings["rolling_window_seconds"]))),
            )
            preview_start = max(0, raw_signal.size - window_samples)
            preview_raw = raw_signal[preview_start:]
            preview_filtered = filtered_signal[preview_start:]
            live_hr = estimate_live_heart_rate_trace(
                signal=preview_filtered,
                sampling_rate=settings["sampling_rate_hz"],
            )
            self._update_preview_plots(
                raw_signal=preview_raw,
                filtered_signal=preview_filtered,
                results=None,
                sampling_rate=settings["sampling_rate_hz"],
                time_offset_seconds=float(preview_start / settings["sampling_rate_hz"]),
                live_heart_rate=live_hr,
            )

        def _save_usb_session_to_file(self, signal: np.ndarray) -> Path:
            session_dir = Path.cwd() / "USB_Sessions"
            session_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            session_file = session_dir / f"usb_session_{timestamp}.csv"
            np.savetxt(session_file, signal, delimiter=",")
            return session_file

        def _complete_analysis_run(
            self,
            *,
            raw_signal: np.ndarray,
            source: str,
            report_input_file: str,
            settings: dict[str, Any],
            completion_status: str,
            result_mode: str,
        ) -> None:
            results = analyze_ecg(
                signal=raw_signal,
                sampling_rate=settings["sampling_rate_hz"],
                filter_mode=settings["filter_mode"],
                low_cut_hz=settings["filter_low_cut_hz"],
                high_cut_hz=settings["filter_high_cut_hz"],
                powerline_frequency_hz=settings["powerline_frequency_hz"],
                rpeak_method=settings["rpeak_method"],
            )
            self.latest_results = results
            self.latest_source = source
            self.latest_result_mode = result_mode
            self.latest_raw_signal = raw_signal
            self.latest_report_input_file = report_input_file
            self.processing_settings = settings
            self._update_preview_plots(
                raw_signal=np.asarray(results["artifacts"]["raw_signal"], dtype=float),
                filtered_signal=np.asarray(results["artifacts"]["filtered_signal"], dtype=float),
                results=results,
                sampling_rate=settings["sampling_rate_hz"],
            )
            self._show_analysis_tab()
            self._set_status(completion_status)
            self._update_action_controls()

        def _show_analysis_tab(self) -> None:
            for tab_index in range(self.tabs.count()):
                if self.tabs.tabText(tab_index) == "Analysis View":
                    self.tabs.setCurrentIndex(tab_index)
                    return
            self.tabs.setCurrentIndex(0)

        def _reset_ui(self) -> None:
            self.acquisition_running = False
            self.usb_acquisition_state = "idle"
            self.usb_timer.stop()
            self.usb_session_chunks = []
            self.usb_session_settings = None
            self.usb_sample_cursor = 0
            self.latest_results = None
            self.latest_source = ""
            self.latest_result_mode = None
            self.latest_raw_signal = None
            self.latest_report_input_file = None
            self.selected_file = None
            self.processing_settings = load_processing_settings()

            self._apply_processing_settings(self.processing_settings)
            self._on_source_mode_changed()
            self._set_initial_plots()
            self._set_status("State reset to saved settings")
            self._update_action_controls()

        def _save_current_processing_settings(self) -> None:
            try:
                settings = self._get_current_processing_settings()
                path = save_processing_settings(settings)
            except ValueError as exc:
                self._show_error("Invalid processing settings", str(exc))
                return
            except OSError as exc:
                self._show_error("Settings save error", f"Could not save settings: {exc}")
                return

            self.processing_settings = settings
            self._show_info("Settings saved", f"Saved ECG processing settings to:\n{path}")
            self._set_status(f"Saved ECG processing settings to {path.name}")

        def _save_reports_from_main_view(self) -> None:
            if not self.latest_results or not self.latest_report_input_file:
                self._show_info("Save Reports", "Run analysis first to enable report export.")
                return

            try:
                output_paths = save_structured_reports(
                    analysis_results=self.latest_results,
                    input_file=self.latest_report_input_file,
                    source=self.latest_source or f"file:{self.latest_report_input_file}",
                    write_json=False,
                )
            except OSError as exc:
                self._show_error("Report export error", f"Could not save reports: {exc}")
                self._set_status(f"Report export failed: {exc}")
                return
            folder_path = output_paths["summary_metrics_csv"].parent
            self._show_info("Reports saved", f"Saved report files to:\n{folder_path}")
            self._set_status(f"Reports saved to {folder_path}")

        def _open_detailed_review(self) -> None:
            if not self.latest_results or not self.latest_report_input_file:
                self._show_info("Open Pre-report", "Run analysis first to open the pre-report review view.")
                return

            dialog = PreReportReviewDialog(
                analysis_results=self.latest_results,
                input_file=self.latest_report_input_file,
                source=self.latest_source or f"file:{self.latest_report_input_file}",
                parent=self,
            )
            dialog.exec()
            self._set_status("Returned from pre-report review window")

        def _update_preview_plots(
            self,
            raw_signal: np.ndarray,
            filtered_signal: np.ndarray,
            results: dict[str, Any] | None,
            sampling_rate: int,
            time_offset_seconds: float = 0.0,
            live_heart_rate: np.ndarray | None = None,
        ) -> None:
            time_axis = np.arange(raw_signal.size, dtype=float) / float(sampling_rate) + float(time_offset_seconds)

            self.ecg_plot.clear()
            self.ecg_plot.plot(time_axis, raw_signal, pen=pg.mkPen("#4C72B0", width=1.0))
            self.ecg_plot.plot(time_axis, filtered_signal, pen=pg.mkPen("#55A868", width=1.2))
            if self.source_selector.currentText() == "USB Input" and results is None:
                self.ecg_plot.setTitle("Raw ECG + Filtered ECG live preview")
            else:
                self.ecg_plot.setTitle("Full ECG: raw, filtered, and R-peaks" if results else "Raw ECG + Filtered ECG preview")

            self.hr_plot.clear()
            if self.source_selector.currentText() == "USB Input" and results is None:
                self.hr_plot.setTitle("Heart-rate live trend")
            else:
                self.hr_plot.setTitle("Heart-rate over full duration")
            self.beat_plot.clear()
            self.beat_plot.setTitle("Beat snippets with average template overlay")

            if results is None:
                if live_heart_rate is not None and live_heart_rate.size:
                    finite_mask = np.isfinite(live_heart_rate)
                    if finite_mask.any():
                        self.hr_plot.plot(
                            time_axis[: live_heart_rate.size][finite_mask],
                            live_heart_rate[finite_mask],
                            pen=pg.mkPen("#8172B3", width=1.2),
                        )
                    else:
                        self.hr_plot.setTitle("Heart-rate live trend unavailable")
                return

            artifacts = results.get("artifacts", {})
            r_peaks = np.asarray(artifacts.get("r_peaks", []), dtype=int)
            heart_rate = np.asarray(artifacts.get("heart_rate_trace_bpm", []), dtype=float)

            if heart_rate.size:
                self.hr_plot.plot(time_axis[: heart_rate.size], heart_rate, pen=pg.mkPen("#8172B3", width=1.2))
            valid_r_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < filtered_signal.size)]
            if valid_r_peaks.size:
                self.ecg_plot.addItem(
                    pg.ScatterPlotItem(
                        x=time_axis[valid_r_peaks],
                        y=filtered_signal[valid_r_peaks],
                        pen=None,
                        brush=pg.mkBrush("#C44E52"),
                        size=7,
                    )
                )
            self._plot_beat_summary(plot_widget=self.beat_plot, analysis_results=results)

        def _plot_beat_summary(self, plot_widget: pg.PlotWidget, analysis_results: dict[str, Any]) -> None:
            artifacts = analysis_results.get("artifacts", {})
            beat_time = np.asarray(artifacts.get("beat_time_offsets_seconds", []), dtype=float)
            beat_snippets = np.asarray(artifacts.get("beat_snippets", []), dtype=float)
            average_template = np.asarray(artifacts.get("average_template", []), dtype=float)
            beat_sample_offsets = np.asarray(artifacts.get("beat_sample_offsets", []), dtype=int)
            r_peaks = np.asarray(artifacts.get("r_peaks", []), dtype=int)
            filtered_signal = np.asarray(artifacts.get("filtered_signal", []), dtype=float)
            beat_landmarks = analysis_results.get("report_tables", {}).get("beat_morphology_landmarks", [])

            if beat_snippets.ndim == 2 and beat_snippets.size:
                for beat in beat_snippets:
                    beat_len = min(beat_time.size, beat.size)
                    if beat_len:
                        plot_widget.plot(beat_time[:beat_len], beat[:beat_len], pen=pg.mkPen("#B0B0B0", width=0.8))
            template_len = min(beat_time.size, average_template.size)
            if template_len:
                plot_widget.plot(
                    beat_time[:template_len],
                    average_template[:template_len],
                    pen=pg.mkPen("#C44E52", width=2.0),
                )
            plot_widget.setLabel("bottom", "Time from R-peak (s)")
            plot_widget.setLabel("left", "Amplitude")

            finite_template = np.isfinite(average_template)
            sampling_rate_hz = float(analysis_results.get("metrics", {}).get("sampling_rate_hz", 0) or 0)
            marker_specs = (
                ("p_peak_sample", "P", "#1F77B4", "t"),
                ("q_peak_sample", "Q", "#2CA02C", "d"),
                ("s_peak_sample", "S", "#FF7F0E", "s"),
                ("t_peak_sample", "T", "#9467BD", "o"),
            )
            per_beat_points = _collect_per_beat_landmark_points(
                beat_time=beat_time,
                beat_snippets=beat_snippets,
                beat_landmarks=beat_landmarks,
                r_peaks=r_peaks,
                beat_sample_offsets=beat_sample_offsets,
                filtered_signal_size=int(filtered_signal.size),
                sampling_rate_hz=sampling_rate_hz,
                landmark_fields=tuple(spec[0] for spec in marker_specs),
            )
            if beat_time.size == average_template.size:
                finite_wave = np.isfinite(beat_time) & finite_template
            else:
                finite_wave = np.array([], dtype=bool)
            for sample_field, label, color, symbol in marker_specs:
                points = per_beat_points.get(sample_field, [])
                if not points:
                    continue
                x_points = [point[0] for point in points]
                y_points = [point[1] for point in points]
                plot_widget.plot(
                    x_points,
                    y_points,
                    pen=None,
                    symbol=symbol,
                    symbolSize=7,
                    symbolBrush=pg.mkBrush(color),
                    symbolPen=pg.mkPen(color),
                )
                if not finite_wave.any():
                    continue
                marker_x = float(np.nanmedian(np.asarray(x_points, dtype=float)))
                marker_y = float(np.interp(marker_x, beat_time[finite_wave], average_template[finite_wave]))
                marker_color = pg.mkColor(color)
                marker_color.setAlpha(120)
                plot_widget.plot(
                    [marker_x],
                    [marker_y],
                    pen=None,
                    symbol=symbol,
                    symbolSize=5,
                    symbolBrush=pg.mkBrush(marker_color),
                    symbolPen=pg.mkPen(marker_color),
                )
                label_item = pg.TextItem(text=label, color=marker_color, anchor=(0.5, 1.4))
                label_item.setPos(marker_x, marker_y)
                plot_widget.addItem(label_item)


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
