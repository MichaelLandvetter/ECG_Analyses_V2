"""Standalone USB ECG stream tester for Raspberry Baguette S3 / ESP32-S3 ECG streaming."""

from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pyqtgraph as pg
import serial
from serial import SerialException
from serial.tools import list_ports

from ecg_analysis import (
    apply_butterworth_bandpass,
    apply_powerline_notch,
    estimate_live_heart_rate_trace,
    validate_filter_settings,
    validate_powerline_frequency,
)
from tools.usb_ecg_stream_utils import StreamDiagnostics, StreamSample, parse_stream_line

try:
    from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal as Signal
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "PyQt5"
except ImportError:
    from PySide6.QtCore import QThread, QTimer, Qt, Signal
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "PySide6"

DEFAULT_SAMPLE_RATE_HZ = 250
DEFAULT_BAUD_RATE = 230400
DEFAULT_ADC_PIN_LABEL = "GPIO2 / ADC1"
DEFAULT_WINDOW_SECONDS = 10
PLOT_REFRESH_MS = 40
FILTER_NONE = "None"
FILTER_BUTTERWORTH = "Butterworth bandpass"
BAUD_OPTIONS = ["115200", "230400", "460800", "921600"]


@dataclass
class PendingBatch:
    """Pending samples awaiting timer-based UI processing."""

    samples: list[StreamSample]
    host_times: list[float]


class SerialReaderThread(QThread):
    """Read and parse CSV samples from a serial port in the background."""

    sample_received = Signal(object, float)
    parse_error = Signal(str)
    status_message = Signal(str)
    serial_error = Signal(str)

    def __init__(self, port: str, baud_rate: int) -> None:
        super().__init__()
        self._port = port
        self._baud_rate = baud_rate
        self._serial: serial.Serial | None = None

    def run(self) -> None:
        try:
            self._serial = serial.Serial(self._port, self._baud_rate, timeout=0.2)
            self.status_message.emit(f"Connected to {self._port} @ {self._baud_rate} baud")
        except Exception as exc:
            self.serial_error.emit(f"Failed to open serial port: {exc}")
            return

        try:
            while not self.isInterruptionRequested():
                try:
                    raw_line = self._serial.readline()
                except SerialException as exc:
                    self.serial_error.emit(f"Serial read error: {exc}")
                    return

                if not raw_line:
                    continue

                host_time = time.monotonic()
                try:
                    decoded = raw_line.decode("utf-8").strip()
                except UnicodeDecodeError:
                    self.parse_error.emit("Received undecodable serial bytes")
                    continue

                try:
                    sample = parse_stream_line(decoded)
                except ValueError as exc:
                    self.parse_error.emit(str(exc))
                    continue

                self.sample_received.emit(sample, host_time)
        finally:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:
                    pass
            self._serial = None

    def stop(self) -> None:
        """Stop the thread and close the serial port."""
        self.requestInterruption()
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
        self.wait(1500)


class USBECGStreamTesterWindow(QMainWindow):
    """Standalone live USB ECG stream tester."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("USB ECG Stream Tester")
        self.resize(1400, 900)

        self.reader_thread: SerialReaderThread | None = None
        self.connected_port: str | None = None
        self.capture_state = "stopped"
        self.pending_samples: deque[tuple[StreamSample, float]] = deque()
        self.diagnostics = StreamDiagnostics(sample_rate_hz=DEFAULT_SAMPLE_RATE_HZ)
        self.record_path: Path | None = None
        self.record_file = None

        self.raw_samples: deque[float] = deque()
        self.filtered_samples: deque[float] = deque()
        self.hr_samples: deque[float] = deque()
        self.time_seconds: deque[float] = deque()

        self._build_ui()
        self._set_buffer_capacity()
        self._connect_signals()
        self.refresh_ports()
        self._refresh_capture_controls()
        self._refresh_filter_controls()

        self.plot_timer = QTimer(self)
        self.plot_timer.setInterval(PLOT_REFRESH_MS)
        self.plot_timer.timeout.connect(self._process_pending_samples)
        self.plot_timer.start()

    def _build_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        controls_layout = QGridLayout()

        serial_group = QGroupBox("Serial controls")
        serial_layout = QFormLayout(serial_group)
        self.port_combo = QComboBox()
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(BAUD_OPTIONS)
        self.baud_combo.setCurrentText(str(DEFAULT_BAUD_RATE))
        self.refresh_button = QPushButton("Refresh ports")
        self.connect_button = QPushButton("Connect")
        serial_button_row = QHBoxLayout()
        serial_button_row.addWidget(self.refresh_button)
        serial_button_row.addWidget(self.connect_button)
        serial_layout.addRow("COM port", self.port_combo)
        serial_layout.addRow("Baud", self.baud_combo)
        serial_layout.addRow(serial_button_row)

        acquisition_group = QGroupBox("Acquisition controls")
        acquisition_layout = QFormLayout(acquisition_group)
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(50, 2000)
        self.sample_rate_spin.setValue(DEFAULT_SAMPLE_RATE_HZ)
        self.window_seconds_spin = QSpinBox()
        self.window_seconds_spin.setRange(2, 60)
        self.window_seconds_spin.setValue(DEFAULT_WINDOW_SECONDS)
        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Resume")
        self.stop_button = QPushButton("Stop")
        self.reset_button = QPushButton("Reset view/state")
        self.record_button = QPushButton("Select CSV output")
        self.record_label = QLabel("No CSV file selected")
        acquisition_button_row = QHBoxLayout()
        acquisition_button_row.addWidget(self.start_button)
        acquisition_button_row.addWidget(self.pause_button)
        acquisition_button_row.addWidget(self.resume_button)
        acquisition_button_row.addWidget(self.stop_button)
        acquisition_button_row.addWidget(self.reset_button)
        acquisition_layout.addRow("Configured fs (Hz)", self.sample_rate_spin)
        acquisition_layout.addRow("Rolling window (s)", self.window_seconds_spin)
        acquisition_layout.addRow(acquisition_button_row)
        acquisition_layout.addRow(self.record_button, self.record_label)

        filter_group = QGroupBox("Filtering controls")
        filter_layout = QFormLayout(filter_group)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems([FILTER_NONE, FILTER_BUTTERWORTH])
        self.low_cut_spin = QSpinBox()
        self.low_cut_spin.setRange(1, 100)
        self.low_cut_spin.setValue(5)
        self.high_cut_spin = QSpinBox()
        self.high_cut_spin.setRange(5, 200)
        self.high_cut_spin.setValue(40)
        self.notch_checkbox = QCheckBox("Enable 50 Hz notch")
        self.notch_checkbox.setChecked(True)
        filter_layout.addRow("Filter", self.filter_combo)
        filter_layout.addRow("Bandpass low cut (Hz)", self.low_cut_spin)
        filter_layout.addRow("Bandpass high cut (Hz)", self.high_cut_spin)
        filter_layout.addRow(self.notch_checkbox)

        telemetry_group = QGroupBox("Telemetry / diagnostics")
        telemetry_layout = QFormLayout(telemetry_group)
        self.live_rate_label = QLabel("0.0 Hz")
        self.total_samples_label = QLabel("0")
        self.dropped_samples_label = QLabel("0")
        self.interval_stats_label = QLabel("n/a")
        self.parse_errors_label = QLabel("0")
        telemetry_layout.addRow("Incoming sample rate", self.live_rate_label)
        telemetry_layout.addRow("Total samples received", self.total_samples_label)
        telemetry_layout.addRow("Dropped samples estimate", self.dropped_samples_label)
        telemetry_layout.addRow("Timestamp jitter / intervals", self.interval_stats_label)
        telemetry_layout.addRow("Serial parse errors", self.parse_errors_label)

        controls_layout.addWidget(serial_group, 0, 0)
        controls_layout.addWidget(acquisition_group, 0, 1)
        controls_layout.addWidget(filter_group, 1, 0)
        controls_layout.addWidget(telemetry_group, 1, 1)
        main_layout.addLayout(controls_layout)

        self.raw_plot = pg.PlotWidget(title="Raw ECG")
        self.filtered_plot = pg.PlotWidget(title="Filtered ECG")
        self.hr_plot = pg.PlotWidget(title="Live HR / Tachometer")
        for plot in (self.raw_plot, self.filtered_plot, self.hr_plot):
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setLabel("bottom", "Time", units="s")
        self.raw_plot.setLabel("left", "ADC")
        self.filtered_plot.setLabel("left", "Filtered")
        self.hr_plot.setLabel("left", "BPM")

        self.raw_curve = self.raw_plot.plot(pen=pg.mkPen("#00bcd4", width=1.5))
        self.filtered_curve = self.filtered_plot.plot(pen=pg.mkPen("#4caf50", width=1.5))
        self.hr_curve = self.hr_plot.plot(pen=pg.mkPen("#ff9800", width=2))

        main_layout.addWidget(self.raw_plot, stretch=2)
        main_layout.addWidget(self.filtered_plot, stretch=2)
        main_layout.addWidget(self.hr_plot, stretch=1)

        status_bar = QStatusBar(self)
        self.setStatusBar(status_bar)
        self.statusBar().showMessage(
            f"Ready — connect ESP32-S3 streaming {DEFAULT_SAMPLE_RATE_HZ} Hz CSV over USB ({QT_BINDING})"
        )

    def _connect_signals(self) -> None:
        self.refresh_button.clicked.connect(self.refresh_ports)
        self.connect_button.clicked.connect(self.toggle_connection)
        self.start_button.clicked.connect(self.start_capture)
        self.pause_button.clicked.connect(self.pause_capture)
        self.resume_button.clicked.connect(self.resume_capture)
        self.stop_button.clicked.connect(self.stop_capture)
        self.reset_button.clicked.connect(self.reset_view_state)
        self.record_button.clicked.connect(self.select_record_path)
        self.filter_combo.currentTextChanged.connect(self._refresh_filter_controls)
        self.sample_rate_spin.valueChanged.connect(self._handle_sample_rate_changed)
        self.window_seconds_spin.valueChanged.connect(self._set_buffer_capacity)

    def refresh_ports(self) -> None:
        """Refresh available serial ports."""
        current_text = self.port_combo.currentText()
        self.port_combo.clear()
        port_names = [port.device for port in list_ports.comports()]
        self.port_combo.addItems(port_names)
        if current_text and current_text in port_names:
            self.port_combo.setCurrentText(current_text)
        self.statusBar().showMessage(f"Detected {len(port_names)} serial port(s)")

    def toggle_connection(self) -> None:
        """Connect or disconnect the selected serial port."""
        if self.reader_thread is not None:
            self.disconnect_serial()
            return

        port = self.port_combo.currentText().strip()
        if not port:
            QMessageBox.warning(self, "No port selected", "Select a serial port before connecting.")
            return

        baud_rate = int(self.baud_combo.currentText())
        self.reader_thread = SerialReaderThread(port=port, baud_rate=baud_rate)
        self.reader_thread.sample_received.connect(self._queue_sample)
        self.reader_thread.parse_error.connect(self._handle_parse_error)
        self.reader_thread.status_message.connect(self.statusBar().showMessage)
        self.reader_thread.serial_error.connect(self._handle_serial_error)
        self.reader_thread.finished.connect(self._handle_reader_finished)
        self.reader_thread.start()
        self.connected_port = port
        self.connect_button.setText("Disconnect")
        self._refresh_capture_controls()

    def disconnect_serial(self) -> None:
        """Disconnect the active serial port."""
        self.stop_capture()
        if self.reader_thread is not None:
            self.reader_thread.stop()
            self.reader_thread = None
        self.connected_port = None
        self.connect_button.setText("Connect")
        self.statusBar().showMessage("Serial port disconnected")
        self._refresh_capture_controls()

    def _handle_reader_finished(self) -> None:
        if self.reader_thread is not None and not self.reader_thread.isRunning():
            self.reader_thread = None
            self.connected_port = None
            self.connect_button.setText("Connect")
            self._refresh_capture_controls()

    def _queue_sample(self, sample: StreamSample, host_time: float) -> None:
        self.pending_samples.append((sample, host_time))

    def _handle_parse_error(self, message: str) -> None:
        self.diagnostics.note_parse_error()
        self.parse_errors_label.setText(str(self.diagnostics.parse_errors))
        self.statusBar().showMessage(f"Parse warning: {message}")

    def _handle_serial_error(self, message: str) -> None:
        self.statusBar().showMessage(message)
        self.stop_capture()
        if self.reader_thread is not None:
            self.reader_thread.stop()
            self.reader_thread = None
        self.connected_port = None
        self.connect_button.setText("Connect")
        self._refresh_capture_controls()
        QMessageBox.warning(self, "Serial connection error", message)

    def start_capture(self) -> None:
        """Begin a new capture session."""
        if self.reader_thread is None:
            QMessageBox.warning(self, "Not connected", "Connect to the ESP32-S3 before starting capture.")
            return

        self._close_record_file()
        self.pending_samples.clear()
        self.raw_samples.clear()
        self.filtered_samples.clear()
        self.hr_samples.clear()
        self.time_seconds.clear()
        self.diagnostics = StreamDiagnostics(sample_rate_hz=self.sample_rate_spin.value())
        self._set_buffer_capacity()
        self._open_record_file_if_needed()
        self.capture_state = "running"
        self._refresh_capture_controls()
        self._update_metrics_labels()
        self._update_plots()
        self.statusBar().showMessage("Capture started")

    def pause_capture(self) -> None:
        """Pause plot/record accumulation while staying connected."""
        if self.capture_state != "running":
            return
        self.capture_state = "paused"
        self.diagnostics.reset_gap_reference()
        self._refresh_capture_controls()
        self.statusBar().showMessage("Capture paused")

    def resume_capture(self) -> None:
        """Resume capture after a pause."""
        if self.capture_state != "paused":
            return
        self.capture_state = "running"
        self.diagnostics.reset_gap_reference()
        self._refresh_capture_controls()
        self.statusBar().showMessage("Capture resumed")

    def stop_capture(self) -> None:
        """Stop capture and close any active CSV recording."""
        if self.capture_state == "stopped":
            self._close_record_file()
            self._refresh_capture_controls()
            return
        self.capture_state = "stopped"
        self._close_record_file()
        self._refresh_capture_controls()
        self.statusBar().showMessage("Capture stopped")

    def reset_view_state(self) -> None:
        """Clear plots, counters, and transient state."""
        self.stop_capture()
        self.pending_samples.clear()
        self.raw_samples.clear()
        self.filtered_samples.clear()
        self.hr_samples.clear()
        self.time_seconds.clear()
        self.diagnostics = StreamDiagnostics(sample_rate_hz=self.sample_rate_spin.value())
        self._update_metrics_labels()
        self._update_plots()
        self.statusBar().showMessage("View and diagnostics reset")

    def select_record_path(self) -> None:
        """Choose an optional CSV output path for live recording."""
        suggested_name = f"usb_ecg_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path_str, _filter = QFileDialog.getSaveFileName(
            self,
            "Select CSV output",
            suggested_name,
            "CSV files (*.csv);;All files (*)",
        )
        if not path_str:
            return
        self.record_path = Path(path_str)
        self.record_label.setText(str(self.record_path))
        self.statusBar().showMessage(f"CSV output selected: {self.record_path}")

    def _open_record_file_if_needed(self) -> None:
        if self.record_path is None:
            return
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        self.record_file = self.record_path.open("w", encoding="utf-8", newline="")
        self.record_file.write(f"# started_utc={datetime.utcnow().isoformat()}Z\n")
        self.record_file.write(f"# port={self.connected_port or 'unknown'}\n")
        self.record_file.write(f"# baud={self.baud_combo.currentText()}\n")
        self.record_file.write(f"# configured_fs_hz={self.sample_rate_spin.value()}\n")
        self.record_file.write("timestamp_ms,sample_index,raw_ecg\n")
        self.record_file.flush()

    def _close_record_file(self) -> None:
        if self.record_file is not None:
            try:
                self.record_file.flush()
                self.record_file.close()
            except Exception:
                pass
        self.record_file = None

    def _process_pending_samples(self) -> None:
        if not self.pending_samples:
            return

        batch_samples: list[StreamSample] = []
        batch_host_times: list[float] = []
        while self.pending_samples:
            sample, host_time = self.pending_samples.popleft()
            batch_samples.append(sample)
            batch_host_times.append(host_time)

        if self.capture_state != "running":
            return

        for sample, host_time in zip(batch_samples, batch_host_times):
            self.diagnostics.observe_sample(sample, host_time)
            self.time_seconds.append(sample.timestamp_ms / 1000.0)
            self.raw_samples.append(float(sample.raw_ecg))
            if self.record_file is not None:
                self.record_file.write(f"{sample.timestamp_ms},{sample.sample_index},{sample.raw_ecg}\n")

        if self.record_file is not None:
            self.record_file.flush()

        self._recompute_filtered_views()
        self._update_metrics_labels()
        self._update_plots()

    def _recompute_filtered_views(self) -> None:
        raw_array = np.asarray(self.raw_samples, dtype=float)
        if raw_array.size == 0:
            self.filtered_samples.clear()
            self.hr_samples.clear()
            return

        filtered = raw_array.copy()
        try:
            if self.filter_combo.currentText() == FILTER_BUTTERWORTH and raw_array.size >= 16:
                sample_rate_hz = self.sample_rate_spin.value()
                low_cut_hz, high_cut_hz = validate_filter_settings(
                    low_cut_hz=float(self.low_cut_spin.value()),
                    high_cut_hz=float(self.high_cut_spin.value()),
                    sampling_rate=sample_rate_hz,
                )
                filtered = apply_butterworth_bandpass(
                    raw_array,
                    sampling_rate=sample_rate_hz,
                    low_cut_hz=low_cut_hz,
                    high_cut_hz=high_cut_hz,
                )
                if self.notch_checkbox.isChecked():
                    filtered = apply_powerline_notch(
                        filtered,
                        sampling_rate=sample_rate_hz,
                        notch_frequency_hz=validate_powerline_frequency(50.0, sample_rate_hz),
                    )
        except Exception as exc:
            self.statusBar().showMessage(f"Filter warning: {exc}")
            filtered = raw_array.copy()

        hr_trace = np.full(filtered.size, np.nan, dtype=float)
        if filtered.size >= self.sample_rate_spin.value():
            try:
                hr_trace = estimate_live_heart_rate_trace(filtered, self.sample_rate_spin.value())
            except Exception as exc:
                self.statusBar().showMessage(f"HR warning: {exc}")

        self.filtered_samples = deque((float(value) for value in filtered), maxlen=self._buffer_capacity())
        self.hr_samples = deque((float(value) for value in hr_trace), maxlen=self._buffer_capacity())

    def _update_plots(self) -> None:
        times = np.asarray(self.time_seconds, dtype=float)
        raw = np.asarray(self.raw_samples, dtype=float)
        filtered = np.asarray(self.filtered_samples, dtype=float)
        hr = np.asarray(self.hr_samples, dtype=float)

        self.raw_curve.setData(times, raw)
        self.filtered_curve.setData(times[: filtered.size], filtered)
        self.hr_curve.setData(times[: hr.size], hr)

    def _update_metrics_labels(self) -> None:
        self.live_rate_label.setText(f"{self.diagnostics.incoming_rate_hz:.1f} Hz")
        self.total_samples_label.setText(str(self.diagnostics.total_samples))
        self.dropped_samples_label.setText(str(self.diagnostics.dropped_samples))
        self.parse_errors_label.setText(str(self.diagnostics.parse_errors))
        interval_stats = self.diagnostics.interval_stats
        if interval_stats is None:
            self.interval_stats_label.setText("n/a")
        else:
            self.interval_stats_label.setText(
                f"mean {interval_stats.mean_ms:.2f} ms | std {interval_stats.std_ms:.2f} | "
                f"min {interval_stats.min_ms:.2f} | max {interval_stats.max_ms:.2f}"
            )

    def _handle_sample_rate_changed(self) -> None:
        self._set_buffer_capacity()
        self._refresh_filter_controls()
        self.reset_view_state()

    def _refresh_filter_controls(self) -> None:
        butterworth_enabled = self.filter_combo.currentText() == FILTER_BUTTERWORTH
        self.low_cut_spin.setEnabled(butterworth_enabled)
        self.high_cut_spin.setEnabled(butterworth_enabled)
        self.notch_checkbox.setEnabled(butterworth_enabled)
        max_high_cut = max(5, (self.sample_rate_spin.value() // 2) - 1)
        self.high_cut_spin.setMaximum(max_high_cut)
        if self.high_cut_spin.value() > max_high_cut:
            self.high_cut_spin.setValue(max_high_cut)

    def _buffer_capacity(self) -> int:
        return self.sample_rate_spin.value() * self.window_seconds_spin.value()

    def _set_buffer_capacity(self) -> None:
        capacity = self._buffer_capacity()
        self.raw_samples = deque(list(self.raw_samples)[-capacity:], maxlen=capacity)
        self.filtered_samples = deque(list(self.filtered_samples)[-capacity:], maxlen=capacity)
        self.hr_samples = deque(list(self.hr_samples)[-capacity:], maxlen=capacity)
        self.time_seconds = deque(list(self.time_seconds)[-capacity:], maxlen=capacity)

    def _refresh_capture_controls(self) -> None:
        is_connected = self.reader_thread is not None
        is_running = self.capture_state == "running"
        is_paused = self.capture_state == "paused"

        self.start_button.setEnabled(is_connected and not is_running and not is_paused)
        self.pause_button.setEnabled(is_running)
        self.resume_button.setEnabled(is_paused)
        self.stop_button.setEnabled(is_running or is_paused)
        self.reset_button.setEnabled(True)
        self.record_button.setEnabled(not is_running)
        self.port_combo.setEnabled(not is_connected)
        self.baud_combo.setEnabled(not is_connected)
        self.refresh_button.setEnabled(not is_connected)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.disconnect_serial()
        super().closeEvent(event)


def main() -> int:
    """Launch the standalone tester."""
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    window = USBECGStreamTesterWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
