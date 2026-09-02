"""Tkinter desktop UI for ECG acquisition and analysis flow."""

from __future__ import annotations

import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
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

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except Exception:
    Figure = Any  # type: ignore[assignment]
    FigureCanvasTkAgg = Any  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False


class ECGDesktopApp(ttk.Frame):
    """Desktop UI for running the clean ECG analysis skeleton."""

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=10)
        self.master.title("ECG Analysis")
        self.master.geometry("1300x820")
        self.pack(fill=tk.BOTH, expand=True)

        self.source_var = tk.StringVar(value=DEFAULT_INPUT_SOURCE)
        self.file_status_var = tk.StringVar(value="No file selected")
        self.filter_mode_var = tk.StringVar(value=DEFAULT_FILTER_MODE)
        self.low_cut_var = tk.StringVar(value=str(DEFAULT_FILTER_LOW_CUT_HZ))
        self.high_cut_var = tk.StringVar(value=str(DEFAULT_FILTER_HIGH_CUT_HZ))
        self.rpeak_method_var = tk.StringVar(value=DEFAULT_RPEAK_METHOD)
        self.status_var = tk.StringVar(value="Ready")

        self.selected_file: str | None = None
        self.acquisition_running = False
        self.filter_armed = False
        self.latest_results: dict[str, Any] | None = None
        self.latest_source = ""

        self._build_layout()
        self._set_initial_plots()
        self._on_source_mode_changed()

    def _build_layout(self) -> None:
        top_controls = ttk.Frame(self)
        top_controls.grid(row=0, column=0, sticky="ew")
        for column in range(5):
            top_controls.columnconfigure(column, weight=1)

        self._build_source_controls(top_controls)
        self._build_file_controls(top_controls)
        self._build_filter_controls(top_controls)
        self._build_rpeak_controls(top_controls)
        self._build_action_controls(top_controls)

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        analysis_view = ttk.Frame(self.notebook, padding=8)
        processing_settings = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(analysis_view, text="Analysis View")
        self.notebook.add(processing_settings, text="ECG Processing Settings")

        self._build_analysis_tab(analysis_view)
        self._build_settings_tab(processing_settings)

        status_label = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status_label.grid(row=2, column=0, sticky="ew", pady=(8, 0))

    def _build_source_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="File or USB Input", padding=8)
        frame.grid(row=0, column=0, sticky="nsew", padx=4)

        source_selector = ttk.Combobox(
            frame,
            textvariable=self.source_var,
            values=("File Replay", "USB Input"),
            state="readonly",
        )
        source_selector.grid(row=0, column=0, sticky="ew")
        source_selector.bind("<<ComboboxSelected>>", lambda _event: self._on_source_mode_changed())
        frame.columnconfigure(0, weight=1)

    def _build_file_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="ECG Data File", padding=8)
        frame.grid(row=0, column=1, sticky="nsew", padx=4)

        ttk.Label(frame, textvariable=self.file_status_var, wraplength=220).grid(row=0, column=0, sticky="w")
        self.file_button = ttk.Button(frame, text="Choose ECG File", command=self._select_ecg_file)
        self.file_button.grid(row=1, column=0, sticky="w", pady=(6, 0))
        frame.columnconfigure(0, weight=1)

    def _build_filter_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="ECG Filter Settings", padding=8)
        frame.grid(row=0, column=2, sticky="nsew", padx=4)

        mode_selector = ttk.Combobox(
            frame,
            textvariable=self.filter_mode_var,
            values=("Butterworth bandpass",),
            state="readonly",
        )
        mode_selector.grid(row=0, column=0, columnspan=2, sticky="ew")

        ttk.Label(frame, text="Low cut (Hz):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(frame, textvariable=self.low_cut_var, width=9).grid(row=1, column=1, sticky="ew", pady=(6, 0))

        ttk.Label(frame, text="High cut (Hz):").grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(frame, textvariable=self.high_cut_var, width=9).grid(row=2, column=1, sticky="ew", pady=(4, 0))

        ttk.Button(frame, text="Apply Filter", command=self._arm_filter).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        frame.columnconfigure(1, weight=1)

    def _build_rpeak_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="R-peak Detection", padding=8)
        frame.grid(row=0, column=3, sticky="nsew", padx=4)

        method_selector = ttk.Combobox(
            frame,
            textvariable=self.rpeak_method_var,
            values=("neurokit", "pantompkins1985", "engzeemod2012", "hamilton2002"),
            state="readonly",
        )
        method_selector.grid(row=0, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)

    def _build_action_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Controls", padding=8)
        frame.grid(row=0, column=4, sticky="nsew", padx=4)

        ttk.Button(frame, text="Start", command=self._start_analysis).grid(row=0, column=0, sticky="ew")
        ttk.Button(frame, text="Stop", command=self._stop_analysis).grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(frame, text="Reset", command=self._reset_ui).grid(row=2, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(frame, text="Open Pre-report", command=self._open_pre_report).grid(row=3, column=0, sticky="ew", pady=(4, 0))
        frame.columnconfigure(0, weight=1)

    def _build_analysis_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        if not MATPLOTLIB_AVAILABLE:
            ttk.Label(
                parent,
                text="Matplotlib is not installed. Install matplotlib to render ECG plots.",
                wraplength=900,
            ).grid(row=0, column=0, sticky="nw")
            self.figure = None
            self.canvas = None
            self.axes_ecg = None
            self.axes_hr = None
            return

        self.figure = Figure(figsize=(11, 6), dpi=100)
        self.axes_ecg = self.figure.add_subplot(211)
        self.axes_hr = self.figure.add_subplot(212)
        self.figure.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_settings_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        message = (
            "This tab is reserved for additional acquisition and processing options. "
            "Current controls are available in the top section for a clean, incremental build."
        )
        ttk.Label(parent, text=message, wraplength=900, justify="left").grid(row=0, column=0, sticky="nw")

    def _set_initial_plots(self) -> None:
        if not MATPLOTLIB_AVAILABLE or self.axes_ecg is None or self.axes_hr is None:
            return
        self.axes_ecg.clear()
        self.axes_ecg.set_title("Raw ECG + Filtered ECG")
        self.axes_ecg.set_ylabel("Amplitude")
        self.axes_ecg.grid(True, alpha=0.3)
        self.axes_ecg.text(0.5, 0.5, "No signal loaded", transform=self.axes_ecg.transAxes, ha="center", va="center")

        self.axes_hr.clear()
        self.axes_hr.set_title("Heart Rate — R-peak tachometer")
        self.axes_hr.set_xlabel("Sample")
        self.axes_hr.set_ylabel("BPM")
        self.axes_hr.grid(True, alpha=0.3)
        self.axes_hr.text(0.5, 0.5, "No analysis yet", transform=self.axes_hr.transAxes, ha="center", va="center")
        self.canvas.draw()

    def _on_source_mode_changed(self) -> None:
        is_file_mode = self.source_var.get() == "File Replay"
        state = tk.NORMAL if is_file_mode else tk.DISABLED
        self.file_button.configure(state=state)
        if not is_file_mode:
            self.file_status_var.set("USB Input selected (placeholder)")
        elif self.selected_file:
            self.file_status_var.set(Path(self.selected_file).name)
        else:
            self.file_status_var.set("No file selected")

    def _select_ecg_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select ECG data file",
            filetypes=[("ECG files", "*.txt *.csv *.npy"), ("All files", "*.*")],
        )
        if not file_path:
            return
        self.selected_file = file_path
        self.file_status_var.set(Path(file_path).name)
        self.status_var.set(f"Selected file: {file_path}")

    def _arm_filter(self) -> None:
        try:
            low_cut_hz = float(self.low_cut_var.get())
            high_cut_hz = float(self.high_cut_var.get())
            validate_filter_settings(
                low_cut_hz=low_cut_hz,
                high_cut_hz=high_cut_hz,
                sampling_rate=DEFAULT_SAMPLING_RATE,
            )
        except ValueError as exc:
            messagebox.showerror("Invalid filter settings", str(exc))
            self.filter_armed = False
            return

        self.filter_armed = True
        self.status_var.set("Filter settings armed and will be applied on Start")

    def _start_analysis(self) -> None:
        source_mode = self.source_var.get()
        if source_mode == "USB Input":
            messagebox.showinfo("USB Input", "USB Input is not implemented yet.")
            self.status_var.set("USB Input placeholder selected")
            return

        if not self.selected_file:
            messagebox.showerror("Missing ECG file", "Select an ECG data file before starting analysis.")
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
                low_cut_hz = float(self.low_cut_var.get())
                high_cut_hz = float(self.high_cut_var.get())
                analysis_signal = apply_butterworth_bandpass(
                    signal=raw_signal,
                    sampling_rate=DEFAULT_SAMPLING_RATE,
                    low_cut_hz=low_cut_hz,
                    high_cut_hz=high_cut_hz,
                )

            results = analyze_ecg(
                signal=analysis_signal,
                sampling_rate=DEFAULT_SAMPLING_RATE,
                rpeak_method=self.rpeak_method_var.get(),
            )

            self.latest_results = results
            self.latest_source = source
            self._update_plots(raw_signal=raw_signal, filtered_signal=analysis_signal, results=results)

            metrics = results.get("metrics", {})
            self.status_var.set(
                "Analysis completed: "
                f"R peaks={metrics.get('r_peak_count')}, "
                f"Mean HR={metrics.get('mean_heart_rate_bpm')} bpm"
            )
        except NeuroKit2UnavailableError as exc:
            messagebox.showerror("Missing dependency", str(exc))
            self.status_var.set("Analysis failed: NeuroKit2 is required")
        except Exception as exc:
            messagebox.showerror("Analysis error", str(exc))
            self.status_var.set(f"Analysis failed: {exc}")
        finally:
            self.acquisition_running = False

    def _stop_analysis(self) -> None:
        if not self.acquisition_running:
            self.status_var.set("Stop requested: no active acquisition loop")
            return
        self.acquisition_running = False
        self.status_var.set("Acquisition loop stopped")

    def _reset_ui(self) -> None:
        self.acquisition_running = False
        self.filter_armed = False
        self.latest_results = None
        self.latest_source = ""
        self.selected_file = None
        self.source_var.set(DEFAULT_INPUT_SOURCE)
        self.low_cut_var.set(str(DEFAULT_FILTER_LOW_CUT_HZ))
        self.high_cut_var.set(str(DEFAULT_FILTER_HIGH_CUT_HZ))
        self.rpeak_method_var.set(DEFAULT_RPEAK_METHOD)
        self._on_source_mode_changed()
        self._set_initial_plots()
        self.status_var.set("State reset")

    def _open_pre_report(self) -> None:
        if not self.latest_results:
            messagebox.showinfo("Pre-report", "Run Start first to generate a pre-report.")
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
        self.status_var.set(f"Pre-report saved to {report_path}")

    def _update_plots(self, raw_signal: np.ndarray, filtered_signal: np.ndarray, results: dict[str, Any]) -> None:
        if not MATPLOTLIB_AVAILABLE or self.axes_ecg is None or self.axes_hr is None:
            return

        self.axes_ecg.clear()
        self.axes_ecg.plot(raw_signal, label="Raw ECG", alpha=0.7)
        self.axes_ecg.plot(filtered_signal, label="Filtered ECG", alpha=0.8)
        self.axes_ecg.set_title("Raw ECG + Filtered ECG")
        self.axes_ecg.set_ylabel("Amplitude")
        self.axes_ecg.grid(True, alpha=0.3)
        self.axes_ecg.legend(loc="upper right")

        hr_trace = np.asarray(results.get("artifacts", {}).get("heart_rate_trace_bpm", []), dtype=float)
        r_peaks = np.asarray(results.get("artifacts", {}).get("r_peaks", []), dtype=int)

        self.axes_hr.clear()
        if hr_trace.size:
            self.axes_hr.plot(hr_trace, label="Heart Rate (BPM)")
        if hr_trace.size and r_peaks.size:
            valid_peaks = r_peaks[r_peaks < hr_trace.size]
            self.axes_hr.scatter(valid_peaks, hr_trace[valid_peaks], c="red", s=12, label="R peaks")
        self.axes_hr.set_title("Heart Rate — R-peak tachometer")
        self.axes_hr.set_xlabel("Sample")
        self.axes_hr.set_ylabel("BPM")
        self.axes_hr.grid(True, alpha=0.3)
        if hr_trace.size:
            self.axes_hr.legend(loc="upper right")

        self.canvas.draw()


def launch_ecg_ui() -> int:
    """Launch the desktop ECG UI."""
    root = tk.Tk()
    ECGDesktopApp(root)
    root.mainloop()
    return 0
