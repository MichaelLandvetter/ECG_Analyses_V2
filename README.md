# ECG_Analyses_V2

ZIP-download-friendly Python ECG scaffold with a desktop UI flow, modular analysis wrappers, and report generation.

## Python recommendation

- Python **3.10+** (3.11 preferred)

## Quick start (Thonny / PyCharm friendly)

1. Download the repository ZIP and extract it.
2. Open the extracted folder in Thonny or PyCharm.
3. Install dependencies in your interpreter/terminal:

```bash
pip install neurokit2 numpy scipy pandas matplotlib
```

4. Launch the desktop app from the project folder:

```bash
python ecg_main.py
```

The desktop UI provides top-level controls for input source, file selection, filter settings, R-peak method selection, and run controls.

## Desktop UI behavior

- **Input source**: `File Replay` (default) or `USB Input` (placeholder, non-crashing message).
- **ECG Data File**: picker for `.txt`, `.csv`, `.npy` data files (enabled in File Replay mode).
- **Filter settings**: Butterworth bandpass default with low/high cut fields and an `Apply Filter` action.
- **R-peak detection methods**: `neurokit`, `pantompkins1985`, `engzeemod2012`, `hamilton2002` (default).
- **Controls**: `Start`, `Stop`, `Reset`, `Open Pre-report`.

Tabs include:
- `Analysis View` (Raw/Filtered ECG plot and Heart Rate/R-peak tachometer plot)
- `ECG Processing Settings` (reserved placeholder for future options)

## CLI usage

If you want the original command-line pipeline behavior, run:

```bash
python ecg_main.py --cli --input-file path/to/ecg.csv --sampling-rate 250 --output-dir output
```

Optional flags:

- `--duration-seconds 15` (controls synthetic duration when no input file is provided)
- `--no-json` (writes text report only)

## Project structure

- `ecg_main.py` - entry point and workflow orchestration
- `ecg_acquisition.py` - ECG file loading and synthetic signal generation
- `ecg_analysis.py` - NeuroKit2-based ECG processing, filtering helpers, and metric extraction
- `ecg_report.py` - text/JSON/pre-report generation utilities
- `ecg_config.py` - centralized defaults and constants
- `ecg_ui.py` - Tkinter desktop UI and Matplotlib embedding
- `tests/` - lightweight starter tests

## Outputs

Reports are written to `output/` by default:

- `ecg_summary_report.txt`
- `ecg_analysis_results.json` (unless `--no-json` is used)

## Future PyInstaller packaging direction

This scaffold keeps module boundaries simple and local-file-friendly (`python ecg_main.py`) to support future bundling as a desktop executable with PyInstaller (e.g., `pyinstaller --onefile ecg_main.py`).

## Optional dependency notes

- `tkinter` ships with standard Python installers on most desktop systems and powers the UI.
- `matplotlib` is required for embedded plotting; if missing, the UI still opens with a clear message instead of crashing.
- `neurokit2` is required for full ECG analysis and R-peak detection; missing dependency errors are shown clearly in UI/CLI.
