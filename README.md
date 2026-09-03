# ECG_Analyses_V2

Clean, ZIP-download-friendly Python ECG analysis app focused on **offline data-file replay** with a desktop review flow and structured CSV report export.

## Python recommendation

- Python **3.10+** (3.11 preferred)

## Dependencies

Install the project dependencies in Thonny, PyCharm, or a terminal:

```bash
pip install neurokit2 numpy scipy pandas PyQt6 pyqtgraph
```

## Run the app

Desktop UI:

```bash
python ecg_main.py
```

CLI pipeline:

```bash
python ecg_main.py --cli --input-file path/to/ecg.csv --sampling-rate 250
```

## Offline file-analysis workflow

1. Launch the app and keep **File Replay** selected in Box 1.
2. Choose an ECG data file (`.txt`, `.csv`, or `.npy`).
3. In **ECG Filter Settings**, select either:
   - `NeuroKit2 ecg_clean`
   - `Butterworth bandpass`
4. Optionally adjust Butterworth low/high cut values and click **Apply Filter** to refresh the preview.
5. In **R-peak Detection**, choose the detector method (`hamilton2002` default).
6. Click **Start** to run the offline analysis. A processing dialog is shown while the full analysis is computed.
7. When processing finishes, a **pre-report review window** opens with:
   - full ECG raw + filtered + R-peaks
   - full-duration heart-rate plot
   - beat snippets with average template overlay plus P/Q/S/T markers from NeuroKit2 delineation
8. Use PyQtGraph pan/zoom in each review plot, and double-click any plot to open an enlarged navigable view.
9. Click **Save Reports** to export CSV report files, or **Back to Analysis** to return without exporting.

## ECG Processing Settings tab

The **ECG Processing Settings** tab stores runtime processing choices for later reuse:

- R-peak detector method
- Sampling rate for offline analysis
- Power-line notch frequency
- Rolling window placeholder settings for future USB streaming

Built-in defaults live in `ecg_config.py` in the project root.

Saved runtime settings are mirrored to:

- `ecg_settings.json` in the project root

On startup the app loads `ecg_settings.json` if it is present and valid. If the file is missing or corrupt, the app falls back to the defaults defined in `ecg_config.py`.

## Report output location policy

When **Save Reports** is used after file analysis:

1. A `Reports/` folder is created next to the selected input file if it does not already exist.
2. A per-run folder is created inside `Reports/` using:

   ```text
   <input_base_name>_<YYYYMMDD_HHMM>
   ```

3. The following CSV files are written into that run folder:

- `summary_metrics.csv`
- `average_template_wave.csv`
- `beat_morphology_landmarks.csv`
- `continuous_time_series.csv`

A text summary report is also written in the same folder.

## Current scope and deferred work

- **Implemented now:** offline file replay analysis, review window, settings save/load, CSV export.
- **Deferred:** USB streaming. The UI keeps a clear placeholder message and does not crash when `USB Input` is selected.

## Project structure

- `ecg_main.py` - app entry point and CLI/UI routing
- `ecg_acquisition.py` - offline ECG file loading and synthetic fallback generation
- `ecg_analysis.py` - NeuroKit2-centered filtering, R-peak detection, analysis outputs
- `ecg_report.py` - structured CSV export and report folder creation
- `ecg_config.py` - defaults and JSON-backed processing settings helpers
- `ecg_ui.py` - PyQt6 desktop workflow and pre-report review window
- `tests/` - lightweight unit tests for core contracts

## Tests

Run the lightweight test suite with:

```bash
python -m unittest discover -s tests -v
```
