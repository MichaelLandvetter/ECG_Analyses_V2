# ECG_Analyses_V2

Clean, ZIP-download-friendly Python ECG analysis app focused on **offline file analysis** and USB acquisition with structured CSV report export.

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

## File Analysis workflow

1. Launch the app and keep **File Analysis** selected in Box 1.
2. Choose an ECG data file (`.txt`, `.csv`, or `.npy`).
3. In **ECG Filter Settings**, select either:
   - `NeuroKit2 ecg_clean`
   - `Butterworth bandpass`
4. Optionally adjust Butterworth low/high cut values and click **Apply Filter** to refresh the preview.
5. In **R-peak Detection**, choose the detector method (`hamilton2002` default).
6. Click **Start** to run the offline analysis. A processing dialog is shown while the full analysis is computed.
7. Results are rendered directly in the main **Analysis View** with:
   - full ECG raw + filtered + R-peaks
   - full-duration heart-rate plot
   - beat snippets with average template overlay plus per-beat P/Q/S/T markers from NeuroKit2 delineation
8. Use **Re-run Analysis** after the first run when processing settings are changed.
9. Click **Save Reports** to export CSV report files.
10. Optionally click **Open Detailed Review** to open the expanded review window.

## USB Input workflow

1. Select **USB Input** in Box 1.
2. Click **Start** to begin acquisition.
3. Click **Pause** and **Resume** as needed during the same acquisition session.
4. Click **End** to finalize the session and save the raw USB capture file.
5. The app then runs the same offline analysis pipeline used by File Analysis and opens the detailed review view.
6. Use **Save Reports** to export reports from the completed USB session analysis.

## ECG Processing Settings tab

The **ECG Processing Settings** tab stores runtime processing choices for later reuse:

- R-peak detector method
- Sampling rate for offline analysis
- Power-line notch frequency
- Rolling window settings used for USB preview buffering

Built-in defaults live in `ecg_config.py` in the project root.

Saved runtime settings are mirrored to:

- `ecg_settings.json` in the project root

On startup the app loads `ecg_settings.json` if it is present and valid. If the file is missing or corrupt, the app falls back to the defaults defined in `ecg_config.py`.

## Report output location policy

When **Save Reports** is used after File Analysis or USB End analysis:

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

## Current scope

- **Implemented now:** File Analysis in the main Analysis View, USB acquisition session controls, shared NeuroKit2 analysis pipeline, settings save/load, CSV export.

## Project structure

- `ecg_main.py` - app entry point and CLI/UI routing
- `ecg_acquisition.py` - offline ECG file loading and synthetic fallback generation
- `ecg_analysis.py` - NeuroKit2-centered filtering, R-peak detection, analysis outputs
- `ecg_report.py` - structured CSV export and report folder creation
- `ecg_config.py` - defaults and JSON-backed processing settings helpers
- `ecg_ui.py` - PyQt6 desktop workflow, mode-specific controls, and detailed review window
- `tests/` - lightweight unit tests for core contracts

## Tests

Run the lightweight test suite with:

```bash
python -m unittest discover -s tests -v
```
