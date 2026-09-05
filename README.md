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
7. Results are rendered directly in the main **Analysis View** with a 3-panel layout:
   - full ECG raw + filtered + R-peaks
   - full-duration heart-rate plot
   - beat snippets with average template overlay plus per-beat P/Q/S/T markers from NeuroKit2 delineation
8. Use **Re-run Analysis** after the first run when processing settings are changed.
9. Click **Save Reports** to export CSV report files.
10. Optionally click **Open Pre-report** to open the expanded review window.

## USB Input workflow

1. Select **USB Input** in Box 1.
2. Click **Start** to begin acquisition (simulated stream with production state transitions).
3. Click **Pause** and **Resume** as needed during the same acquisition session.
4. Click **End** to finalize the session and save the raw USB capture file.
5. During capture, the Analysis View switches to a 2-panel live layout (raw/filtered preview + live HR trend) and hides beat snippets.
6. On **End**, the app saves the raw USB file, runs offline full analysis from the saved file, then opens the pre-report review window automatically.
7. Use **Save Reports** to export reports from the completed USB session analysis.

## USB Stream Tester

This repository also includes a standalone pre-integration USB ECG stream tester for an ESP32-S3 board streaming raw analog ECG over USB serial.

### Wiring assumptions

- SparkFun analog ECG board analog output -> **GPIO2 / ADC1** on the Raspberry Baguette S3 (ESP32-S3)
- ECG board ground -> ESP32-S3 ground
- ECG board power -> board-appropriate supply rail

### Firmware upload

1. Open `/home/runner/work/ECG_Analyses_V2/ECG_Analyses_V2/firmware/esp32s3_ecg_stream/esp32s3_ecg_stream.ino` in the Arduino IDE.
2. Select your ESP32-S3 board and USB CDC-capable port.
3. Confirm the sketch constants near the top:
   - sample rate: `250`
   - ADC pin: `GPIO2`
   - baud rate: `230400`
4. Upload the sketch.
5. Open a serial monitor or the tester below and confirm lines appear in this format:

   ```text
   timestamp_ms,sample_index,raw_ecg
   1234,567,2048
   ```

### Run the standalone tester

1. Install the tester dependencies:

   ```bash
   pip install -r requirements_usb_tester.txt
   ```

2. Launch the tester from the repository root:

   ```bash
   python tools/usb_ecg_stream_tester.py
   ```

3. In the tester UI:
   - refresh and select the correct COM/serial port
   - choose the baud rate used by the firmware
   - click **Connect**
   - optionally select a CSV output path
   - click **Start** to begin live capture
   - use **Pause**, **Resume**, **Stop**, and **Reset view/state** as needed
4. Use the filter controls to view raw ECG, Butterworth bandpass output, and optional 50 Hz notch filtering.
5. Review the telemetry panel for incoming sample rate, dropped-sample estimate, timestamp interval stats, and parse errors.

The standalone recorder writes CSV captures in the same streaming format:

```text
timestamp_ms,sample_index,raw_ecg
```

Optional comment-prefixed metadata header lines are included before the CSV header when recording is enabled.

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

1. A `Reports/` folder is created next to the analysis input file if it does not already exist.
   - File Analysis input file: user-selected ECG file
   - USB session input file: generated capture under `<app_working_directory>/USB_Sessions/usb_session_<YYYYMMDD_HHMMSS_microseconds>.csv`
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

- **Implemented now:** File Analysis (3-panel review), USB live capture controls (Start/Pause/Resume/End), automatic USB end-to-offline-analysis pre-report flow, settings save/load, CSV export.

## Project structure

- `ecg_main.py` - app entry point and CLI/UI routing
- `ecg_acquisition.py` - offline ECG file loading and synthetic fallback generation
- `ecg_analysis.py` - NeuroKit2-centered filtering, R-peak detection, analysis outputs
- `ecg_report.py` - structured CSV export and report folder creation
- `ecg_config.py` - defaults and JSON-backed processing settings helpers
- `ecg_ui.py` - PyQt6 desktop workflow, mode-specific controls/layouts, and pre-report review window
- `tests/` - lightweight unit tests for core contracts

## Tests

Run the lightweight test suite with:

```bash
python -m unittest discover -s tests -v
```
