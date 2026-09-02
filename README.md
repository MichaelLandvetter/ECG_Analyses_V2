# ECG_Analyses_V2

ZIP-download-friendly Python scaffold for ECG acquisition, NeuroKit2 analysis, and report generation.

## Python recommendation

- Python **3.10+** (3.11 preferred)

## Quick start (Thonny / PyCharm friendly)

1. Download the repository ZIP and extract it.
2. Open the extracted folder in Thonny or PyCharm.
3. Install dependencies in your interpreter/terminal:

```bash
pip install neurokit2 numpy scipy pandas matplotlib
```

4. Run the pipeline from the project folder:

```bash
python ecg_main.py
```

This runs a synthetic ECG path when no input file is supplied.

## CLI usage

```bash
python ecg_main.py --input-file path/to/ecg.csv --sampling-rate 250 --output-dir output
```

Optional flags:

- `--duration-seconds 15` (controls synthetic duration when no input file is provided)
- `--no-json` (writes text report only)

## Project structure

- `ecg_main.py` - entry point and workflow orchestration
- `ecg_acquisition.py` - ECG file loading and synthetic signal generation
- `ecg_analysis.py` - NeuroKit2-based ECG processing and metric extraction
- `ecg_report.py` - text/JSON report generation utilities
- `ecg_config.py` - centralized defaults and constants
- `tests/` - lightweight starter tests

## Outputs

Reports are written to `output/` by default:

- `ecg_summary_report.txt`
- `ecg_analysis_results.json` (unless `--no-json` is used)

## Future PyInstaller packaging direction

This scaffold keeps module boundaries simple and local-file-friendly (`python ecg_main.py`) to support future bundling as a desktop executable with PyInstaller (e.g., `pyinstaller --onefile ecg_main.py`).
