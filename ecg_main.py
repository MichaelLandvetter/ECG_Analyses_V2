"""Entry point for the ECG analysis scaffold application."""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from ecg_acquisition import acquire_ecg_signal
from ecg_analysis import NeuroKit2UnavailableError, analyze_ecg
from ecg_config import DEFAULT_OUTPUT_DIR, DEFAULT_SAMPLING_RATE, DEFAULT_SYNTHETIC_DURATION_SECONDS
from ecg_report import save_reports, save_structured_reports


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run ECG analysis using NeuroKit2.")
    parser.add_argument("--input-file", type=str, default=None, help="Path to ECG data file (.npy/.csv/.txt).")
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=DEFAULT_SAMPLING_RATE,
        help=f"Sampling rate in Hz (default: {DEFAULT_SAMPLING_RATE}).",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=DEFAULT_SYNTHETIC_DURATION_SECONDS,
        help=(
            "Synthetic ECG duration in seconds when --input-file is not provided "
            f"(default: {DEFAULT_SYNTHETIC_DURATION_SECONDS})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for report outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable JSON output and write only text report.",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Force CLI pipeline mode instead of launching the desktop UI.",
    )
    return parser


def run(input_file: Optional[str], sampling_rate: int, duration_seconds: int, output_dir: str, no_json: bool) -> int:
    """Execute the ECG workflow."""
    signal, source = acquire_ecg_signal(
        input_file=input_file,
        sampling_rate=sampling_rate,
        duration_seconds=duration_seconds,
    )
    results = analyze_ecg(signal=signal, sampling_rate=sampling_rate)

    if input_file:
        report_paths = save_structured_reports(
            analysis_results=results,
            input_file=input_file,
            source=source,
            write_json=not no_json,
        )
    else:
        report_paths = save_reports(
            analysis_results=results,
            output_directory=output_dir,
            source=source,
            write_json=not no_json,
        )

    print("ECG analysis completed successfully.")
    print("Saved outputs:")
    for key, path in report_paths.items():
        print(f"- {key}: {path.name}")

    return 0


def main() -> int:
    """Program entry point."""
    parser = build_parser()
    args = parser.parse_args()
    should_run_cli = (
        args.cli
        or args.input_file is not None
        or args.sampling_rate != DEFAULT_SAMPLING_RATE
        or args.duration_seconds != DEFAULT_SYNTHETIC_DURATION_SECONDS
        or args.output_dir != DEFAULT_OUTPUT_DIR
        or args.no_json
    )

    if not should_run_cli:
        try:
            from ecg_ui import launch_ecg_ui

            return launch_ecg_ui()
        except Exception as exc:
            print(f"UI startup error: {exc}", file=sys.stderr)
            return 1

    try:
        return run(
            input_file=args.input_file,
            sampling_rate=args.sampling_rate,
            duration_seconds=args.duration_seconds,
            output_dir=args.output_dir,
            no_json=args.no_json,
        )
    except NeuroKit2UnavailableError as exc:
        print(f"Dependency error: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Processing error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
