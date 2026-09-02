"""Input loading and synthetic ECG generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ecg_config import (
    DEFAULT_INPUT_COLUMN_INDEX,
    DEFAULT_SYNTHETIC_DURATION_SECONDS,
    SYNTHETIC_HEART_RATE_BPM,
    SYNTHETIC_RANDOM_SEED,
)


def generate_synthetic_ecg(duration_seconds: int, sampling_rate: int) -> np.ndarray:
    """Generate a synthetic ECG signal.

    Uses NeuroKit2's simulator when available; otherwise falls back to a simple
    deterministic waveform approximation.
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than 0")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be greater than 0")

    sample_count = int(duration_seconds * sampling_rate)

    try:
        import neurokit2 as nk

        return np.asarray(
            nk.ecg_simulate(
                duration=duration_seconds,
                sampling_rate=sampling_rate,
                heart_rate=SYNTHETIC_HEART_RATE_BPM,
                random_state=SYNTHETIC_RANDOM_SEED,
            ),
            dtype=float,
        )
    except Exception:
        time_axis = np.linspace(0, duration_seconds, sample_count, endpoint=False)
        heart_rate_hz = SYNTHETIC_HEART_RATE_BPM / 60.0
        base_wave = np.sin(2 * np.pi * heart_rate_hz * time_axis)
        qrs_like = 0.25 * np.sin(2 * np.pi * 3 * heart_rate_hz * time_axis)
        rng = np.random.default_rng(SYNTHETIC_RANDOM_SEED)
        noise = 0.03 * rng.standard_normal(sample_count)
        return base_wave + qrs_like + noise


def load_ecg_signal_from_file(
    input_file: str,
    column_index: int = DEFAULT_INPUT_COLUMN_INDEX,
) -> np.ndarray:
    """Load ECG samples from a file path.

    Supported formats:
    - `.npy`: NumPy binary array
    - text/CSV-like files readable by `numpy.loadtxt`
    """
    path = Path(input_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"ECG input file not found: {path}")

    if path.suffix.lower() == ".npy":
        data = np.asarray(np.load(path), dtype=float)
    else:
        try:
            data = np.asarray(np.loadtxt(path, delimiter=","), dtype=float)
        except ValueError:
            data = np.asarray(np.loadtxt(path), dtype=float)

    if data.ndim == 0:
        raise ValueError(f"No ECG samples were found in file: {path}")
    if data.ndim == 1:
        return data

    if column_index < 0 or column_index >= data.shape[1]:
        raise ValueError(
            f"column_index {column_index} is out of bounds for input shape {data.shape}"
        )
    return data[:, column_index]


def acquire_ecg_signal(
    input_file: Optional[str],
    sampling_rate: int,
    duration_seconds: int = DEFAULT_SYNTHETIC_DURATION_SECONDS,
) -> Tuple[np.ndarray, str]:
    """Load ECG from file or generate synthetic ECG when no file is supplied."""
    if input_file:
        signal = load_ecg_signal_from_file(input_file)
        return signal, f"file:{Path(input_file).expanduser().resolve()}"

    signal = generate_synthetic_ecg(duration_seconds=duration_seconds, sampling_rate=sampling_rate)
    return signal, "synthetic"
