"""NeuroKit2-based ECG processing and metric extraction."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from ecg_config import DEFAULT_RPEAK_METHOD


class NeuroKit2UnavailableError(ImportError):
    """Raised when NeuroKit2 is required but not installed."""


def _ensure_neurokit2():
    try:
        import neurokit2 as nk
    except ImportError as exc:
        raise NeuroKit2UnavailableError(
            "NeuroKit2 is required for ECG analysis. Install dependencies with: "
            "pip install neurokit2 numpy scipy pandas matplotlib"
        ) from exc
    return nk


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value_f):
        return None
    return value_f


def validate_filter_settings(low_cut_hz: float, high_cut_hz: float, sampling_rate: int) -> tuple[float, float]:
    """Validate bandpass settings against the sampling rate."""
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be greater than 0")
    nyquist_hz = sampling_rate / 2.0
    if low_cut_hz <= 0.0:
        raise ValueError("low_cut_hz must be greater than 0")
    if high_cut_hz <= low_cut_hz:
        raise ValueError("high_cut_hz must be greater than low_cut_hz")
    if high_cut_hz >= nyquist_hz:
        raise ValueError(f"high_cut_hz must be less than Nyquist ({nyquist_hz:.2f} Hz)")
    return low_cut_hz, high_cut_hz


def apply_butterworth_bandpass(
    signal: Sequence[float] | np.ndarray,
    sampling_rate: int,
    low_cut_hz: float,
    high_cut_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter and return the filtered signal."""
    try:
        from scipy.signal import butter, filtfilt
    except Exception as exc:
        raise RuntimeError(
            "SciPy is required for Butterworth filtering. Install dependencies with: "
            "pip install neurokit2 numpy scipy pandas matplotlib"
        ) from exc

    validate_filter_settings(low_cut_hz=low_cut_hz, high_cut_hz=high_cut_hz, sampling_rate=sampling_rate)
    signal_array = np.asarray(signal, dtype=float).flatten()
    if signal_array.size < 3:
        raise ValueError("ECG signal must contain at least 3 samples")

    nyquist_hz = sampling_rate / 2.0
    b, a = butter(order, [low_cut_hz / nyquist_hz, high_cut_hz / nyquist_hz], btype="band")
    return filtfilt(b, a, signal_array)


def detect_r_peaks(
    signal: Sequence[float] | np.ndarray,
    sampling_rate: int,
    method: str = DEFAULT_RPEAK_METHOD,
) -> list[int]:
    """Detect R peaks using NeuroKit2 and return sample indices."""
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be greater than 0")
    supported_methods = ("neurokit", "pantompkins1985", "engzeemod2012", "hamilton2002")
    method_normalized = method.strip().lower()
    if method_normalized not in supported_methods:
        raise ValueError(f"Unsupported R-peak method: {method}")

    signal_array = np.asarray(signal, dtype=float).flatten()
    if signal_array.size < 3:
        raise ValueError("ECG signal must contain at least 3 samples")

    nk = _ensure_neurokit2()
    cleaned_signal = nk.ecg_clean(signal_array, sampling_rate=sampling_rate, method="neurokit")
    _, processing_info = nk.ecg_peaks(
        cleaned_signal,
        sampling_rate=sampling_rate,
        method=method_normalized,
        correct_artifacts=True,
    )
    return np.asarray(processing_info.get("ECG_R_Peaks", []), dtype=int).tolist()


def analyze_ecg(
    signal: Sequence[float] | np.ndarray,
    sampling_rate: int,
    rpeak_method: str = DEFAULT_RPEAK_METHOD,
) -> Dict[str, Any]:
    """Analyze an ECG signal and return metrics plus processed artifacts."""
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be greater than 0")

    signal_array = np.asarray(signal, dtype=float).flatten()
    if signal_array.size < 3:
        raise ValueError("ECG signal must contain at least 3 samples")

    nk = _ensure_neurokit2()
    rpeak_method_normalized = rpeak_method.strip().lower()

    try:
        cleaned_signal = np.asarray(
            nk.ecg_clean(signal_array, sampling_rate=sampling_rate, method="neurokit"),
            dtype=float,
        )
        _, processing_info = nk.ecg_peaks(
            cleaned_signal,
            sampling_rate=sampling_rate,
            method=rpeak_method_normalized,
            correct_artifacts=True,
        )
        r_peaks = np.asarray(processing_info.get("ECG_R_Peaks", []), dtype=int)
        heart_rate_trace = np.asarray(
            nk.signal_rate(r_peaks, sampling_rate=sampling_rate, desired_length=signal_array.size),
            dtype=float,
        )
    except Exception as exc:
        raise RuntimeError(f"NeuroKit2 failed to process ECG signal: {exc}") from exc

    rr_intervals_ms = (
        np.diff(r_peaks) / float(sampling_rate) * 1000.0 if r_peaks.size > 1 else np.array([])
    )

    mean_heart_rate = _safe_float(np.nanmean(heart_rate_trace))

    hrv_rmssd = None
    hrv_sdnn = None
    if r_peaks.size > 1:
        try:
            hrv_time = nk.hrv_time(processing_info, sampling_rate=sampling_rate, show=False)
            hrv_rmssd = _safe_float(hrv_time.iloc[0].get("HRV_RMSSD"))
            hrv_sdnn = _safe_float(hrv_time.iloc[0].get("HRV_SDNN"))
        except Exception:
            hrv_rmssd = None
            hrv_sdnn = None

    metrics: Dict[str, Any] = {
        "sampling_rate_hz": int(sampling_rate),
        "num_samples": int(signal_array.size),
        "duration_seconds": float(signal_array.size / sampling_rate),
        "r_peak_count": int(r_peaks.size),
        "mean_heart_rate_bpm": mean_heart_rate,
        "rr_mean_ms": _safe_float(np.nanmean(rr_intervals_ms)) if rr_intervals_ms.size else None,
        "rr_std_ms": _safe_float(np.nanstd(rr_intervals_ms)) if rr_intervals_ms.size else None,
        "hrv_rmssd_ms": hrv_rmssd,
        "hrv_sdnn_ms": hrv_sdnn,
    }

    artifacts: Dict[str, Any] = {
        "r_peaks": r_peaks.tolist(),
        "cleaned_signal": cleaned_signal.tolist(),
        "heart_rate_trace_bpm": heart_rate_trace.tolist(),
    }

    return {"metrics": metrics, "artifacts": artifacts}
