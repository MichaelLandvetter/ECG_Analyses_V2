"""NeuroKit2-based ECG processing and metric extraction."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


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
    if np.isnan(value_f):
        return None
    return value_f


def analyze_ecg(signal: Sequence[float] | np.ndarray, sampling_rate: int) -> Dict[str, Any]:
    """Analyze an ECG signal and return metrics plus processed artifacts."""
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be greater than 0")

    signal_array = np.asarray(signal, dtype=float).flatten()
    if signal_array.size < 3:
        raise ValueError("ECG signal must contain at least 3 samples")

    nk = _ensure_neurokit2()

    try:
        processed_signals, processing_info = nk.ecg_process(
            signal_array, sampling_rate=sampling_rate
        )
    except Exception as exc:
        raise RuntimeError(f"NeuroKit2 failed to process ECG signal: {exc}") from exc

    r_peaks = np.asarray(processing_info.get("ECG_R_Peaks", []), dtype=int)
    rr_intervals_ms = (
        np.diff(r_peaks) / float(sampling_rate) * 1000.0 if r_peaks.size > 1 else np.array([])
    )

    mean_heart_rate = _safe_float(np.nanmean(processed_signals["ECG_Rate"]))

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
        "cleaned_signal": np.asarray(processed_signals["ECG_Clean"], dtype=float).tolist(),
        "heart_rate_trace_bpm": np.asarray(processed_signals["ECG_Rate"], dtype=float).tolist(),
    }

    return {"metrics": metrics, "artifacts": artifacts}
