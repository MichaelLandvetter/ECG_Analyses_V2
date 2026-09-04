"""NeuroKit2-based ECG processing, filtering, and report table preparation."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from ecg_config import (
    DEFAULT_FILTER_HIGH_CUT_HZ,
    DEFAULT_FILTER_LOW_CUT_HZ,
    DEFAULT_FILTER_MODE,
    DEFAULT_POWERLINE_FREQUENCY_HZ,
    DEFAULT_RPEAK_METHOD,
)

SUPPORTED_FILTER_MODES: tuple[str, ...] = ("NeuroKit2 ecg_clean", "Butterworth bandpass")
SUPPORTED_RPEAK_METHODS: tuple[str, ...] = (
    "neurokit",
    "pantompkins1985",
    "engzeemod2012",
    "hamilton2002",
)


class NeuroKit2UnavailableError(ImportError):
    """Raised when NeuroKit2 is required but not installed."""


def _ensure_neurokit2():
    try:
        import neurokit2 as nk
    except ImportError as exc:
        raise NeuroKit2UnavailableError(
            "NeuroKit2 is required for ECG analysis. Install dependencies with: "
            "pip install neurokit2 numpy scipy pandas"
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


def _coerce_array(values: Any, expected_length: int) -> np.ndarray:
    if expected_length <= 0:
        return np.array([], dtype=float)
    if values is None:
        return np.full(expected_length, np.nan, dtype=float)
    try:
        array = np.asarray(values, dtype=float).flatten()
    except (TypeError, ValueError):
        return np.full(expected_length, np.nan, dtype=float)
    if array.size == expected_length:
        return array
    if array.size == 0:
        return np.full(expected_length, np.nan, dtype=float)
    aligned = np.full(expected_length, np.nan, dtype=float)
    aligned[: min(expected_length, array.size)] = array[:expected_length]
    return aligned


def validate_filter_mode(filter_mode: str) -> str:
    """Validate the configured filter mode and return its canonical label."""
    normalized = filter_mode.strip().lower()
    if normalized == "neurokit2 ecg_clean":
        return "NeuroKit2 ecg_clean"
    if normalized == "butterworth bandpass":
        return "Butterworth bandpass"
    raise ValueError(f"Unsupported filter mode: {filter_mode}")


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


def validate_powerline_frequency(notch_frequency_hz: float, sampling_rate: int) -> float:
    """Validate an optional power-line notch frequency."""
    if notch_frequency_hz <= 0:
        raise ValueError("powerline_frequency_hz must be greater than 0")
    nyquist_hz = sampling_rate / 2.0
    if notch_frequency_hz >= nyquist_hz:
        raise ValueError(f"powerline_frequency_hz must be less than Nyquist ({nyquist_hz:.2f} Hz)")
    return notch_frequency_hz


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
            "pip install neurokit2 numpy scipy pandas"
        ) from exc

    validate_filter_settings(low_cut_hz=low_cut_hz, high_cut_hz=high_cut_hz, sampling_rate=sampling_rate)
    signal_array = np.asarray(signal, dtype=float).flatten()
    if signal_array.size < 3:
        raise ValueError("ECG signal must contain at least 3 samples")

    nyquist_hz = sampling_rate / 2.0
    b, a = butter(order, [low_cut_hz / nyquist_hz, high_cut_hz / nyquist_hz], btype="band")
    return filtfilt(b, a, signal_array)


def apply_powerline_notch(
    signal: Sequence[float] | np.ndarray,
    sampling_rate: int,
    notch_frequency_hz: float,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Apply a power-line notch filter to the signal."""
    try:
        from scipy.signal import filtfilt, iirnotch
    except Exception as exc:
        raise RuntimeError(
            "SciPy is required for power-line notch filtering. Install dependencies with: "
            "pip install neurokit2 numpy scipy pandas"
        ) from exc

    validate_powerline_frequency(notch_frequency_hz=notch_frequency_hz, sampling_rate=sampling_rate)
    signal_array = np.asarray(signal, dtype=float).flatten()
    b, a = iirnotch(notch_frequency_hz, quality_factor, fs=float(sampling_rate))
    return filtfilt(b, a, signal_array)


def clean_ecg_signal(
    signal: Sequence[float] | np.ndarray,
    sampling_rate: int,
    filter_mode: str = DEFAULT_FILTER_MODE,
    low_cut_hz: float = DEFAULT_FILTER_LOW_CUT_HZ,
    high_cut_hz: float = DEFAULT_FILTER_HIGH_CUT_HZ,
    powerline_frequency_hz: float = DEFAULT_POWERLINE_FREQUENCY_HZ,
) -> np.ndarray:
    """Return the filtered ECG signal using the selected filter strategy."""
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be greater than 0")

    signal_array = np.asarray(signal, dtype=float).flatten()
    if signal_array.size < 3:
        raise ValueError("ECG signal must contain at least 3 samples")

    filter_mode_canonical = validate_filter_mode(filter_mode)
    if filter_mode_canonical == "Butterworth bandpass":
        filtered_signal = apply_butterworth_bandpass(
            signal=signal_array,
            sampling_rate=sampling_rate,
            low_cut_hz=low_cut_hz,
            high_cut_hz=high_cut_hz,
        )
        return apply_powerline_notch(
            signal=filtered_signal,
            sampling_rate=sampling_rate,
            notch_frequency_hz=powerline_frequency_hz,
        )

    nk = _ensure_neurokit2()
    try:
        return np.asarray(
            nk.ecg_clean(
                signal_array,
                sampling_rate=sampling_rate,
                method="neurokit",
                powerline=powerline_frequency_hz,
            ),
            dtype=float,
        )
    except Exception as exc:
        raise RuntimeError(f"NeuroKit2 failed to clean ECG signal: {exc}") from exc


def estimate_live_heart_rate_trace(
    signal: Sequence[float] | np.ndarray,
    sampling_rate: int,
) -> np.ndarray:
    """Estimate a lightweight streaming heart-rate trace from a filtered ECG signal."""
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be greater than 0")

    signal_array = np.asarray(signal, dtype=float).flatten()
    if signal_array.size < max(3, int(0.8 * sampling_rate)):
        return np.full(signal_array.size, np.nan, dtype=float)

    try:
        from scipy.signal import find_peaks

        minimum_distance = max(1, int(0.3 * sampling_rate))
        prominence = max(0.05, float(np.nanstd(signal_array)) * 0.5)
        peaks, _properties = find_peaks(signal_array, distance=minimum_distance, prominence=prominence)
    except Exception:
        candidate = np.where(
            (signal_array[1:-1] > signal_array[:-2]) & (signal_array[1:-1] >= signal_array[2:]),
        )[0] + 1
        minimum_distance = max(1, int(0.3 * sampling_rate))
        peaks = candidate[::minimum_distance] if candidate.size else np.array([], dtype=int)

    peaks = np.asarray(peaks, dtype=int)
    if peaks.size < 2:
        return np.full(signal_array.size, np.nan, dtype=float)

    hr_trace = np.full(signal_array.size, np.nan, dtype=float)
    for left_peak, right_peak in zip(peaks[:-1], peaks[1:], strict=False):
        rr_samples = right_peak - left_peak
        if rr_samples <= 0:
            continue
        bpm = 60.0 * float(sampling_rate) / float(rr_samples)
        if not np.isfinite(bpm):
            continue
        hr_trace[left_peak : right_peak + 1] = bpm
    return hr_trace


def detect_r_peaks(
    signal: Sequence[float] | np.ndarray,
    sampling_rate: int,
    method: str = DEFAULT_RPEAK_METHOD,
    filter_mode: str = DEFAULT_FILTER_MODE,
    low_cut_hz: float = DEFAULT_FILTER_LOW_CUT_HZ,
    high_cut_hz: float = DEFAULT_FILTER_HIGH_CUT_HZ,
    powerline_frequency_hz: float = DEFAULT_POWERLINE_FREQUENCY_HZ,
) -> list[int]:
    """Detect R peaks using NeuroKit2 and return sample indices."""
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be greater than 0")

    method_normalized = method.strip().lower()
    if method_normalized not in SUPPORTED_RPEAK_METHODS:
        raise ValueError(f"Unsupported R-peak method: {method}")

    cleaned_signal = clean_ecg_signal(
        signal=signal,
        sampling_rate=sampling_rate,
        filter_mode=filter_mode,
        low_cut_hz=low_cut_hz,
        high_cut_hz=high_cut_hz,
        powerline_frequency_hz=powerline_frequency_hz,
    )

    nk = _ensure_neurokit2()
    _, processing_info = nk.ecg_peaks(
        cleaned_signal,
        sampling_rate=sampling_rate,
        method=method_normalized,
        correct_artifacts=True,
    )
    return np.asarray(processing_info.get("ECG_R_Peaks", []), dtype=int).tolist()


def _build_template_data(
    filtered_signal: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: int,
) -> dict[str, Any]:
    pre_samples = max(1, int(round(0.25 * sampling_rate)))
    post_samples = max(1, int(round(0.40 * sampling_rate)))
    sample_offsets = np.arange(-pre_samples, post_samples + 1, dtype=int)

    beat_snippets: list[list[float]] = []
    for peak_index, peak_sample in enumerate(r_peaks, start=1):
        start = int(peak_sample) - pre_samples
        stop = int(peak_sample) + post_samples + 1
        if start < 0 or stop > filtered_signal.size:
            continue
        beat_snippets.append(filtered_signal[start:stop].astype(float).tolist())

    beat_matrix = np.asarray(beat_snippets, dtype=float) if beat_snippets else np.empty((0, sample_offsets.size))
    if beat_matrix.size:
        average_template = np.nanmean(beat_matrix, axis=0)
    else:
        average_template = np.full(sample_offsets.size, np.nan, dtype=float)

    average_template_rows = [
        {
            "sample_offset": int(sample_offset),
            "time_offset_seconds": float(sample_offset / sampling_rate),
            "average_amplitude": _safe_float(amplitude),
            "beats_contributed": int(beat_matrix.shape[0]),
        }
        for sample_offset, amplitude in zip(sample_offsets, average_template, strict=False)
    ]

    return {
        "beat_sample_offsets": sample_offsets.tolist(),
        "beat_time_offsets_seconds": (sample_offsets / float(sampling_rate)).astype(float).tolist(),
        "beat_snippets": beat_snippets,
        "average_template": average_template.astype(float).tolist(),
        "average_template_wave": average_template_rows,
    }


def _build_landmark_rows(
    r_peaks: np.ndarray,
    delineation_info: dict[str, Any],
    sampling_rate: int,
) -> list[dict[str, Any]]:
    field_map = {
        "p_onset_sample": "ECG_P_Onsets",
        "p_peak_sample": "ECG_P_Peaks",
        "q_peak_sample": "ECG_Q_Peaks",
        "qrs_onset_sample": "ECG_R_Onsets",
        "qrs_offset_sample": "ECG_R_Offsets",
        "s_peak_sample": "ECG_S_Peaks",
        "t_peak_sample": "ECG_T_Peaks",
        "t_offset_sample": "ECG_T_Offsets",
    }
    aligned_fields = {
        output_name: _coerce_array(delineation_info.get(source_name), r_peaks.size)
        for output_name, source_name in field_map.items()
    }

    rows: list[dict[str, Any]] = []
    for beat_index, r_peak_sample in enumerate(r_peaks, start=1):
        row: dict[str, Any] = {
            "beat_index": beat_index,
            "r_peak_sample": int(r_peak_sample),
            "r_peak_time_seconds": float(r_peak_sample / sampling_rate),
        }
        for field_name, values in aligned_fields.items():
            row[field_name] = _safe_float(values[beat_index - 1])
        rows.append(row)
    return rows


def _build_continuous_rows(
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    heart_rate_trace: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: int,
) -> list[dict[str, Any]]:
    r_peak_mask = np.zeros(raw_signal.size, dtype=int)
    valid_r_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < raw_signal.size)]
    r_peak_mask[valid_r_peaks] = 1

    return [
        {
            "sample_index": int(sample_index),
            "time_seconds": float(sample_index / sampling_rate),
            "raw_ecg": float(raw_signal[sample_index]),
            "filtered_ecg": float(filtered_signal[sample_index]),
            "heart_rate_bpm": _safe_float(heart_rate_trace[sample_index]),
            "is_r_peak": int(r_peak_mask[sample_index]),
        }
        for sample_index in range(raw_signal.size)
    ]


def analyze_ecg(
    signal: Sequence[float] | np.ndarray,
    sampling_rate: int,
    rpeak_method: str = DEFAULT_RPEAK_METHOD,
    filter_mode: str = DEFAULT_FILTER_MODE,
    low_cut_hz: float = DEFAULT_FILTER_LOW_CUT_HZ,
    high_cut_hz: float = DEFAULT_FILTER_HIGH_CUT_HZ,
    powerline_frequency_hz: float = DEFAULT_POWERLINE_FREQUENCY_HZ,
) -> Dict[str, Any]:
    """Analyze an ECG signal and return plotting artifacts plus export-ready tables."""
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be greater than 0")

    signal_array = np.asarray(signal, dtype=float).flatten()
    if signal_array.size < 3:
        raise ValueError("ECG signal must contain at least 3 samples")

    method_normalized = rpeak_method.strip().lower()
    if method_normalized not in SUPPORTED_RPEAK_METHODS:
        raise ValueError(f"Unsupported R-peak method: {rpeak_method}")

    filtered_signal = clean_ecg_signal(
        signal=signal_array,
        sampling_rate=sampling_rate,
        filter_mode=filter_mode,
        low_cut_hz=low_cut_hz,
        high_cut_hz=high_cut_hz,
        powerline_frequency_hz=powerline_frequency_hz,
    )

    nk = _ensure_neurokit2()
    try:
        _, processing_info = nk.ecg_peaks(
            filtered_signal,
            sampling_rate=sampling_rate,
            method=method_normalized,
            correct_artifacts=True,
        )
        r_peaks = np.asarray(processing_info.get("ECG_R_Peaks", []), dtype=int)
        heart_rate_trace = np.asarray(
            nk.signal_rate(r_peaks, sampling_rate=sampling_rate, desired_length=signal_array.size),
            dtype=float,
        )
    except Exception as exc:
        raise RuntimeError(f"NeuroKit2 failed to process ECG signal: {exc}") from exc

    try:
        _, delineation_info = nk.ecg_delineate(
            filtered_signal,
            r_peaks,
            sampling_rate=sampling_rate,
            method="dwt",
            show=False,
        )
    except Exception:
        delineation_info = {}

    rr_intervals_ms = (
        np.diff(r_peaks) / float(sampling_rate) * 1000.0 if r_peaks.size > 1 else np.array([], dtype=float)
    )
    mean_heart_rate = _safe_float(np.nanmean(heart_rate_trace)) if heart_rate_trace.size else None

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
        "filter_mode": validate_filter_mode(filter_mode),
        "filter_low_cut_hz": float(low_cut_hz),
        "filter_high_cut_hz": float(high_cut_hz),
        "powerline_frequency_hz": float(powerline_frequency_hz),
        "rpeak_method": method_normalized,
        "r_peak_count": int(r_peaks.size),
        "mean_heart_rate_bpm": mean_heart_rate,
        "rr_mean_ms": _safe_float(np.nanmean(rr_intervals_ms)) if rr_intervals_ms.size else None,
        "rr_std_ms": _safe_float(np.nanstd(rr_intervals_ms)) if rr_intervals_ms.size else None,
        "hrv_rmssd_ms": hrv_rmssd,
        "hrv_sdnn_ms": hrv_sdnn,
    }

    template_data = _build_template_data(
        filtered_signal=filtered_signal,
        r_peaks=r_peaks,
        sampling_rate=sampling_rate,
    )
    beat_landmark_rows = _build_landmark_rows(
        r_peaks=r_peaks,
        delineation_info=delineation_info,
        sampling_rate=sampling_rate,
    )
    continuous_rows = _build_continuous_rows(
        raw_signal=signal_array,
        filtered_signal=filtered_signal,
        heart_rate_trace=heart_rate_trace,
        r_peaks=r_peaks,
        sampling_rate=sampling_rate,
    )

    artifacts: Dict[str, Any] = {
        "time_seconds": (np.arange(signal_array.size) / float(sampling_rate)).astype(float).tolist(),
        "raw_signal": signal_array.astype(float).tolist(),
        "filtered_signal": filtered_signal.astype(float).tolist(),
        "r_peaks": r_peaks.tolist(),
        "heart_rate_trace_bpm": heart_rate_trace.astype(float).tolist(),
        "beat_sample_offsets": template_data["beat_sample_offsets"],
        "beat_time_offsets_seconds": template_data["beat_time_offsets_seconds"],
        "beat_snippets": template_data["beat_snippets"],
        "average_template": template_data["average_template"],
    }

    return {
        "metrics": metrics,
        "artifacts": artifacts,
        "report_tables": {
            "summary_metrics": [metrics],
            "average_template_wave": template_data["average_template_wave"],
            "beat_morphology_landmarks": beat_landmark_rows,
            "continuous_time_series": continuous_rows,
        },
    }
