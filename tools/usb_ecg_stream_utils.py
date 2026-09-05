"""Utilities for the standalone USB ECG stream tester."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque

UINT32_MODULUS = 2**32


@dataclass(frozen=True)
class StreamSample:
    """Single parsed ECG stream sample."""

    timestamp_ms: int
    sample_index: int
    raw_ecg: int


@dataclass(frozen=True)
class IntervalStats:
    """Summary stats for device timestamp intervals."""

    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float


def parse_stream_line(line: str) -> StreamSample:
    """Parse one `timestamp_ms,sample_index,raw_ecg` CSV line."""
    stripped = line.strip()
    if not stripped:
        raise ValueError("Empty serial line")

    parts = [part.strip() for part in stripped.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 CSV fields, received {len(parts)}")

    try:
        timestamp_ms = int(parts[0], 10)
        sample_index = int(parts[1], 10)
        raw_ecg = int(parts[2], 10)
    except ValueError as exc:
        raise ValueError(f"Invalid integer field in serial line: {stripped}") from exc

    if timestamp_ms < 0:
        raise ValueError("timestamp_ms must be non-negative")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")

    return StreamSample(timestamp_ms=timestamp_ms, sample_index=sample_index % UINT32_MODULUS, raw_ecg=raw_ecg)


def estimate_dropped_samples(previous_index: int | None, current_index: int) -> int:
    """Estimate dropped samples from `sample_index` gaps with uint32 rollover support."""
    if previous_index is None:
        return 0

    delta = (current_index - previous_index) % UINT32_MODULUS
    if delta <= 1:
        return 0
    return delta - 1


class StreamDiagnostics:
    """Track lightweight transport diagnostics for the tester UI."""

    def __init__(self, sample_rate_hz: int, history_seconds: float = 3.0, interval_history: int = 512) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be greater than 0")
        if history_seconds <= 0:
            raise ValueError("history_seconds must be greater than 0")
        if interval_history <= 1:
            raise ValueError("interval_history must be greater than 1")

        self.sample_rate_hz = int(sample_rate_hz)
        self.total_samples = 0
        self.dropped_samples = 0
        self.parse_errors = 0
        self._last_sample_index: int | None = None
        self._last_timestamp_ms: int | None = None
        self._recent_host_times: Deque[float] = deque(maxlen=max(2, int(math.ceil(self.sample_rate_hz * history_seconds))))
        self._recent_intervals_ms: Deque[float] = deque(maxlen=interval_history)

    def reset(self) -> None:
        """Reset all counters and histories."""
        self.total_samples = 0
        self.dropped_samples = 0
        self.parse_errors = 0
        self._last_sample_index = None
        self._last_timestamp_ms = None
        self._recent_host_times.clear()
        self._recent_intervals_ms.clear()

    def reset_gap_reference(self) -> None:
        """Drop gap/timestamp continuity when pausing or restarting capture."""
        self._last_sample_index = None
        self._last_timestamp_ms = None
        self._recent_host_times.clear()
        self._recent_intervals_ms.clear()

    def note_parse_error(self) -> None:
        """Increment parse error counter."""
        self.parse_errors += 1

    def observe_sample(self, sample: StreamSample, host_time: float) -> None:
        """Record a newly received sample."""
        self.total_samples += 1
        self.dropped_samples += estimate_dropped_samples(self._last_sample_index, sample.sample_index)

        if self._last_timestamp_ms is not None:
            interval_ms = float(sample.timestamp_ms - self._last_timestamp_ms)
            if interval_ms >= 0.0:
                self._recent_intervals_ms.append(interval_ms)

        self._last_sample_index = sample.sample_index
        self._last_timestamp_ms = sample.timestamp_ms
        self._recent_host_times.append(float(host_time))

    @property
    def incoming_rate_hz(self) -> float:
        """Estimate incoming sample rate from recent host-side arrival timestamps."""
        if len(self._recent_host_times) < 2:
            return 0.0
        elapsed = self._recent_host_times[-1] - self._recent_host_times[0]
        if elapsed <= 0.0:
            return 0.0
        return (len(self._recent_host_times) - 1) / elapsed

    @property
    def interval_stats(self) -> IntervalStats | None:
        """Return interval summary stats from recent device timestamps."""
        if not self._recent_intervals_ms:
            return None
        values = list(self._recent_intervals_ms)
        mean_ms = sum(values) / len(values)
        variance = sum((value - mean_ms) ** 2 for value in values) / len(values)
        return IntervalStats(
            mean_ms=mean_ms,
            min_ms=min(values),
            max_ms=max(values),
            std_ms=math.sqrt(variance),
        )
