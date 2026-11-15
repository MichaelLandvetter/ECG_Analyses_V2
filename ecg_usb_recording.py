"""Recording primitives for the USB ECG stream."""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

from tools.usb_ecg_stream_utils import StreamSample, UINT32_MODULUS


class USBRecording:
    """Write raw samples and preserve device timing for later integrity checks."""

    def __init__(self, path: Path, sample_rate_hz: int = 250) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        self.path = path
        self.sample_rate_hz = sample_rate_hz
        self.sample_count = 0
        self.discontinuities = 0
        self._previous: StreamSample | None = None
        self._file: TextIO = path.open("x", encoding="utf-8", newline="")
        try:
            self._file.write(f"# sample_rate_hz={sample_rate_hz}\n")
            self._file.write("# raw_ecg,device_timestamp_ms,device_sample_index\n")
            self._file.flush()
        except OSError:
            self._file.close()
            raise

    def append(self, sample: StreamSample) -> None:
        if self._previous is not None:
            index_delta = (sample.sample_index - self._previous.sample_index) % UINT32_MODULUS
            time_delta = (sample.timestamp_ms - self._previous.timestamp_ms) % UINT32_MODULUS
            expected_ms = 1000.0 / self.sample_rate_hz
            if index_delta != 1 or not 0 < time_delta <= expected_ms + 2.0:
                self.discontinuities += 1
        self._file.write(f"{sample.raw_ecg},{sample.timestamp_ms},{sample.sample_index}\n")
        self.sample_count += 1
        self._previous = sample

    def resume(self) -> None:
        """Start a new recorded segment after an intentional pause."""
        self._file.write("# acquisition resumed after pause\n")
        self._previous = None

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()
