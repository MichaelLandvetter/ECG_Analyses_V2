"""Serial transport for the VER_V-A ECG device."""

from __future__ import annotations

import time
from typing import Callable, Protocol

from ecg_usb_discovery import BAUD_RATE, IDENTIFY_REQUEST, IDENTIFY_RESPONSE
from tools.usb_ecg_stream_utils import StreamSample, parse_stream_line, split_serial_lines

MAX_PENDING_BYTES = 4096


class USBStreamError(Exception):
    """The device did not identify itself or the stream became invalid."""


class SerialConnection(Protocol):
    """Operations required from a pyserial-compatible connection."""

    @property
    def in_waiting(self) -> int: ...

    def read(self, size: int) -> bytes: ...

    def write(self, data: bytes) -> int: ...

    def close(self) -> None: ...


class ECGSerialStream:
    """Read verified ECG samples from one serial connection."""

    def __init__(
        self,
        port: str,
        serial_factory: Callable[..., SerialConnection],
        *,
        identity_timeout: float = 3.5,
        silence_timeout: float = 3.0,
    ) -> None:
        if identity_timeout <= 0 or silence_timeout <= 0:
            raise ValueError("Timeouts must be positive")
        self._serial = serial_factory(port, BAUD_RATE, timeout=0.1, write_timeout=0.2)
        self._pending = b""
        self._queued_lines: list[bytes] = []
        self._identity_timeout = identity_timeout
        self._silence_timeout = silence_timeout
        self._last_sample_at: float | None = None
        self._identified = False

    def close(self) -> None:
        """Release the serial port."""
        self._serial.close()

    def _read_lines(self) -> list[bytes]:
        chunk = self._serial.read(max(1, self._serial.in_waiting))
        lines, self._pending = split_serial_lines(self._pending, chunk)
        if len(self._pending) > MAX_PENDING_BYTES or any(len(line) > MAX_PENDING_BYTES for line in lines):
            raise USBStreamError("Serial message exceeded the maximum length.")
        return lines

    def identify(self) -> None:
        """Confirm the firmware identity before accepting samples."""
        deadline = time.monotonic() + self._identity_timeout
        next_request = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_request:
                written = self._serial.write(IDENTIFY_REQUEST)
                if written != len(IDENTIFY_REQUEST):
                    raise USBStreamError("Could not send the complete identification request.")
                next_request = now + 0.5
            lines = self._read_lines()
            for index, line in enumerate(lines):
                if line.strip() == IDENTIFY_RESPONSE:
                    self._queued_lines = lines[index + 1:]
                    self._identified = True
                    self._last_sample_at = time.monotonic()
                    return
                if b"Failed to initialize ADS1115" in line:
                    raise USBStreamError(line.decode("utf-8", errors="replace"))
        raise USBStreamError("Device did not respond to the VER_V-A identification request.")

    def read_samples(self) -> list[StreamSample]:
        """Read a batch, failing explicitly on invalid data or a stalled stream."""
        if not self._identified:
            raise USBStreamError("Device must be identified before reading samples.")
        samples: list[StreamSample] = []
        lines, self._queued_lines = self._queued_lines, []
        if not lines:
            lines = self._read_lines()
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped == IDENTIFY_RESPONSE:
                continue
            try:
                text = stripped.decode("utf-8")
                sample = parse_stream_line(text)
            except (UnicodeDecodeError, ValueError) as exc:
                raise USBStreamError(f"Invalid serial data: {stripped!r}") from exc
            original_index = int(text.split(",")[1], 10)
            if (
                not 0 <= sample.timestamp_ms < 2**32
                or not 0 <= original_index < 2**32
                or not -32768 <= sample.raw_ecg <= 32767
            ):
                raise USBStreamError(f"Sample values are outside the device range: {text}")
            samples.append(sample)
        now = time.monotonic()
        if samples:
            self._last_sample_at = now
        elif self._last_sample_at is not None and now - self._last_sample_at >= self._silence_timeout:
            raise USBStreamError("No ECG samples received from the device.")
        return samples
