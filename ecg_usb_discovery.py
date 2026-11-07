"""Identify VER_V-A firmware without interpreting arbitrary serial data as ECG."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable

IDENTIFY_REQUEST = b"VER_VA_IDENTIFY\n"
IDENTIFY_RESPONSE = b"VER_VA,1,ESP32S3,ADS1115,250"
BAUD_RATE = 230400
ESPRESSIF_USB_VENDOR_ID = 0x303A


@dataclass(frozen=True)
class DiscoveryResult:
    ports: tuple[str, ...]
    errors: tuple[str, ...]


def is_esp32_port(port: object) -> bool:
    """Conservatively select ports explicitly identified as Espressif or ESP32."""
    if getattr(port, "vid", None) == ESPRESSIF_USB_VENDOR_ID:
        return True
    description = " ".join(
        str(getattr(port, name, "") or "") for name in ("description", "manufacturer", "product")
    ).casefold()
    return "espressif" in description or "esp32" in description


def discover_devices(
    ports: Iterable[object],
    serial_factory: Callable[..., object],
    *,
    manual: bool = False,
    cancelled: Callable[[], bool] = lambda: False,
    timeout: float = 3.5,
) -> DiscoveryResult:
    """Probe selected ports for a versioned identity, allowing for a board reset on open."""
    found: list[str] = []
    errors: list[str] = []
    for port in ports:
        if cancelled():
            break
        if not manual and not is_esp32_port(port):
            continue
        name = str(getattr(port, "device"))
        try:
            with serial_factory(name, BAUD_RATE, timeout=0.1, write_timeout=0.2) as connection:
                deadline = time.monotonic() + timeout
                next_request = 0.0
                while time.monotonic() < deadline and not cancelled():
                    now = time.monotonic()
                    if now >= next_request:
                        connection.write(IDENTIFY_REQUEST)
                        next_request = now + 0.5
                    if connection.readline().strip() == IDENTIFY_RESPONSE:
                        found.append(name)
                        break
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
    return DiscoveryResult(tuple(found), tuple(errors))
