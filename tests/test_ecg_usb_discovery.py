"""Tests for USB identity probing."""

import unittest
from types import SimpleNamespace

from ecg_usb_discovery import IDENTIFY_REQUEST, IDENTIFY_RESPONSE, discover_devices, is_esp32_port


class FakeSerial:
    def __init__(self, response: bytes):
        self.response = response
        self.requests = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def write(self, data):
        self.requests.append(data)

    def readline(self):
        return self.response


class TestUSBDiscovery(unittest.TestCase):
    def test_selects_espressif_metadata_not_unrelated_ports(self):
        self.assertTrue(is_esp32_port(SimpleNamespace(vid=0x303A)))
        self.assertTrue(is_esp32_port(SimpleNamespace(description="ESP32-S3 USB")))
        self.assertFalse(is_esp32_port(SimpleNamespace(vid=0x1234, description="Other USB")))

    def test_identifies_exact_response_and_closes_port(self):
        port = SimpleNamespace(device="COM4", vid=0x303A)
        serial = FakeSerial(IDENTIFY_RESPONSE + b"\n")
        result = discover_devices([port], lambda *args, **kwargs: serial)
        self.assertEqual(result.ports, ("COM4",))
        self.assertEqual(serial.requests, [IDENTIFY_REQUEST])
        self.assertTrue(serial.closed)

    def test_csv_stream_does_not_count_as_identity(self):
        port = SimpleNamespace(device="COM4", vid=0x303A)
        result = discover_devices([port], lambda *args, **kwargs: FakeSerial(b"123,4,567\n"), timeout=0.001)
        self.assertEqual(result.ports, ())

    def test_identity_can_follow_stream_samples(self):
        port = SimpleNamespace(device="COM4", vid=0x303A)
        serial = FakeSerial(b"")
        lines = iter([b"123,4,567\n", IDENTIFY_RESPONSE + b"\r\n"])
        serial.readline = lambda: next(lines)
        result = discover_devices([port], lambda *args, **kwargs: serial)
        self.assertEqual(result.ports, ("COM4",))

    def test_manual_port_bypasses_metadata_filter(self):
        port = SimpleNamespace(device="COM4", vid=0x1234)
        result = discover_devices([port], lambda *args, **kwargs: FakeSerial(IDENTIFY_RESPONSE), manual=True)
        self.assertEqual(result.ports, ("COM4",))

    def test_reports_port_errors_and_continues(self):
        ports = [SimpleNamespace(device="COM1", vid=0x303A), SimpleNamespace(device="COM2", vid=0x303A)]

        def open_port(name, *args, **kwargs):
            if name == "COM1":
                raise OSError("port busy")
            return FakeSerial(IDENTIFY_RESPONSE)

        result = discover_devices(ports, open_port)
        self.assertEqual(result.ports, ("COM2",))
        self.assertIn("COM1: port busy", result.errors)


if __name__ == "__main__":
    unittest.main()
