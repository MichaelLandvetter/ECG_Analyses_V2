"""Tests for the USB ECG serial transport."""

import unittest
from unittest.mock import patch

from ecg_usb_discovery import IDENTIFY_REQUEST, IDENTIFY_RESPONSE
from ecg_usb_stream import ECGSerialStream, USBStreamError


class FakeSerial:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.writes = []
        self.closed = False

    @property
    def in_waiting(self):
        return len(self.chunks[0]) if self.chunks else 0

    def read(self, size):
        return self.chunks.pop(0) if self.chunks else b""

    def write(self, data):
        self.writes.append(data)
        return len(data)

    def close(self):
        self.closed = True


class TestECGSerialStream(unittest.TestCase):
    def make_stream(self, chunks, **kwargs):
        serial = FakeSerial(chunks)
        stream = ECGSerialStream("COM4", lambda *args, **kw: serial, **kwargs)
        self.addCleanup(stream.close)
        return stream, serial

    def test_identifies_then_reads_fragmented_samples(self):
        stream, serial = self.make_stream([
            b"0,0,100\n" + IDENTIFY_RESPONSE[:12],
            IDENTIFY_RESPONSE[12:] + b"\n",
            b"4,1,120",
            b"00\n8,2,12001\n",
        ])
        stream.identify()
        self.assertEqual(serial.writes, [IDENTIFY_REQUEST])
        self.assertEqual(stream.read_samples(), [])
        samples = stream.read_samples()
        self.assertEqual([(s.timestamp_ms, s.sample_index, s.raw_ecg) for s in samples],
                         [(4, 1, 12000), (8, 2, 12001)])

    def test_requires_identification_before_reading(self):
        stream, _ = self.make_stream([])
        with self.assertRaisesRegex(USBStreamError, "identified"):
            stream.read_samples()

    def test_samples_after_identity_in_same_chunk_are_preserved(self):
        stream, _ = self.make_stream([IDENTIFY_RESPONSE + b"\n4,1,100\n"])
        stream.identify()
        self.assertEqual(stream.read_samples()[0].raw_ecg, 100)

    def test_invalid_and_oversized_data_fail_explicitly(self):
        for invalid in (b"not a sample\n", b"4,1,40000\n", b"4,4294967296,1\n", b"x" * 4097):
            with self.subTest(invalid=invalid[:30]):
                stream, _ = self.make_stream([IDENTIFY_RESPONSE + b"\n", invalid])
                stream.identify()
                with self.assertRaises(USBStreamError):
                    stream.read_samples()

    def test_firmware_initialization_error_is_reported(self):
        stream, _ = self.make_stream([b"Failed to initialize ADS1115 at I2C address 0x48\n"])
        with self.assertRaisesRegex(USBStreamError, "Failed to initialize ADS1115"):
            stream.identify()

    def test_stalled_stream_is_reported(self):
        stream, _ = self.make_stream([IDENTIFY_RESPONSE + b"\n"], silence_timeout=3)
        with patch("ecg_usb_stream.time.monotonic", side_effect=[0, 0, 0, 0, 4]):
            stream.identify()
            with self.assertRaisesRegex(USBStreamError, "No ECG samples"):
                stream.read_samples()

    def test_identity_timeout_is_reported(self):
        stream, _ = self.make_stream([], identity_timeout=1)
        with patch("ecg_usb_stream.time.monotonic", side_effect=[0, 0, 0, 2]):
            with self.assertRaisesRegex(USBStreamError, "did not respond"):
                stream.identify()


if __name__ == "__main__":
    unittest.main()
