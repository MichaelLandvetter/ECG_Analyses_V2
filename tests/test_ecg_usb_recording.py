"""Tests for recording real USB samples without losing device timing."""

import tempfile
import unittest
from pathlib import Path

from ecg_acquisition import load_ecg_signal_from_file
from ecg_usb_recording import USBRecording
from tools.usb_ecg_stream_utils import StreamSample


class TestUSBRecording(unittest.TestCase):
    def test_recording_is_loadable_and_preserves_device_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ecg.csv"
            recording = USBRecording(path)
            recording.append(StreamSample(100, 4, 12000))
            recording.append(StreamSample(104, 5, 12010))
            recording.close()
            self.assertEqual(load_ecg_signal_from_file(str(path)).tolist(), [12000, 12010])
            self.assertIn("12010,104,5", path.read_text())
            self.assertEqual(recording.discontinuities, 0)

    def test_sample_loss_and_timing_gaps_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            recording = USBRecording(Path(directory) / "ecg.csv")
            recording.append(StreamSample(100, 1, 10))
            recording.append(StreamSample(108, 3, 11))
            recording.append(StreamSample(200, 4, 12))
            recording.close()
            self.assertEqual(recording.discontinuities, 2)

    def test_pause_resume_excludes_intentional_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ecg.csv"
            recording = USBRecording(path)
            recording.append(StreamSample(100, 1, 10))
            recording.resume()
            recording.append(StreamSample(1000, 100, 11))
            recording.close()
            self.assertEqual(recording.sample_count, 2)
            self.assertEqual(recording.discontinuities, 0)
            self.assertIn("# acquisition resumed after pause", path.read_text())
            self.assertEqual(load_ecg_signal_from_file(str(path)).tolist(), [10, 11])

    def test_counter_rollover_and_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ecg.csv"
            recording = USBRecording(path)
            recording.append(StreamSample(2**32 - 4, 2**32 - 1, 10))
            recording.append(StreamSample(0, 0, 11))
            recording.close()
            self.assertEqual(recording.discontinuities, 0)
            with self.assertRaises(FileExistsError):
                USBRecording(path)


if __name__ == "__main__":
    unittest.main()
