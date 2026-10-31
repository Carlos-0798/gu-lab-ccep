"""Numerical and file-handling checks using synthetic inputs only."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import artifact_ccep as artifact
import filter_ccep as base


class ProcessingTests(unittest.TestCase):
    def test_commands_use_stream_offsets_and_group_biphasic_samples(self):
        command = np.zeros(32000)
        indices = np.arange(1000, 31000, 1000)
        for index in indices:
            command[index : index + 2] = 200
            command[index + 3 : index + 5] = -200
        block = SimpleNamespace(
            streams={
                "eS1r": SimpleNamespace(data=command, fs=1000, start_time=2.0),
                "LFP1": SimpleNamespace(fs=2000, start_time=1.5),
            }
        )
        onsets, centers = base.command_centers(block)
        np.testing.assert_array_equal(onsets, np.arange(3.0, 33.0))
        np.testing.assert_array_equal(centers, 3000 + np.arange(30) * 2000)

    def test_unexpected_event_count_stops_processing(self):
        for data in (np.zeros(1000), np.r_[200.0, np.zeros(999)]):
            block = SimpleNamespace(
                streams={
                    "eS1r": SimpleNamespace(data=data, fs=1000, start_time=0.0),
                    "LFP1": SimpleNamespace(fs=1000, start_time=0.0),
                }
            )
            with self.assertRaises(ValueError):
                base.command_centers(block)

    def test_nonfinite_command_stops_processing(self):
        block = SimpleNamespace(
            streams={
                "eS1r": SimpleNamespace(data=[np.nan], fs=1000, start_time=0.0),
                "LFP1": SimpleNamespace(fs=1000, start_time=0.0),
            }
        )
        with self.assertRaises(ValueError):
            base.command_centers(block)

    def test_incomplete_epoch_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            base.extract(np.zeros((2, 100)), np.array([1, 50]), np.arange(-5, 10))

    def test_per_trial_baseline_preserves_response_variation(self):
        times = np.array([-0.25, -0.10, -0.05, 0.0, 0.10])
        trials = np.array(
            [[[4.0, 6.0, 100.0, 8.0, 10.0]], [[10.0, 14.0, 200.0, 19.0, 21.0]]]
        )
        corrected, offsets = base.baseline_correct(trials, times)
        np.testing.assert_array_equal(offsets[:, 0, 0], [5.0, 12.0])
        np.testing.assert_array_equal(corrected[:, 0, 3], [3.0, 7.0])
        self.assertEqual(corrected[:, 0, 3].mean(), 5.0)
        self.assertEqual(corrected[:, 0, 3].std(ddof=0), 2.0)

    def test_notch_suppresses_60hz_and_preserves_10hz(self):
        fs = 1000.0
        time = np.arange(30000) / fs
        values = np.sin(2 * np.pi * 10 * time) + 0.5 * np.sin(2 * np.pi * 60 * time)
        filtered = base.apply_filter(values, fs, "notch60")
        interior = (time > 5) & (time < 25)
        self.assertGreater(
            base.sine_amplitude(time[interior], filtered[interior], 10), 0.99
        )
        self.assertLess(
            base.sine_amplitude(time[interior], filtered[interior], 60), 0.001
        )
        np.testing.assert_array_equal(
            base.apply_filter(values, fs, "no_extra_filter"), values
        )

    def test_filter_does_not_mix_channels(self):
        time = np.arange(10000) / 1000.0
        values = np.vstack([np.sin(2 * np.pi * 10 * time), np.zeros(time.size)])
        for mode in base.MODES:
            result = base.apply_filter(values, 1000.0, mode)
            np.testing.assert_array_equal(result[1], np.zeros(time.size))

    def test_copy_changes_only_selected_rows_and_window(self):
        values = np.arange(1200.0).reshape(4, 300)
        original = values.copy()
        centers = np.array([80, 220])
        copied = artifact.copy_pre_event(values, centers, [1, 2], 6)
        allowed = np.zeros_like(values, dtype=bool)
        for center in centers:
            allowed[1:3, center : center + 6] = True
            np.testing.assert_array_equal(
                copied[1:3, center : center + 6], original[1:3, center - 6 : center]
            )
        np.testing.assert_array_equal(values, original)
        np.testing.assert_array_equal(copied[~allowed], original[~allowed])
        np.testing.assert_array_equal(
            copied[:, centers + 20], original[:, centers + 20]
        )

    def test_copy_requires_context(self):
        for center, count in ((2, 6), (98, 6), (50, 0)):
            with self.assertRaises(ValueError):
                artifact.copy_pre_event(np.zeros((2, 100)), [center], [0], count)

    def test_gray_band_uses_population_sd_and_marks_voltage_zoom(self):
        fig, ax = plt.subplots()
        try:
            base.draw_axis(
                ax,
                np.array([0.0, 0.1]),
                np.array([[0.001, 0.001], [0.003, 0.003]]),
                "",
                (-0.1, 0.2),
            )
            vertices = ax.collections[0].get_paths()[0].vertices[:, 1]
            self.assertAlmostEqual(vertices.min(), 0.001)
            self.assertAlmostEqual(vertices.max(), 0.003)
            self.assertIn("Voltage zoom", ax.texts[0].get_text())
        finally:
            plt.close(fig)

    def test_single_mouse_paths_and_output_separation(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(base, "BLOCKS", {}),
            patch.object(base, "OUT", Path(temp)),
        ):
            root = Path(temp)
            args = SimpleNamespace(
                mouse_1_block=root / "first block",
                mouse_2_block=root / "missing",
                output_dir=root / "results",
            )
            base.configure_paths(args, [1])
            self.assertEqual(set(base.BLOCKS), {1})
            args.output_dir = args.mouse_1_block / "results"
            with self.assertRaises(ValueError):
                base.configure_paths(args, [1])


class CacheTests(unittest.TestCase):
    def test_filter_cache_detects_changed_result_and_missing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            block = root / "block"
            block.mkdir()
            (block / "record.tsq").write_bytes(b"index")
            (block / "record.tev").write_bytes(b"samples")
            out = root / "results"
            (out / "mouse_1").mkdir(parents=True)
            cache = out / "mouse_1/filtered_epochs.npz"
            cache.write_bytes(b"original result")
            with (
                patch.object(base, "BLOCKS", {1: block}),
                patch.object(base, "OUT", out),
                patch.object(base, "build") as build,
            ):
                report = {
                    "configuration": base.CONFIG,
                    "software": base.runtime_signature(),
                    "protected_sha256": base.protected_hashes(),
                    "result_sha256": {"1": base.digest(cache)},
                }
                base.write_json(out / "processing_summary.json", report)
                base.ensure_results()
                build.assert_not_called()
                cache.write_bytes(b"changed result")
                with self.assertRaises(ValueError):
                    base.ensure_results()
                cache.unlink()
                base.ensure_results()
                build.assert_called_once()

    def test_artifact_cache_detects_input_and_configuration_changes(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(base, "BLOCKS", {1: Path(temp)}),
            patch.object(artifact, "OUT", Path(temp)),
            patch.object(base, "ensure_results"),
            patch.object(artifact, "protected", return_value={"input": "old"}),
            patch.object(artifact, "build") as build,
        ):
            path = artifact.cache_path(1)
            path.parent.mkdir()
            path.write_bytes(b"result")
            report = {
                "configuration": artifact.CONFIG,
                "software": artifact.processing_signature(),
                "protected_sha256_after": {"input": "old"},
                "result_sha256": {"1": base.digest(path)},
            }
            report_path = Path(temp) / "processing_summary.json"
            base.write_json(report_path, report)
            artifact.ensure_results()
            build.assert_not_called()
            with patch.object(artifact, "protected", return_value={"input": "changed"}):
                with self.assertRaises(ValueError):
                    artifact.ensure_results()
            saved = json.loads(report_path.read_text())
            saved["configuration"]["copy_window_s"] = [0, 0.01]
            base.write_json(report_path, saved)
            with self.assertRaises(ValueError):
                artifact.ensure_results()


if __name__ == "__main__":
    unittest.main()
