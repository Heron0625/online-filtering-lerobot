"""Hardware-free regression tests; run using the documented unittest command."""

from __future__ import annotations

import contextlib
import json
import queue
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from record import (
    EpisodeWriter,
    Inputs,
    Keyboard,
    SamplingStats,
    convert,
    image_bgr,
    load_config,
    numeric_value,
    record,
)


class RecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config(Path(__file__).with_name("config.yaml"))
        # Keep the existing 50Hz fixtures, independent of deployment's chosen FPS.
        self.cfg["dataset"]["fps"] = 50
        self.cfg["recording"]["frequency"]["allow_faster"] = False
        for stream in self.cfg["streams"].values():
            stream["fps"] = 50
        for camera in self.cfg["cameras"].values():
            camera.update(width=32, height=24)
        self.inputs = Inputs(self.cfg)
        self.image = SimpleNamespace(
            encoding="rgb8",
            width=32,
            height=24,
            step=96,
            data=np.full((24, 32, 3), [255, 10, 0], dtype=np.uint8).tobytes(),
        )

    def fill(self) -> dict:
        for index, (name, stream) in enumerate(self.cfg["streams"].items()):
            self.inputs.put(name, np.full(stream["size"], index, dtype=np.float32), index)
        for name in self.cfg["cameras"]:
            self.inputs.put(name, self.image, 123)
        return self.inputs.snapshot(time.monotonic())

    def test_action_is_master_command_and_order_is_left_then_right(self) -> None:
        snapshot = self.fill()
        np.testing.assert_array_equal(self.inputs.vector(snapshot, "state"), [0] * 7 + [1] + [2] * 7 + [3])
        np.testing.assert_array_equal(self.inputs.vector(snapshot, "action"), [4] * 7 + [5] + [6] * 7 + [7])

    def test_missing_stale_future_and_skew_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "缺失"):
            self.inputs.snapshot(time.monotonic())
        self.fill()
        with self.assertRaisesRegex(RuntimeError, "超时"):
            self.inputs.snapshot(time.monotonic() + 1)
        with self.assertRaisesRegex(RuntimeError, "缺失"):
            self.inputs.snapshot(time.monotonic() - 1)
        now = time.monotonic()
        with self.inputs.lock:
            from record import Sample

            self.inputs.buffers["state_left_joint"].clear()
            self.inputs.buffers["state_left_joint"].append(Sample(now - 0.09, 0, np.zeros(7)))
        with self.assertRaisesRegex(RuntimeError, "跨度"):
            self.inputs.snapshot(now)

    def test_units_padding_and_nonfinite_values(self) -> None:
        stream = self.cfg["streams"]["state_left_joint"]
        value = numeric_value(SimpleNamespace(angle=[180] * 8), stream)
        np.testing.assert_allclose(value, [np.pi] * 7)
        with self.assertRaises(ValueError):
            numeric_value(SimpleNamespace(angle=[float("nan")] * 8), stream)
        with self.assertRaises(ValueError):
            numeric_value(SimpleNamespace(angle=[1] * 6), stream)
        image = SimpleNamespace(encoding="rgb8", width=1, height=1, step=4, data=bytes([255, 0, 0, 99]))
        np.testing.assert_array_equal(image_bgr(image, {"width": 1, "height": 1}), [[[0, 0, 255]]])

    def test_accept_discard_and_writer_error(self) -> None:
        snapshot = self.fill()
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            writer = EpisodeWriter(session, 0, self.cfg)
            writer.submit(1.0, self.inputs, snapshot)
            writer.submit(1.02, self.inputs, snapshot)
            accepted = writer.finish(True)
            assert accepted is not None
            self.assertEqual(accepted.name, "episode_000000.sqlite3")
            with sqlite3.connect(accepted) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM frames").fetchone()[0], 2)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM images").fetchone()[0], 8)
                stamps = json.loads(db.execute("SELECT stamps FROM frames LIMIT 1").fetchone()[0])
                self.assertEqual(stamps["head"]["header_ns"], 123)
            writer = EpisodeWriter(session, 1, self.cfg)
            writer.submit(1.0, self.inputs, snapshot)
            self.assertIsNone(writer.finish(False))
            self.assertFalse(writer.path.exists())
            writer = EpisodeWriter(session, 2, self.cfg)
            writer.submit(1.0, self.inputs, snapshot)
            with self.assertRaisesRegex(RuntimeError, "太短"):
                writer.finish(True)
            snapshot["head"].value.encoding = "invalid"
            writer = EpisodeWriter(session, 3, self.cfg)
            writer.submit(1.0, self.inputs, snapshot)
            with self.assertRaisesRegex(RuntimeError, "暂存失败"):
                writer.finish(True)
            self.assertTrue(writer.path.exists())
            self.assertFalse((session / "episode_000003.sqlite3").exists())

    def test_keyboard_handles_fragmented_right_arrow(self) -> None:
        keyboard = Keyboard()
        keyboard.pending = ""
        with patch("record.select.select", return_value=([1], [], [])), patch("record.sys.stdin") as stdin:
            stdin.fileno.return_value = 0
            with patch("record.os.read", side_effect=[b"\x1b[", b"C", b"nfs"]):
                self.assertEqual(keyboard.read(0), [])
                self.assertEqual(keyboard.read(0), ["right"])
                self.assertEqual(keyboard.read(0), ["n", "f", "s"])

    def test_full_session_keyboard_workflow(self) -> None:
        events = iter([["right"], ["s"], [], [], ["n"], ["right"], [], ["f"], ["s"]])
        keyboard = Mock()
        keyboard.__enter__ = Mock(return_value=keyboard)
        keyboard.__exit__ = Mock(return_value=None)

        def read(timeout: float) -> list[str]:
            time.sleep(timeout)
            self.fill()
            return next(events)

        keyboard.read.side_effect = read
        with tempfile.TemporaryDirectory() as directory:
            self.cfg["dataset"]["save_dir"] = directory
            self.fill()
            with (
                patch("record.Inputs", return_value=self.inputs),
                patch("record.Keyboard", return_value=keyboard),
                patch("record.hardware", return_value=contextlib.nullcontext([])),
                patch("record.connect_ros", return_value=(Mock(), Mock(), Mock())),
                patch.object(self.inputs, "check_rates", return_value={}),
                patch.object(self.inputs, "check_new"),
                patch.object(self.inputs, "match", side_effect=RuntimeError("偶发缺帧")),
            ):
                session = record(self.cfg)
            self.assertIsNotNone(session)
            assert session is not None
            self.assertEqual(len(list(session.glob("episode_*.sqlite3"))), 1)
            self.assertTrue((session / "episode_000000.sqlite3").exists())
            self.assertFalse((session / "lerobot").exists())  # conversion starts only after record returns

    def feed_rates(self, overrides: dict[str, float] | None = None, skipped: str | None = None) -> None:
        specs = {**self.cfg["streams"], **self.cfg["cameras"]}
        for name, spec in specs.items():
            fps = (overrides or {}).get(name, spec["fps"])
            for index in range(int(1.2 * fps) + 1):
                if name == skipped and index == int(0.6 * fps):
                    continue
                with patch("record.time.monotonic", return_value=index / fps):
                    self.inputs.put(name, None, index + 1)

    def test_rates_require_warmup_and_accept_configured_source_rates(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "预热"):
            self.inputs.check_rates(0)
        self.feed_rates()
        rates = self.inputs.check_rates(1.2)
        self.assertAlmostEqual(rates["head"], 60)
        self.assertAlmostEqual(rates["state_left_joint"], 50)
        self.assertEqual(len(rates), 12)

    def test_uniform_60hz_profile_passes_only_with_60hz_inputs(self) -> None:
        self.cfg["dataset"]["fps"] = 60
        for stream in self.cfg["streams"].values():
            stream["fps"] = 60
        self.inputs = Inputs(self.cfg)
        self.feed_rates()
        rates = self.inputs.check_rates(1.2)
        for rate in rates.values():
            self.assertAlmostEqual(rate, 60)
        self.inputs = Inputs(self.cfg)
        self.feed_rates({"state_left_joint": 50})
        with self.assertRaisesRegex(RuntimeError, "频率异常.*state_left_joint"):
            self.inputs.check_rates(1.2)

    def test_faster_sources_can_be_downsampled_but_slow_sources_still_fail(self) -> None:
        self.cfg["dataset"]["fps"] = 60
        self.cfg["recording"]["frequency"]["allow_faster"] = True
        for stream in self.cfg["streams"].values():
            stream["fps"] = 60
        self.inputs = Inputs(self.cfg)
        self.feed_rates({"action_left_joint": 102, "action_right_joint": 100})
        rates = self.inputs.check_rates(1.2)
        self.assertAlmostEqual(rates["action_left_joint"], 102)
        self.assertAlmostEqual(rates["action_right_joint"], 100)
        self.inputs = Inputs(self.cfg)
        self.feed_rates({"state_left_joint": 50, "action_left_joint": 102})
        with self.assertRaisesRegex(RuntimeError, "频率异常.*state_left_joint"):
            self.inputs.check_rates(1.2)

    def test_low_high_rates_and_single_missing_message_are_rejected(self) -> None:
        for changes, skipped, reason in (
            ({"head": 30}, None, "断帧.*head"),
            ({"head": 55}, None, "频率异常.*head"),
            ({"state_left_joint": 100}, None, "频率异常.*state_left_joint"),
            ({}, "head", "断帧.*head"),
            ({}, "action_right_gripper", "断帧.*action_right_gripper"),
        ):
            self.inputs = Inputs(self.cfg)
            self.feed_rates(changes, skipped)
            with self.subTest(changes=changes, skipped=skipped), self.assertRaisesRegex(RuntimeError, reason):
                self.inputs.check_rates(1.2)

    def test_short_silence_fails_even_before_old_freshness_timeout(self) -> None:
        self.feed_rates()
        with self.assertRaisesRegex(RuntimeError, "断帧"):
            self.inputs.check_rates(1.24)  # only 40ms old, previously allowed by max_age_ms=100

    def test_reusing_arrival_or_nonzero_header_is_rejected(self) -> None:
        from record import Sample

        first = {"head": Sample(1.0, 100, self.image)}
        self.inputs.check_new(first, None)
        with self.assertRaisesRegex(RuntimeError, "重复帧.*head"):
            self.inputs.check_new(first, first)
        with self.assertRaisesRegex(RuntimeError, "重复帧.*head"):
            self.inputs.check_new({"head": Sample(1.02, 100, self.image)}, first)
        # Static scene/joint values are valid when a new source timestamp arrives.
        self.inputs.check_new({"head": Sample(1.02, 101, self.image)}, first)
        self.inputs.check_new({"head": Sample(1.02, 0, self.image)}, {"head": Sample(1.0, 0, self.image)})

    def test_stale_input_discards_episode_in_recording_loop(self) -> None:
        keyboard = Mock()
        keyboard.__enter__ = Mock(return_value=keyboard)
        keyboard.__exit__ = Mock(return_value=None)
        events = iter([["right"], [], [], [], [], ["s"]])

        def read(timeout: float) -> list[str]:
            time.sleep(timeout)
            return next(events)

        keyboard.read.side_effect = read
        self.fill()
        with tempfile.TemporaryDirectory() as directory:
            self.cfg["dataset"]["save_dir"] = directory
            with (
                patch("record.Inputs", return_value=self.inputs),
                patch.object(self.inputs, "check_rates", return_value={}),
                patch.object(self.inputs, "resample", side_effect=RuntimeError("head 数据过期")),
                patch("record.Keyboard", return_value=keyboard),
                patch("record.hardware", return_value=contextlib.nullcontext([])),
                patch("record.connect_ros", return_value=(Mock(), Mock(), Mock())),
            ):
                self.assertIsNone(record(self.cfg))
            self.assertEqual(list(Path(directory).rglob("*.sqlite3")), [])

    def put_at(self, name: str, arrival: float, header: int) -> None:
        value = self.image if name in self.cfg["cameras"] else np.zeros(self.cfg["streams"][name]["size"])
        with patch("record.time.monotonic", return_value=arrival):
            self.inputs.put(name, value, header)

    def test_sync_chooses_nearest_on_either_side_instead_of_latest(self) -> None:
        for name in self.inputs.buffers:
            self.put_at(name, 0.991, 1)
            self.put_at(name, 1.002, 2)
            self.put_at(name, 1.014, 3)
        match = self.inputs.match(1.0, None, 1.025)
        self.assertTrue(all(sample.header_ns == 2 for sample in match.values()))
        self.assertTrue(all(sample.arrival > 1.0 for sample in match.values()))
        self.assertTrue(all(sample.header_ns == 1 for sample in self.inputs.snapshot(1.0).values()))

    def test_sync_finds_alternative_group_with_bounded_skew(self) -> None:
        self.cfg["recording"]["sync"]["max_span_ms"] = 8
        for name in self.inputs.buffers:
            self.put_at(name, 1.005, 1)
        self.inputs.buffers["head"].clear()
        self.put_at("head", 0.994, 2)  # individually nearest, but >8ms from the others
        self.put_at("head", 1.008, 3)  # a valid alternative
        match = self.inputs.match(1.0, None, 1.025)
        self.assertEqual(match["head"].header_ns, 3)
        times = [sample.arrival for sample in match.values()]
        self.assertLessEqual(max(times) - min(times), 0.008)

    def test_sync_rejects_missing_reused_and_incoherent_inputs(self) -> None:
        for name in self.inputs.buffers:
            self.put_at(name, 1.0, 100)
        first = self.inputs.match(1.0, None, 1.025)
        with self.assertRaisesRegex(RuntimeError, "同步配对失败"):
            self.inputs.match(1.02, first, 1.045)
        for name in self.inputs.buffers:
            self.put_at(name, 1.02, 99)  # receipt is new but source timestamp rolled back
        with self.assertRaisesRegex(RuntimeError, "同步配对失败"):
            self.inputs.match(1.02, first, 1.045)
        self.inputs = Inputs(self.cfg)
        for name in self.inputs.buffers:
            self.put_at(name, 1.014 if name == "head" else 0.986, 1)
        with self.assertRaisesRegex(RuntimeError, "组内最大时间差"):
            self.inputs.match(1.0, None, 1.025)

    def test_sync_50hz_grid_uses_unique_messages_from_50_60_100hz_sources(self) -> None:
        self.cfg["recording"]["frequency"]["allow_faster"] = True
        for name in self.inputs.buffers:
            rate = 60 if name in self.cfg["cameras"] else 100 if name.startswith("action_") else 50
            for index in range(rate * 2):
                self.put_at(name, index / rate + 0.001, index + 1)
        # The bounded buffers retain only their latest messages; check a common
        # recent interval. All output ticks remain exactly 20ms apart.
        previous = None
        for index in range(10):
            tick = 1.7 + index / 50
            group = self.inputs.match(tick, previous, tick + 0.025)
            if previous:
                self.inputs.check_new(group, previous)
            for sample in group.values():
                self.assertLessEqual(abs(sample.arrival - tick), 0.015)
            previous = group

    def test_hold_last_preserves_other_stream_updates_and_rejects_stale(self) -> None:
        for name in self.inputs.buffers:
            self.put_at(name, 1.0, 1)
        previous = self.inputs.resample(1.0, None, 1.025)
        for name in self.inputs.buffers:
            if name != "head":
                self.put_at(name, 1.04, 2)
        group = self.inputs.resample(1.04, previous, 1.065)
        self.assertEqual(group["head"].arrival, 1.0)
        self.assertEqual(group["state_left_joint"].arrival, 1.04)
        # Repeated source stamps cannot reset freshness, even on fresh callbacks.
        self.put_at("head", 1.08, 1)
        with self.assertRaisesRegex(RuntimeError, "head 数据过期"):
            self.inputs.resample(1.08, group, 1.105)

    def test_repeat_warning_does_not_stop_recording(self) -> None:
        for name in self.inputs.buffers:
            self.put_at(name, 1.0, 1)
        group = self.inputs.resample(1.0, None, 1.025)
        stats = SamplingStats(self.cfg)
        stats.observe(group, None, 1.025)
        with patch("record.print") as output:
            for i in range(1, 4):
                held = self.inputs.resample(1 + i * 0.02, group, 1.025 + i * 0.02)
                stats.observe(held, group, 1.025 + i * 0.02)
                group = held
        self.assertEqual(stats.repeats["head"], 3)
        self.assertTrue(any("严重告警" in str(call) for call in output.call_args_list))
        self.assertEqual(stats.frames, 4)

    def test_queue_overflow_does_not_silently_drop_frames(self) -> None:
        snapshot = self.fill()
        with tempfile.TemporaryDirectory() as directory:
            writer = EpisodeWriter(Path(directory), 0, self.cfg)
            with (
                patch.object(writer.queue, "put_nowait", side_effect=queue.Full),
                self.assertRaisesRegex(RuntimeError, "队列已满"),
            ):
                writer.submit(1.0, self.inputs, snapshot)
            writer.finish(False)

    def test_real_lerobot_conversion(self) -> None:
        # Encode four real MP4s and read the resulting metadata/parquet via the local API.
        import yaml

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            (session / "config.yaml").write_text(yaml.safe_dump(self.cfg))
            snapshot = self.fill()
            for episode in range(2):
                writer = EpisodeWriter(session, episode, self.cfg)
                for i in range(4):
                    writer.submit(i / 50, self.inputs, snapshot)
                raw = writer.finish(True)
                assert raw is not None
                with sqlite3.connect(raw) as db:
                    stamps = [
                        json.loads(row[0]) for row in db.execute("SELECT stamps FROM frames ORDER BY idx")
                    ]
                self.assertFalse(stamps[0]["head"]["reused"])
                self.assertTrue(stamps[1]["head"]["reused"])
                self.assertIn("age_ms", stamps[1]["head"])
            destination = convert(session)
            dataset = LeRobotDataset(self.cfg["dataset"]["repo_id"], root=destination, video_backend="pyav")
            self.assertEqual(dataset.fps, 50)
            self.assertEqual(dataset.num_episodes, 2)
            self.assertEqual(dataset.num_frames, 8)
            self.assertEqual(tuple(dataset[0]["action"].shape), (16,))
            np.testing.assert_array_equal(dataset[0]["action"].numpy(), [4] * 7 + [5] + [6] * 7 + [7])
            image = dataset[0]["observation.images.head"].numpy()
            self.assertEqual(image.shape, (3, 24, 32))
            self.assertGreater(image[0].mean(), 0.8)  # red must stay RGB, not BGR
            self.assertLess(image[2].mean(), 0.1)
            self.assertEqual(len(list(session.glob("episode_*.sqlite3"))), 2)
            with self.assertRaises(FileExistsError):
                convert(session)


if __name__ == "__main__":
    unittest.main()
