"""Playback tests use a simulated robot and clock; no hardware is contacted."""

from __future__ import annotations

import contextlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import yaml
from reply import (
    CancelledError,
    Episode,
    RosRobot,
    align,
    checked_publish,
    config,
    drivers,
    play,
    play_episode,
    prepare,
    running_driver,
    save_prepared,
    select_episode,
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


class Keys:
    def __init__(self, clock: Clock, events: list[tuple[float, str]] | None = None) -> None:
        self.clock = clock
        self.events = list(events or [])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, timeout: float = 0) -> list[str]:
        self.clock.now += max(0, timeout)
        if self.events and self.events[0][0] <= self.clock.now:
            return [self.events.pop(0)[1]]
        return []


class Robot:
    def __init__(self, clock: Clock, follow: bool = True) -> None:
        self.clock = clock
        self.follow = follow
        self.position = np.zeros(14)
        self.commands: list[tuple[float, np.ndarray, np.ndarray | None]] = []
        self.closed = False
        self.held = False

    def current(self) -> np.ndarray:
        return self.position.copy()

    def check_publishers(self) -> None:
        pass

    def publish(self, joints: np.ndarray, grippers: np.ndarray | None = None) -> None:
        self.commands.append((self.clock.now, joints.copy(), None if grippers is None else grippers.copy()))
        if self.follow:
            self.position = joints.copy()

    def hold(self) -> None:
        self.held = True

    def close(self) -> None:
        self.closed = True


class PlaybackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = config(Path(__file__).with_name("config.yaml"))
        self.cfg["dataset"]["speed_multiplier"] = 1.0
        self.cfg["runtime"].update(start_slave_arms=True, start_bridge=True, start_grippers=True)
        self.cfg["alignment"]["min_duration_s"] = 0.1
        self.cfg["alignment"]["settle_timeout_s"] = 0.2
        self.clock = Clock()
        self.robot = Robot(self.clock)

    def data(self) -> tuple[np.ndarray, np.ndarray]:
        state = np.zeros((4, 16), dtype=np.float32)
        state[:, :7] = np.arange(4)[:, None] * 0.001
        state[:, 8:15] = -np.arange(4)[:, None] * 0.001
        state[:, [7, 15]] = 200  # feedback is not a gripper command
        action = state.copy()
        action[:, :7] += 0.02
        action[:, [7, 15]] = [0, 1]
        return state, action

    def test_raw_loader_episode_order_units_and_gripper_command(self) -> None:
        state, action = self.data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.yaml").write_text(yaml.safe_dump({"dataset": {"fps": 50}}))
            for label in ("000000", "000002", "000003.partial"):
                with sqlite3.connect(root / f"episode_{label}.sqlite3") as db:
                    db.execute("CREATE TABLE frames (idx INTEGER, tick REAL, state BLOB, action BLOB)")
                    db.executemany(
                        "INSERT INTO frames VALUES (?,?,?,?)",
                        [
                            (i, i / 50, s.tobytes(), a.tobytes())
                            for i, (s, a) in enumerate(zip(state, action, strict=True))
                        ],
                    )
            self.cfg["dataset"]["path"] = str(root)
            prepare(self.cfg, root / "prepared")
            files = sorted((root / "prepared").glob("*.npz"))
            self.assertEqual(len(files), 2)
            episode = Episode.load(files[1])
            self.assertEqual(episode.label, "episode_000002")
            self.assertEqual(episode.fps, 50)
            np.testing.assert_allclose(episode.joints[1, :7], np.rad2deg(0.001))
            np.testing.assert_array_equal(episode.grippers, [[0, 1]] * 4)

    def test_real_lerobot_parquet_loader(self) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        state, action = self.data()
        names = [
            f"{side}_joint_{i + 1}" if i < 7 else f"{side}_gripper"
            for side in ("left", "right")
            for i in range(8)
        ]
        features = {
            name: {"dtype": "float32", "shape": (16,), "names": names}
            for name in ("observation.state", "action")
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = LeRobotDataset.create(
                repo_id="local/reply_test", root=root / "lerobot", fps=50, features=features, use_videos=False
            )
            for _ in range(2):
                for s, a in zip(state, action, strict=True):
                    dataset.add_frame({"observation.state": s, "action": a, "task": "test"})
                dataset.save_episode(parallel_encoding=False)
            dataset.finalize()
            self.cfg["dataset"]["path"] = str(root)
            prepare(self.cfg, root / "prepared")
            self.assertEqual(len(list((root / "prepared").glob("*.npz"))), 2)
            episode = Episode.load(root / "prepared" / "episode_000000.npz")
            np.testing.assert_allclose(episode.joints[1, :7], np.rad2deg(0.001))
            # Both job-directory and lerobot-directory paths are supported.
            self.cfg["dataset"]["path"] = str(root / "lerobot")
            prepare(self.cfg, root / "prepared_again")

    def test_invalid_data_rejected_before_any_commands(self) -> None:
        state, action = self.data()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            bad = state.copy()
            bad[1, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "NaN"):
                save_prepared(out, 0, "bad", bad, action, 50, self.cfg)
            bad = state.copy()
            bad[1, 0] = 2
            with self.assertRaisesRegex(ValueError, "速度"):
                save_prepared(out, 0, "bad", bad, action, 50, self.cfg)
            action[0, 7] = 255
            with self.assertRaisesRegex(ValueError, "夹爪"):
                save_prepared(out, 0, "bad", state, action, 50, self.cfg)
            self.assertFalse(list(out.glob("*.npz")))

    def test_alignment_is_smooth_and_speed_bounded(self) -> None:
        with patch("reply.time.monotonic", self.clock.monotonic):
            align(self.robot, np.full(14, 10.0), self.cfg, Keys(self.clock))
        positions = np.asarray([command[1] for command in self.robot.commands])
        speed = np.abs(np.diff(positions, axis=0)) * self.cfg["playback"]["command_hz"]
        self.assertLessEqual(float(speed.max()), 10.001)
        np.testing.assert_array_equal(positions[-1], np.full(14, 10.0))
        self.assertTrue(all(command[2] is None for command in self.robot.commands))

    def test_cancel_during_alignment_and_playback(self) -> None:
        episode = Episode(np.linspace(np.zeros(14), np.ones(14), 51), np.zeros((51, 2)), 50, "test")
        for at in (0.04, 0.3):
            clock = Clock()
            robot = Robot(clock)
            with patch("reply.time.monotonic", clock.monotonic), self.assertRaises(CancelledError):
                play_episode(robot, episode, self.cfg, Keys(clock, [(at, "f")]))
            self.assertLess(len(robot.commands), 61)
            self.assertLess(robot.current()[0], 1)

    def test_alignment_must_reach_actual_first_frame(self) -> None:
        robot = Robot(self.clock, follow=False)
        with (
            patch("reply.time.monotonic", self.clock.monotonic),
            self.assertRaisesRegex(RuntimeError, "对齐超时"),
        ):
            align(robot, np.full(14, 5), self.cfg, Keys(self.clock))

    def test_episode_sequence_requires_an_arrow_each_time(self) -> None:
        state, action = self.data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(2):
                save_prepared(root, i, f"episode_{i}", state, action, 50, self.cfg)
            keys = Keys(self.clock, [(0.1, "right"), (1.0, "right")])
            with (
                patch("reply.time.monotonic", self.clock.monotonic),
                patch("reply.Keyboard", return_value=keys),
                patch("reply.drivers", return_value=contextlib.nullcontext(lambda: None)),
                patch("reply.RosRobot", return_value=self.robot),
                patch("reply.input", return_value="0"),
            ):
                play(self.cfg, root)
            with_grip = [command for command in self.robot.commands if command[2] is not None]
            self.assertEqual(len(with_grip), 8)
            self.assertTrue(self.robot.closed)
            self.assertTrue(self.robot.held)
            self.assertGreaterEqual(with_grip[4][0], 1.1)

    def test_episode_selection_uses_recorded_number_and_rejects_invalid_input(self) -> None:
        state, action = self.data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, number in enumerate((0, 2, 12)):
                save_prepared(root, index, f"episode_{number:06d}", state, action, 50, self.cfg)
            paths = sorted(root.glob("*.npz"))
            with patch("reply.input", side_effect=["1", "-1", "abc", "0012"]):
                self.assertEqual(select_episode(paths), 2)
            with patch("reply.input", return_value=""):
                self.assertEqual(select_episode(paths), 0)
            with patch("reply.input", return_value="f"), self.assertRaises(CancelledError):
                select_episode(paths)

    def test_play_from_selected_episode_skips_earlier_segments(self) -> None:
        state, action = self.data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                save_prepared(root, index, f"episode_{index:06d}", state, action, 50, self.cfg)
            with (
                patch("reply.input", return_value="2"),
                patch("reply.time.monotonic", self.clock.monotonic),
                patch("reply.Keyboard", return_value=Keys(self.clock, [(0.1, "right")])),
                patch("reply.drivers", return_value=contextlib.nullcontext(lambda: None)),
                patch("reply.RosRobot", return_value=self.robot),
                patch("reply.play_episode") as replay,
            ):
                play(self.cfg, root)
            replay.assert_called_once()
            self.assertEqual(replay.call_args.args[1].label, "episode_000002")

    def test_no_hold_command_before_first_arrow(self) -> None:
        # A user exiting from the initial prompt must not move the robot.
        robot = RosRobot.__new__(RosRobot)
        robot.commanded = False
        robot.hold()

    def test_arm_commands_continue_while_unchanged_grippers_are_not_repeated(self) -> None:
        robot = RosRobot.__new__(RosRobot)
        robot.cfg = self.cfg
        robot.node = Mock()
        robot.last_grippers = {}
        robot.JointAngle = lambda: SimpleNamespace(header=SimpleNamespace())
        robot.GripperStatus = lambda: SimpleNamespace(header=SimpleNamespace())
        robot.publishers = {side: Mock() for side in ("left", "right")}
        robot.grippers = {side: Mock() for side in ("left", "right")}
        for index in range(100):
            robot.publish(np.full(14, index / 100), np.array([0.0, 1.0]))
        for side in ("left", "right"):
            self.assertEqual(robot.publishers[side].publish.call_count, 100)
            self.assertEqual(robot.grippers[side].publish.call_count, 1)
        robot.publish(np.ones(14), np.array([1.0, 1.0]))
        self.assertEqual(robot.grippers["left"].publish.call_count, 2)
        self.assertEqual(robot.grippers["right"].publish.call_count, 1)
        self.assertEqual(robot.grippers["left"].publish.call_args.args[0].status, 1.0)

    def test_tracking_error_identifies_joint_and_does_not_publish(self) -> None:
        target = np.zeros(14)
        target[9] = 20
        with self.assertRaisesRegex(RuntimeError, "右臂第 3 关节.*反馈 0.00.*目标 20.00"):
            checked_publish(self.robot, target, None, self.cfg)
        self.assertEqual(self.robot.commands, [])

    def test_driver_launch_arguments_cleanup_and_failure_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.cfg["runtime"]["driver_log_dir"] = directory
            children = [Mock(pid=90000 + i) for i in range(5)]
            for process in children:
                process.poll.return_value = None
            with (
                patch("reply.running_driver", return_value=False),
                patch("reply.shutil.which", return_value="/mock/executable"),
                patch("reply.Path.exists", return_value=True),
                patch("reply.subprocess.Popen", side_effect=children) as launch,
                patch("reply.os.killpg") as kill,
            ):
                with drivers(self.cfg) as check:
                    check()
                    commands = [call.args[0] for call in launch.call_args_list]
                    self.assertEqual(commands[0], ["zenoh-bridge-ros2dds"])
                    self.assertIn("192.168.11.60", commands[1])
                    self.assertIn("192.168.11.61", commands[2])
                    self.assertEqual(commands[3][-4:], ["-s", "left", "-p", "/dev/ttyUSB0"])
                    self.assertEqual(commands[4][-4:], ["-s", "right", "-p", "/dev/ttyUSB1"])
                    children[2].poll.return_value = 1
                    with self.assertRaisesRegex(RuntimeError, "slave_right.*日志"):
                        check()
                self.assertEqual(kill.call_count, 4)
                self.assertTrue(all(process.wait.called for process in children))

    def test_existing_drivers_are_not_launched_or_stopped(self) -> None:
        with (
            patch("reply.running_driver", return_value=True),
            patch("reply.subprocess.Popen") as launch,
            patch("reply.os.killpg") as kill,
        ):
            with drivers(self.cfg) as check:
                check()
            launch.assert_not_called()
            kill.assert_not_called()

    def test_driver_reuse_requires_matching_side_and_endpoint(self) -> None:
        entry = Mock()
        argv = [
            "/opt/arm-servo-adapter/bin/slave_arm",
            "-a",
            "192.168.11.60",
            "-S",
            "left",
            "-P",
            "8080",
            "-r",
            "10001",
        ]
        entry.read_bytes.return_value = ("\0".join(argv) + "\0").encode()
        with patch("reply.Path.glob", return_value=[entry]):
            self.assertTrue(running_driver(argv, "-S"))
            self.assertFalse(running_driver([part if part != "left" else "right" for part in argv], "-S"))
            mismatch = [part if part != "192.168.11.60" else "192.168.11.61" for part in argv]
            with self.assertRaisesRegex(RuntimeError, "YAML 不一致"):
                running_driver(mismatch, "-S")


if __name__ == "__main__":
    unittest.main()
