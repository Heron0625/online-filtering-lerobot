"""No model downloads, ROS drivers or physical motion in these tests."""

from __future__ import annotations

import socket
import struct
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from common import decode_packet, encode_packet, load_config, validate_actions
from infer import CancelledError, Client, execute_chunk
from policy_worker import PolicyRunner, load_native_weights, receive_exact, serve
from train import training_args


class InferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config(Path(__file__).with_name("config.yaml"))
        # Fixed test timing also verifies compatibility with older 50Hz datasets.
        self.cfg["control"].update(action_hz=50, command_hz=50)
        self.state = np.zeros(16)
        self.actions = np.zeros((10, 16))
        self.actions[:, :7] = np.arange(1, 11)[:, None] * 0.001
        self.actions[:, 8:15] = -self.actions[:, :7]

    def test_actions_use_radians_and_reject_whole_unsafe_chunk(self) -> None:
        np.testing.assert_array_equal(validate_actions(self.actions, self.state, self.cfg), self.actions)
        for reason in ("nan", "gripper", "first", "speed", "shape"):
            bad = self.actions.copy()
            if reason == "nan":
                bad[9, 0] = np.nan
            elif reason == "gripper":
                bad[9, 7] = 200
            elif reason == "first":
                bad[0, 0] = 1
            elif reason == "speed":
                bad[9, 0] = 0.05
            else:
                bad = bad[:, :15]
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                validate_actions(bad, self.state, self.cfg)

    def test_execution_preserves_timing_units_and_gripper_order(self) -> None:
        self.cfg["control"]["enabled"] = True
        now = [0.0]
        robot = Mock()
        robot.current.return_value = np.zeros(14)
        observations = Mock()
        observations.state.return_value = self.state
        actions = self.actions.copy()
        actions[:, 15] = 1

        def poll() -> None:
            now[0] += 0.001

        with patch("infer.time.monotonic", side_effect=lambda: now[0]):
            execute_chunk(robot, observations, actions, self.cfg, poll)
        self.assertAlmostEqual(now[0], 0.201, places=2)
        final = robot.publish.call_args.args
        np.testing.assert_allclose(final[0][:7], np.rad2deg(0.01))
        np.testing.assert_allclose(final[0][7:], np.rad2deg(-0.01))
        np.testing.assert_array_equal(final[1], [0, 1])

    def test_dry_run_and_cancel_do_not_send(self) -> None:
        robot = Mock()
        observations = Mock()
        observations.state.return_value = self.state
        now = [0.0]

        def poll() -> None:
            now[0] += 0.001

        with patch("infer.time.monotonic", side_effect=lambda: now[0]):
            execute_chunk(robot, observations, self.actions, self.cfg, poll)
        robot.publish.assert_not_called()
        self.cfg["control"]["enabled"] = True
        with self.assertRaises(CancelledError):
            execute_chunk(robot, observations, self.actions, self.cfg, Mock(side_effect=CancelledError))
        robot.publish.assert_not_called()

    def test_ipc_handles_partial_delivery_without_pickle(self) -> None:
        left, right = socket.socketpair()
        client = Client.__new__(Client)
        client.connection = left
        left.setblocking(False)
        data = encode_packet(actions=self.actions, latency_s=0.5)

        def writer() -> None:
            try:
                for offset in range(0, len(data), 13):
                    right.sendall(data[offset : offset + 13])
            finally:
                right.close()

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            response = client.receive(time.monotonic() + 2, lambda: None)
            np.testing.assert_array_equal(response["actions"], self.actions)
        finally:
            client.close()
            thread.join()
        packet = encode_packet(error="test")
        self.assertEqual(str(decode_packet(packet[4:])["error"]), "test")
        self.assertEqual(struct.unpack("!I", packet[:4])[0], len(packet) - 4)

    def test_worker_eof_and_cancel_while_waiting(self) -> None:
        left, right = socket.socketpair()
        right.close()
        try:
            with self.assertRaises(EOFError):
                receive_exact(left, 4)
        finally:
            left.close()
        client = Client.__new__(Client)
        client.connection = Mock()
        with self.assertRaises(CancelledError):
            client.receive(100, Mock(side_effect=CancelledError))
        client.connection.recv.assert_not_called()

    def test_full_local_worker_handshake_prediction_and_shutdown(self) -> None:
        runner = Mock()
        runner.predict.return_value = self.actions
        failures = []
        with tempfile.TemporaryDirectory(prefix="pi05_test_") as directory:
            address = str(Path(directory) / "worker.sock")

            def worker() -> None:
                try:
                    serve(self.cfg, address)
                except Exception as exc:
                    failures.append(exc)

            with patch("policy_worker.PolicyRunner", return_value=runner):
                thread = threading.Thread(target=worker, daemon=True)
                thread.start()
                client = Client(address, 2, lambda: time.sleep(0.001))
                try:
                    result = client.predict({"state": self.state}, 2, lambda: time.sleep(0.001))
                    np.testing.assert_array_equal(result["actions"], self.actions)
                finally:
                    client.close()
                    thread.join(timeout=2)
                self.assertFalse(thread.is_alive())
                self.assertEqual(failures, [])

    def test_worker_rgb_chw_state_and_postprocessing(self) -> None:
        import torch

        runner = PolicyRunner.__new__(PolicyRunner)
        runner.torch = torch
        runner.cfg = self.cfg
        runner.policy = Mock()
        runner.policy.predict_action_chunk.return_value = torch.zeros((1, 50, 16))
        runner.pre = Mock(side_effect=lambda batch: batch)
        runner.post = Mock(side_effect=lambda actions: actions + 0.125)
        data = {"state": self.state, "reset": np.array(True)}
        for name, camera in self.cfg["cameras"].items():
            image = np.zeros((camera["height"], camera["width"], 3), np.uint8)
            image[:, :, 0] = 255
            data[name] = image
        result = runner.predict(data)
        self.assertEqual(result.shape, (10, 16))
        np.testing.assert_allclose(result, 0.125)
        batch = runner.pre.call_args.args[0]
        self.assertEqual(batch["observation.images.head"].shape, (3, 480, 640))
        self.assertEqual(float(batch["observation.images.head"][0, 0, 0]), 1)
        self.assertEqual(float(batch["observation.images.head"][2, 0, 0]), 0)
        self.assertEqual(batch["observation.state"].shape, (16,))
        runner.policy.reset.assert_called_once()
        runner.pre.reset.assert_called_once()

    def test_native_checkpoint_load_is_strict(self) -> None:
        import torch
        from safetensors.torch import save_model

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = torch.nn.Linear(3, 2)
            save_model(source, str(path / "model.safetensors"))
            target = torch.nn.Linear(3, 2)
            load_native_weights(target, path, "cpu")
            torch.testing.assert_close(source.weight, target.weight)
            with self.assertRaises(RuntimeError):
                load_native_weights(torch.nn.Sequential(torch.nn.Linear(3, 2)), path, "cpu")
            (path / "model.safetensors").unlink()
            with self.assertRaises(FileNotFoundError):
                load_native_weights(target, path, "cpu")

    def test_worker_dispatches_checkpoint_type_and_preserves_string_dtype(self) -> None:
        from lerobot.configs import FeatureType, PolicyFeature
        from lerobot.policies.online_filtering_pi05.configuration_online_filtering_pi05 import (
            OnlineFilteringPI05Config,
        )
        from lerobot.policies.pi05.configuration_pi05 import PI05Config

        inputs = {
            f"observation.images.{name}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
            for name in self.cfg["cameras"]
        }
        inputs["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(16,))
        outputs = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))}
        for config_type in (PI05Config, OnlineFilteringPI05Config):
            with (
                self.subTest(config_type=config_type.__name__),
                tempfile.TemporaryDirectory() as directory,
                ExitStack() as stack,
            ):
                path = Path(directory)
                for filename in (
                    "config.json",
                    "model.safetensors",
                    "policy_preprocessor.json",
                    "policy_postprocessor.json",
                ):
                    (path / filename).touch()
                cfg = {
                    **self.cfg,
                    "model": {**self.cfg["model"], "checkpoint": directory, "dinov3_config": directory},
                }
                pcfg = config_type(input_features=inputs.copy(), output_features=outputs.copy(), device="cpu")
                stack.enter_context(
                    patch("lerobot.configs.PreTrainedConfig.from_pretrained", return_value=pcfg)
                )
                base = stack.enter_context(patch("lerobot.policies.pi05.modeling_pi05.PI05Policy"))
                online = stack.enter_context(
                    patch(
                        "lerobot.policies.online_filtering_pi05.modeling_online_filtering_pi05.OnlineFilteringPI05Policy"
                    )
                )
                stack.enter_context(patch("transformers.DINOv3ViTConfig.from_pretrained"))
                stack.enter_context(patch("transformers.DINOv3ViTModel"))
                stack.enter_context(patch("lerobot.policies.pi05.online_filtering.DINOv3PatchEncoder"))
                pre, post = Mock(), Mock()
                stack.enter_context(
                    patch("lerobot.policies.factory.make_pre_post_processors", return_value=(pre, post))
                )
                reconcile = stack.enter_context(
                    patch(
                        "lerobot.policies.online_filtering_pi05.processor_online_filtering_pi05.reconcile_online_filtering_pi05_pretrained_processors",
                        return_value=(pre, post),
                    )
                )
                load = stack.enter_context(patch("policy_worker.load_native_weights"))
                runner = PolicyRunner(cfg)
                self.assertEqual(pcfg.dtype, "bfloat16")
                expected = online if config_type is OnlineFilteringPI05Config else base
                self.assertIs(runner.policy, expected.return_value)
                load.assert_called_once_with(expected.return_value, path, "cuda")
                self.assertEqual(reconcile.call_count, int(config_type is OnlineFilteringPI05Config))

    def test_worker_rejects_invalid_model_output(self) -> None:
        import torch

        runner = PolicyRunner.__new__(PolicyRunner)
        runner.torch, runner.cfg = torch, self.cfg
        runner.policy, runner.pre, runner.post = (
            Mock(),
            Mock(side_effect=lambda b: b),
            Mock(side_effect=lambda a: a),
        )
        data = {"state": self.state}
        for name, camera in self.cfg["cameras"].items():
            data[name] = np.zeros((camera["height"], camera["width"], 3), np.uint8)
        for output in (torch.zeros(1, 2, 16), torch.zeros(1, 50, 7), torch.full((1, 50, 16), float("nan"))):
            runner.policy.predict_action_chunk.return_value = output
            with self.assertRaises(ValueError):
                runner.predict(data)

    def test_invalid_outputs_never_publish_and_stale_observations_stop(self) -> None:
        robot = Mock()
        observations = Mock()
        observations.state.return_value = self.state
        bad = self.actions.copy()
        bad[-1, 0] = 5
        with self.assertRaises(ValueError):
            execute_chunk(robot, observations, bad, self.cfg, Mock())
        robot.publish.assert_not_called()
        observations.state.side_effect = RuntimeError("feedback expired")
        with self.assertRaisesRegex(RuntimeError, "expired"):
            execute_chunk(robot, observations, self.actions, self.cfg, Mock())
        robot.publish.assert_not_called()

    def test_cloud_training_args_preserve_pi05_and_dataset(self) -> None:
        args = training_args(self.cfg)
        self.assertIn("--policy.type=pi05", args)
        self.assertIn("--policy.pretrained_path=lerobot/pi05_base", args)
        self.assertIn("--policy.n_action_steps=10", args)
        self.assertIn(f"--dataset.root={self.cfg['training']['dataset_root']}", args)


if __name__ == "__main__":
    unittest.main()
