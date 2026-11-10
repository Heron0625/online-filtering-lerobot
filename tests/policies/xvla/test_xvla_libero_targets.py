import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.utils import (
    axis_angle_to_matrix,
    libero_action_to_xvla_ee6d,
    libero_relative_action_to_xvla_ee6d,
    libero_state_to_xvla_proprio,
    remap_action_prefix_positional_embedding,
    rotate6d_to_axis_angle,
)


def test_empty_camera_is_not_counted_twice() -> None:
    image = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
    state = PolicyFeature(type=FeatureType.STATE, shape=(8,))
    config = XVLAConfig(
        input_features={
            "observation.images.image": image,
            "observation.images.image2": image,
            "observation.state": state,
        },
        num_image_views=3,
        empty_cameras=1,
    )

    config.validate_features()
    config.validate_features()

    assert config.num_image_views == 3
    assert len(config.image_features) == 3
    assert "observation.images.empty_camera_0" in config.image_features


def test_libero_action_to_xvla_ee6d_layout_and_roundtrip() -> None:
    action = torch.tensor([[[0.1, -0.2, 0.3, 0.2, -0.1, 0.4, -1.0], [0.0, 0.0, 0.0, -0.3, 0.2, 0.1, 1.0]]])
    converted = libero_action_to_xvla_ee6d(action)

    assert converted.shape == (1, 2, 20)
    torch.testing.assert_close(converted[..., :3], action[..., :3])
    torch.testing.assert_close(converted[..., 9], torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(converted[..., 10:], torch.zeros_like(converted[..., 10:]))
    recovered = rotate6d_to_axis_angle(converted[0, :, 3:9].numpy())
    np.testing.assert_allclose(recovered, action[0, :, 3:6].numpy(), atol=2e-5)


def test_libero_state_to_xvla_proprio_layout() -> None:
    state = torch.tensor([[0.1, 0.2, 0.3, 0.2, -0.1, 0.4, 0.7, -0.7]])
    converted = libero_state_to_xvla_proprio(state)
    assert converted.shape == (1, 20)
    torch.testing.assert_close(converted[:, :3], state[:, :3])
    body = Rotation.from_rotvec(state[:, 3:6].numpy()).as_matrix()
    offset = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    expected = body @ offset
    np.testing.assert_allclose(converted[:, 3:6], expected[:, :, 0], atol=2e-5)
    np.testing.assert_allclose(converted[:, 6:9], expected[:, :, 1], atol=2e-5)
    torch.testing.assert_close(converted[:, 9:], torch.zeros_like(converted[:, 9:]))


def test_libero_relative_action_reproduces_osc_pose_goal() -> None:
    state = torch.tensor([[[0.1, -0.2, 0.8, 0.2, -0.1, 0.4, 0.02, -0.02]]])
    action = torch.tensor([[[0.5, -2.0, 0.25, 0.2, -0.4, 2.0, 1.0]]])
    target = libero_relative_action_to_xvla_ee6d(state, action)

    # OSC_POSE clips to [-1, 1] and maps position commands to [-0.05, 0.05].
    torch.testing.assert_close(target[..., :3], torch.tensor([[[0.125, -0.25, 0.8125]]]))
    torch.testing.assert_close(target[..., 9], torch.ones(1, 1))
    current = axis_angle_to_matrix(state[..., 3:6])
    delta = axis_angle_to_matrix(0.5 * action[..., 3:6].clamp(-1, 1))
    offset = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    expected_rotation = delta @ current @ offset
    expected_6d = torch.cat((expected_rotation[..., :, 0], expected_rotation[..., :, 1]), dim=-1)
    torch.testing.assert_close(target[..., 3:9], expected_6d)
    torch.testing.assert_close(target[..., 10:], torch.zeros_like(target[..., 10:]))


def test_zero_rotation_retains_goal_across_observation_changes() -> None:
    # The second observation moves, but zero rotation must retain the first
    # command's goal rather than use the newly measured pose.
    state = torch.zeros(1, 4, 8)
    state[0, :, 5] = torch.tensor([0.0, 0.1, 0.2, 0.3])
    action = torch.zeros(1, 4, 7)
    action[0, 0, 5] = 0.4
    action[0, 3, 5] = 0.2
    result = libero_relative_action_to_xvla_ee6d(state, action)
    torch.testing.assert_close(result[:, 1:3, 3:9], result[:, :1, 3:9].expand(-1, 2, -1))
    assert not torch.allclose(result[:, 3, 3:9], result[:, 0, 3:9])
    # A chunk starting inside the zero-command run needs the preceding goal.
    r = result[0, 0, 3:9]
    initial = torch.stack((r[:3], r[3:], torch.linalg.cross(r[:3], r[3:])), dim=-1)[None]
    chunk = libero_relative_action_to_xvla_ee6d(state[:, 1:3], action[:, 1:3], initial_goal_rotation=initial)
    torch.testing.assert_close(chunk, result[:, 1:3])


def test_dataset_wrist_orientation_is_explicit_and_consumed_once() -> None:
    from lerobot.policies.xvla.processor_xvla import (
        XVLA_WRIST_FLIP_KEY,
        XVLADatasetImageOrientationProcessorStep,
    )

    step = XVLADatasetImageOrientationProcessorStep()
    image = torch.arange(24).reshape(2, 1, 3, 4)
    obs = {"observation.images.image2": image, XVLA_WRIST_FLIP_KEY: torch.tensor([[True], [False]])}
    processed = step.observation(obs)
    torch.testing.assert_close(processed["observation.images.image2"][0], image[0].flip((-2, -1)))
    torch.testing.assert_close(processed["observation.images.image2"][1], image[1])
    assert XVLA_WRIST_FLIP_KEY not in processed
    torch.testing.assert_close(
        step.observation(processed)["observation.images.image2"], processed["observation.images.image2"]
    )


def test_positional_remap_preserves_existing_actions_and_conditions() -> None:
    embedding = torch.arange(512 * 3, dtype=torch.float32).reshape(1, 512, 3)
    remapped = remap_action_prefix_positional_embedding(embedding, 30, 50)
    torch.testing.assert_close(remapped[:, :30], embedding[:, :30])
    torch.testing.assert_close(remapped[:, 30:50], embedding[:, 29:30].expand(-1, 20, -1))
    torch.testing.assert_close(remapped[:, 50:], embedding[:, 30:492])


def test_full_episode_matches_recorded_robosuite_controller() -> None:
    """Reference goals came from controller.set_goal, including five zero-rotation commands."""
    from pathlib import Path

    reference = np.load(Path(__file__).parents[2] / "fixtures/xvla_libero_controller_reference.npz")
    predicted = libero_relative_action_to_xvla_ee6d(
        torch.from_numpy(reference["state"]),
        torch.from_numpy(reference["action"]),
        initial_goal_rotation=torch.from_numpy(reference["initial_goal"]),
    )
    torch.testing.assert_close(predicted, torch.from_numpy(reference["target"]), atol=1e-6, rtol=1e-6)


def test_training_rejects_relative_windows_without_episode_history() -> None:
    from types import SimpleNamespace

    import pytest

    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    policy = SimpleNamespace(config=XVLAConfig(chunk_size=2, n_action_steps=2))
    with pytest.raises(ValueError, match="full-episode controller history"):
        XVLAPolicy._prepare_action_targets(policy, {"action": torch.zeros(1, 2, 7)})
