"""Physical semantics of the experimental target-to-target relative execution interface."""

import torch

from lerobot.policies.xvla.utils import (
    libero_action_to_xvla_ee6d,
    libero_delta_command_goal,
    xvla_chunk_to_libero_delta,
    xvla_pose_rotation_matrix,
)


def pose(x, angle):
    return libero_action_to_xvla_ee6d(torch.tensor([x, 0.0, 0.0, *angle, 1.0]))


def test_chunk_deltas_reconstruct_targets_under_perfect_tracking():
    anchor = pose(0.0, [0.0, 0.0, 0.0])
    targets = torch.stack(
        [pose(0.01, [0.1, 0.0, 0.0]), pose(0.02, [0.0, 0.1, 0.0]), pose(0.03, [0.0, 0.0, 0.1])]
    )
    commands = xvla_chunk_to_libero_delta(targets, anchor)
    current = anchor
    previous_goal = None
    for command, target in zip(commands, targets, strict=True):
        actual = libero_delta_command_goal(command, current, previous_goal)
        torch.testing.assert_close(actual[:3], target[:3])
        torch.testing.assert_close(
            xvla_pose_rotation_matrix(actual), xvla_pose_rotation_matrix(target), atol=2e-6, rtol=2e-6
        )
        current, previous_goal = target, actual
    torch.testing.assert_close(commands[:, :3], torch.tensor([[0.2, 0.0, 0.0]]).expand(3, -1))


def test_delayed_installation_keeps_intra_chunk_difference_without_reanchoring():
    targets = torch.stack(
        [pose(0.01, [0.0, 0.0, 0.0]), pose(0.02, [0.0, 0.0, 0.0]), pose(0.03, [0.0, 0.0, 0.0])]
    )
    commands = xvla_chunk_to_libero_delta(targets, pose(0.0, [0.0, 0.0, 0.0]))
    measured = pose(0.10, [0.0, 0.0, 0.0])
    actual = libero_delta_command_goal(commands[2], measured)
    torch.testing.assert_close(actual[0], torch.tensor(0.11))
    assert not torch.allclose(actual[:3], targets[2, :3])


def test_relative_command_clips_motion_and_retains_previous_zero_rotation_goal():
    measured = pose(0.0, [0.0, 0.0, 0.0])
    previous = pose(0.0, [0.2, 0.0, 0.0])
    goal = libero_delta_command_goal(torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]), measured, previous)
    torch.testing.assert_close(goal[0], torch.tensor(0.05))
    torch.testing.assert_close(xvla_pose_rotation_matrix(goal), xvla_pose_rotation_matrix(previous))
    assert goal[9] == 0


def test_gripper_is_a_command_not_a_temporal_difference():
    targets = torch.stack([pose(0.01, [0.0, 0.0, 0.0]), pose(0.02, [0.0, 0.0, 0.0])])
    targets[0, 9] = 0
    command = xvla_chunk_to_libero_delta(targets, pose(0.0, [0.0, 0.0, 0.0]))
    torch.testing.assert_close(command[:, 6], torch.tensor([-1.0, 1.0]))


def test_expert_controller_goals_are_not_future_measured_positions():
    # Same 1cm relative goal offset every control step, with only 2mm actual progress.
    measured_x = torch.tensor([0.0, 0.002, 0.004])
    targets = torch.stack([pose(float(x + 0.01), [0.0, 0.0, 0.0]) for x in measured_x])
    commands = xvla_chunk_to_libero_delta(targets, pose(0.0, [0.0, 0.0, 0.0]))
    torch.testing.assert_close(commands[:, 0], torch.tensor([0.2, 0.04, 0.04]))
    # Original OSC commands would remain .2; differencing wrongly treats the
    # unachieved previous goal as the measured starting point.
    assert not torch.allclose(commands[1:, 0], torch.full((2,), 0.2))


def test_execution_time_conversion_preserves_expert_relative_commands_with_tracking_lag():
    from lerobot.policies.xvla.utils import xvla_target_to_libero_delta

    for x in [0.0, 0.002, 0.004]:
        measured = pose(x, [0.0, 0.0, 0.0])
        target = pose(x + 0.01, [0.0, 0.0, 0.0])
        command = xvla_target_to_libero_delta(target, measured)
        torch.testing.assert_close(command[0], torch.tensor(0.2))
        actual = libero_delta_command_goal(command, measured)
        torch.testing.assert_close(actual[:3], target[:3])
        torch.testing.assert_close(
            xvla_pose_rotation_matrix(actual), xvla_pose_rotation_matrix(target), atol=2e-7, rtol=0
        )


def test_execution_time_conversion_can_replace_previous_zero_rotation_goal():
    from lerobot.policies.xvla.utils import xvla_target_to_libero_delta

    measured = pose(0.0, [0.0, 0.0, 0.0])
    previous = pose(0.0, [0.2, 0.0, 0.0])
    command = xvla_target_to_libero_delta(measured, measured, previous)
    assert command[3:6].any()
    actual = libero_delta_command_goal(command, measured, previous)
    torch.testing.assert_close(
        xvla_pose_rotation_matrix(actual), xvla_pose_rotation_matrix(measured), atol=2e-7, rtol=0
    )


def test_accumulated_zero_deltas_keep_unachieved_target():
    from lerobot.policies.xvla.utils import accumulate_xvla_delta

    start = pose(0.0, [0.0, 0.0, 0.0])
    fixed_targets = torch.stack([pose(0.01, [0.0, 0.0, 0.0])] * 4)
    commands = xvla_chunk_to_libero_delta(fixed_targets, start)
    base = start
    for command in commands:
        base = accumulate_xvla_delta(command, base)
        torch.testing.assert_close(base[:3], fixed_targets[0, :3])


def test_accumulated_delayed_suffix_keeps_switch_offset_and_never_replays_prefix():
    from lerobot.policies.xvla.utils import accumulate_xvla_delta

    start = pose(0.0, [0.0, 0.0, 0.0])
    targets = torch.stack(
        [pose(0.01, [0.1, 0.0, 0.0]), pose(0.02, [0.0, 0.1, 0.0]), pose(0.03, [0.0, 0.0, 0.1])]
    )
    commands = xvla_chunk_to_libero_delta(targets, start)
    switch_state = pose(0.10, [0.0, 0.2, 0.0])
    base = accumulate_xvla_delta(commands[2], switch_state)
    torch.testing.assert_close(base[0], torch.tensor(0.11))
    assert not torch.allclose(base[:3], targets[2, :3])
    expected_rotation = (
        xvla_pose_rotation_matrix(targets[2])
        @ xvla_pose_rotation_matrix(targets[1]).T
        @ xvla_pose_rotation_matrix(switch_state)
    )
    torch.testing.assert_close(xvla_pose_rotation_matrix(base), expected_rotation, atol=2e-6, rtol=2e-6)


def test_accumulation_without_delay_recovers_the_full_predicted_path():
    from lerobot.policies.xvla.utils import accumulate_xvla_delta

    start = pose(0.0, [0.0, 0.0, 0.0])
    targets = torch.stack(
        [pose(0.01, [0.1, 0.0, 0.0]), pose(0.02, [0.0, 0.1, 0.0]), pose(0.03, [0.0, 0.0, 0.1])]
    )
    base = start
    for command, target in zip(xvla_chunk_to_libero_delta(targets, start), targets, strict=True):
        base = accumulate_xvla_delta(command, base)
        torch.testing.assert_close(base[:3], target[:3])
        torch.testing.assert_close(
            xvla_pose_rotation_matrix(base), xvla_pose_rotation_matrix(target), atol=2e-6, rtol=2e-6
        )
