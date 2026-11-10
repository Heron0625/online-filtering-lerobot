import math

import numpy as np
import torch


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert exponential-coordinate axis angles to rotation matrices."""
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"expected axis-angle vectors with width 3, got {axis_angle.shape}")
    x, y, z = axis_angle.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1).reshape(
        *axis_angle.shape[:-1], 3, 3
    )
    theta = torch.linalg.vector_norm(axis_angle, dim=-1)
    a = torch.sinc(theta / torch.pi)
    b = 0.5 * torch.sinc(theta / (2 * torch.pi)).square()
    identity = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    return identity + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)


def axis_angle_to_rotation_6d(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis angles to X-VLA's concatenated first-two-column representation."""
    matrix = axis_angle_to_matrix(axis_angle)
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def libero_action_to_xvla_ee6d(action: torch.Tensor) -> torch.Tensor:
    """Map LIBERO ``[xyz, axis-angle, gripper]`` actions to X-VLA's 20D layout.

    LIBERO gripper commands use ``-1/+1`` while X-VLA trains gripper logits with
    BCE targets in ``0/1``.  The second 10D arm slot remains zero for LIBERO.
    """
    if action.shape[-1] != 7:
        raise ValueError(f"expected 7D LIBERO actions, got {action.shape}")
    result = action.new_zeros(*action.shape[:-1], 20)
    result[..., :3] = action[..., :3]
    result[..., 3:9] = axis_angle_to_rotation_6d(action[..., 3:6])
    result[..., 9] = ((action[..., 6] + 1) / 2).clamp(0, 1)
    return result


def libero_body_to_controller_rotation(state: torch.Tensor) -> torch.Tensor:
    """Convert the LIBERO Panda body rotvec to the OSC grip-site frame."""
    offset = state.new_tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    return axis_angle_to_matrix(state[..., 3:6]) @ offset


def libero_relative_action_to_xvla_ee6d(
    state: torch.Tensor,
    action: torch.Tensor,
    *,
    initial_goal_rotation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reproduce LIBERO OSC_POSE controller goals as X-VLA absolute EE6D targets.

    LIBERO uses robosuite's default OSC_POSE controller: input commands are
    clipped to [-1, 1], position deltas are scaled by 0.05, orientation
    exponential coordinates by 0.5, and the rotation goal is left-multiplied
    onto the controller site orientation, not the observed Panda body frame.
    Zero rotation commands retain the previous goal. The penultimate dimension
    is a complete, chronological episode, never an independently sampled chunk.
    If omitted, the initial goal is the first measured site orientation.
    Position/orientation limits are disabled.
    """
    if state.shape[:-1] != action.shape[:-1]:
        raise ValueError(f"state/action leading shapes differ: {state.shape} vs {action.shape}")
    if state.shape[-1] != 8 or action.shape[-1] != 7:
        raise ValueError("expected LIBERO state/action widths 8 and 7")

    result = action.new_zeros(*action.shape[:-1], 20)
    clipped = action[..., :6].clamp(-1, 1)
    result[..., :3] = state[..., :3] + 0.05 * clipped[..., :3]
    current_rotation = libero_body_to_controller_rotation(state)
    delta_rotation = axis_angle_to_matrix(0.5 * clipped[..., 3:6])
    goal_rotation = delta_rotation @ current_rotation
    zero_rotation = (clipped[..., 3:6] == 0).all(dim=-1)
    if action.ndim == 1:
        initial = current_rotation if initial_goal_rotation is None else initial_goal_rotation
        goal_rotation = torch.where(zero_rotation[..., None, None], initial, goal_rotation)
    else:
        initial = current_rotation[..., 0, :, :] if initial_goal_rotation is None else initial_goal_rotation
        # Gather the most recent nonzero command, including history before any
        # training window. Index zero addresses the episode's initial goal.
        indices = torch.arange(action.shape[-2], device=action.device) + 1
        indices = torch.where(zero_rotation, 0, indices).cummax(dim=-1).values
        candidates = torch.cat((initial.unsqueeze(-3), goal_rotation), dim=-3)
        goal_rotation = candidates.gather(-3, indices[..., None, None].expand(*indices.shape, 3, 3))
    result[..., 3:9] = torch.cat((goal_rotation[..., :, 0], goal_rotation[..., :, 1]), dim=-1)
    result[..., 9] = ((action[..., 6] + 1) / 2).clamp(0, 1)
    return result


def libero_state_to_xvla_proprio(state: torch.Tensor) -> torch.Tensor:
    """Map the 8D LIBERO dataset state to the 20D X-VLA proprio layout."""
    if state.shape[-1] != 8:
        raise ValueError(f"expected 8D LIBERO state, got {state.shape}")
    result = state.new_zeros(*state.shape[:-1], 20)
    result[..., :3] = state[..., :3]
    rotation = libero_body_to_controller_rotation(state)
    result[..., 3:9] = torch.cat((rotation[..., :, 0], rotation[..., :, 1]), dim=-1)
    # Match LiberoProcessorStep: the single-arm auxiliary/gripper proprio slot
    # is zero and the unused second-arm slot is all zeros.
    return result


def remap_action_prefix_positional_embedding(
    embedding: torch.Tensor, old_horizon: int, new_horizon: int
) -> torch.Tensor:
    """Extend an action-prefix sequence without shifting pretrained condition positions.

    X-VLA concatenates ``[action, condition]`` before adding one shared positional
    embedding.  Increasing the action horizon therefore requires moving the old
    condition positions by the same offset.  Existing action positions are kept
    exact; only newly introduced action positions use interpolation.
    """
    if embedding.ndim != 3 or embedding.shape[0] != 1:
        raise ValueError(f"expected positional embedding [1, length, dim], got {embedding.shape}")
    if not 0 < old_horizon < new_horizon < embedding.shape[1]:
        raise ValueError("horizons must satisfy 0 < old < new < embedding length")
    offset = new_horizon - old_horizon
    result = embedding.clone()
    # Learned positions do not define a meaningful extrapolation rule.  Reuse
    # the last known action position for new slots; this preserves all existing
    # action positions and avoids the discontinuity caused by stretched
    # interpolation wrapping back into the old range.
    result[:, old_horizon:new_horizon] = embedding[:, old_horizon - 1 : old_horizon]
    result[:, new_horizon:] = embedding[:, old_horizon : embedding.shape[1] - offset]
    return result


def mat2quat(rmat):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (x,y,z,w) float quaternion angles
    """
    mat = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = mat[0, 0]
    m01 = mat[0, 1]
    m02 = mat[0, 2]
    m10 = mat[1, 0]
    m11 = mat[1, 1]
    m12 = mat[1, 2]
    m20 = mat[2, 0]
    m21 = mat[2, 1]
    m22 = mat[2, 2]
    # symmetric matrix k
    k = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    k /= 3.0
    # quaternion is Eigen vector of k that corresponds to largest eigenvalue
    w, v = np.linalg.eigh(k)
    inds = np.array([3, 0, 1, 2])
    q1 = v[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]


def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def rotate6d_to_axis_angle(r6d):
    """
    r6d: np.ndarray, shape (N, 6)
    return: np.ndarray, shape (N, 3), axis-angle vectors
    """
    flag = 0
    if len(r6d.shape) == 1:
        r6d = r6d[None, ...]
        flag = 1

    a1 = r6d[:, 0:3]
    a2 = r6d[:, 3:6]

    # b1
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-6)

    # b2
    dot_prod = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2_orth = a2 - dot_prod * b1
    b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + 1e-6)

    # b3
    b3 = np.cross(b1, b2, axis=-1)

    rotation_matrix = np.stack([b1, b2, b3], axis=-1)  # shape: (N, 3, 3)

    axis_angle_list = []
    for i in range(rotation_matrix.shape[0]):
        quat = mat2quat(rotation_matrix[i])
        axis_angle = quat2axisangle(quat)
        axis_angle_list.append(axis_angle)

    axis_angle_array = np.stack(axis_angle_list, axis=0)  # shape: (N, 3)

    if flag == 1:
        axis_angle_array = axis_angle_array[0]

    return axis_angle_array


def mat_to_rotate6d(abs_action):
    if len(abs_action.shape) == 2:
        return np.concatenate([abs_action[:3, 0], abs_action[:3, 1]], axis=-1)
    elif len(abs_action.shape) == 3:
        return np.concatenate([abs_action[:, :3, 0], abs_action[:, :3, 1]], axis=-1)
    else:
        raise NotImplementedError


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def xvla_pose_rotation_matrix(pose: torch.Tensor) -> torch.Tensor:
    """Decode the first arm's EE6D orientation without subtracting rotation coordinates."""
    first = torch.nn.functional.normalize(pose[..., 3:6], dim=-1)
    second = pose[..., 6:9] - (first * pose[..., 6:9]).sum(-1, keepdim=True) * first
    if (pose[..., 3:6].norm(dim=-1) < 1e-8).any() or (second.norm(dim=-1) < 1e-8).any():
        raise ValueError("degenerate first-arm EE6D rotation")
    second = torch.nn.functional.normalize(second, dim=-1)
    return torch.stack((first, second, torch.linalg.cross(first, second)), dim=-1)


def xvla_chunk_to_libero_delta(actions: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Difference consecutive absolute targets into normalized OSC commands (not original PI05 labels).

    Only index zero uses the trigger-time measured anchor. Delayed installation
    slices this already-differenced chunk without reanchoring its first executed target.
    """
    if actions.ndim != 2 or actions.shape[1] != 20 or anchor.shape != (20,):
        raise ValueError("expected a (H,20) chunk and a (20,) measured trigger pose")
    previous = torch.cat((anchor[None].to(actions), actions[:-1]), dim=0)
    rotations = xvla_pose_rotation_matrix(actions)
    old_rotations = xvla_pose_rotation_matrix(previous)
    relative = (rotations @ old_rotations.transpose(-1, -2)).detach().double().cpu().numpy()
    vectors = np.stack([quat2axisangle(mat2quat(matrix)) for matrix in relative])
    command = actions.new_zeros(actions.shape[0], 7)
    command[:, :3] = (actions[:, :3] - previous[:, :3]) / 0.05
    command[:, 3:6] = torch.as_tensor(vectors, device=actions.device, dtype=actions.dtype) / 0.5
    command[:, 6] = torch.where(actions[:, 9] > 0.5, 1.0, -1.0)
    return command


def libero_delta_command_goal(
    command: torch.Tensor, measured: torch.Tensor, previous_goal: torch.Tensor | None = None
) -> torch.Tensor:
    """Reconstruct the actual absolute OSC target sent by a normalized relative command."""
    if command.shape != (7,) or measured.shape != (20,):
        raise ValueError("expected a (7,) relative command and a (20,) measured pose")
    clipped = command.clamp(-1, 1)
    result = measured.new_zeros(20)
    result[:3] = measured[:3] + clipped[:3].to(measured) * 0.05
    rotation = axis_angle_to_matrix(clipped[3:6].to(measured) * 0.5) @ xvla_pose_rotation_matrix(measured)
    if not clipped[3:6].any() and previous_goal is not None:
        rotation = xvla_pose_rotation_matrix(previous_goal.to(measured))
    result[3:9] = torch.cat((rotation[:, 0], rotation[:, 1]))
    result[9] = ((clipped[6] + 1) / 2).to(measured)
    return result


def xvla_target_to_libero_delta(
    target: torch.Tensor, measured: torch.Tensor, previous_goal: torch.Tensor | None = None
) -> torch.Tensor:
    """Convert one absolute target against the measured pose at execution time.

    This preserves the intended target when OSC limits do not clip the command.
    It does not turn a saved absolute-target model into an open-loop delta policy.
    OSC retains its previous orientation goal on an exactly zero rotation command;
    a 1e-7 rad update handles a requested current orientation when that goal differs.
    """
    if target.shape != (20,) or measured.shape != (20,):
        raise ValueError("expected (20,) absolute target and measured pose")
    command = xvla_chunk_to_libero_delta(target[None], measured)[0]
    if not command[3:6].any() and (
        previous_goal is None
        or not torch.allclose(
            xvla_pose_rotation_matrix(previous_goal.to(target)),
            xvla_pose_rotation_matrix(target),
            rtol=0,
            atol=1e-7,
        )
    ):
        command[3] = 2e-7
    return command


def accumulate_xvla_delta(command: torch.Tensor, target_base: torch.Tensor) -> torch.Tensor:
    """Apply a fixed normalized delta to a persistent target base, not a live measured pose."""
    return libero_delta_command_goal(command, target_base, target_base)
