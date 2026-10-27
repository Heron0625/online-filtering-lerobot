"""Spatial fast-token isolation and pre-fusion slow masking regression tests."""

from types import SimpleNamespace

import torch

from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
from lerobot.policies.pi05.online_filtering import FastTokenResampler


def test_spatial_resampler_camera_and_proprio_isolation():
    """Each camera owns 32 visual tokens; proprio bypasses visual compression."""
    torch.manual_seed(42)
    module = FastTokenResampler(
        8, 16, num_fast_tokens=64, num_heads=4, num_cameras=2, state_dim=7, spatial_tokens=True
    )
    initial = torch.randn(2, 2, 16, 8)
    current = torch.randn_like(initial)
    states = torch.randn(2, 7)
    delays = torch.tensor([2, 8])
    baseline = module(initial, current, delays=delays, states=states)
    changed = current.clone()
    changed[:, 0] += torch.randn_like(changed[:, 0])
    image_changed = module(initial, changed, delays=delays, states=states)
    state_changed = module(initial, current, delays=delays, states=states + 1)
    assert baseline.shape == (2, 65, 16)
    assert torch.equal(baseline[:, 32:], image_changed[:, 32:])
    assert not torch.equal(baseline[:, :32], image_changed[:, :32])
    assert torch.equal(baseline[:, :64], state_changed[:, :64])
    assert not torch.equal(baseline[:, 64:], state_changed[:, 64:])
    baseline.square().sum().backward()
    assert module.position_proj.weight.grad is not None
    assert module.state_proj[0].weight.grad is not None
    assert torch.isfinite(module.queries.grad).all()


def test_slow_mask_excludes_visual_tokens_before_language_fusion():
    """Masked visual embeddings are zero and invalid, while language stays visible."""
    fake = SimpleNamespace(
        paligemma_with_expert=SimpleNamespace(
            embed_image=lambda image: torch.ones(image.shape[0], 4, 8),
            embed_language_tokens=lambda tokens: torch.ones(tokens.shape[0], tokens.shape[1], 8),
        ),
        _apply_checkpoint=lambda fn, *args: fn(*args),
    )
    embeddings, valid, _ = PI05Pytorch.embed_prefix(
        fake,
        [torch.zeros(1, 3, 8, 8)],
        [torch.ones(1, dtype=torch.bool)],
        torch.ones(1, 2, dtype=torch.long),
        torch.ones(1, 2, dtype=torch.bool),
        slow_visual_masks=[torch.tensor([[True, False, True, False]])],
    )
    assert valid.tolist() == [[True, False, True, False, True, True]]
    assert (embeddings[:, [1, 3]] == 0).all()
    assert (embeddings[:, 4:] == 1).all()


def test_executed_command_is_normalized_after_environment_conversion():
    """Clipping/discretization changes must survive the round-trip into the committed prefix."""
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor.normalize_processor import NormalizerProcessorStep
    from research.scripts.eval_libero_synthetic_delay import normalize_executed_command

    normalizer = NormalizerProcessorStep(
        features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
        stats={"action": {"mean": torch.zeros(7), "std": torch.full((7,), 2.0)}},
    )
    planned = torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]])
    sent = planned.clamp(-1, 1)
    sent[:, -1] = 1
    committed = normalize_executed_command(normalizer, sent)
    assert torch.allclose(committed, sent / 2)
    assert not torch.allclose(committed, planned / 2)
