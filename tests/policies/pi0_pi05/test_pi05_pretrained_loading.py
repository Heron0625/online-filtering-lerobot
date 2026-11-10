"""Pretrained loading must fail before a training run can use uninitialized weights."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from lerobot.policies.pi05.modeling_pi05 import PI05Policy


class TinyPI05Policy(PI05Policy):
    def __init__(self, config, **kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        self.weight = torch.nn.Parameter(torch.zeros(2))

    def _fix_pytorch_state_dict_keys(self, state_dict, config):
        return state_dict

    def _prepare_pretrained_state_dict(self, state_dict):
        return state_dict


def test_unreadable_checkpoint_raises(monkeypatch):
    def missing(*args, **kwargs):
        raise OSError("weights not found")

    monkeypatch.setattr("transformers.utils.cached_file", missing)
    with pytest.raises(RuntimeError, match="Could not load PI05 pretrained weights"):
        TinyPI05Policy.from_pretrained("missing-checkpoint", config=SimpleNamespace())


@pytest.mark.parametrize("state", [{"weight": torch.ones(3)}, {"wrong_key": torch.ones(2)}])
def test_incompatible_checkpoint_raises(monkeypatch, tmp_path, state):
    path = tmp_path / "model.safetensors"
    save_file(state, path)
    monkeypatch.setattr("transformers.utils.cached_file", lambda *args, **kwargs: str(path))
    with pytest.raises(RuntimeError, match="Could not load PI05 pretrained weights"):
        TinyPI05Policy.from_pretrained(tmp_path, config=SimpleNamespace())


def test_matching_checkpoint_loads(monkeypatch, tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.ones(2)}, path)
    monkeypatch.setattr("transformers.utils.cached_file", lambda *args, **kwargs: str(path))
    policy = TinyPI05Policy.from_pretrained(tmp_path, config=SimpleNamespace())
    assert torch.equal(policy.weight, torch.ones(2))
