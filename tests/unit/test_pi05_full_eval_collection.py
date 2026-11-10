"""Check that PI05 full curves only publish complete, matched native-control runs."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def pool_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "research/scripts"))
    return importlib.import_module("run_pi05_full_delay_eval")


def write_summary(out, module, *, field=None, value=None):
    data = {
        "suite": "libero_spatial", "task_id": 0,
        "policy_path": module.ONLINE,
        "xvla_precomputed_deltas": False, "xvla_relative_execution": False,
        "control_mode": "relative", "evaluation_protocol": "continuous_online_v1",
        "xvla_observation_delivery": "direct", "xvla_fast_ablation": "normal",
        "control_fps": 20, "replan_interval": 25, "delay_range": [25, 25],
        "diagnostic_paired_seed": True,
        "episodes": [
            {"episode": i, "success": True, "steps": 100, "queue_underruns": 0,
             "delta_clipped_commands": 0, "smoke_test_capped": False,
             "initial_action_diagnostics": {"model_shape": [1, 50, 7]}}
            for i in range(10)
        ],
    }
    if field:
        data[field] = value
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(data))


def test_matched_online_run_is_collected(pool_module, tmp_path):
    out = tmp_path / "online_d25"
    write_summary(out, pool_module)
    assert len(pool_module.collect(out, "libero_spatial", [0], 25, 10)) == 10


@pytest.mark.parametrize("field,value", [
    ("replan_interval", 20), ("control_mode", "absolute"),
    ("xvla_observation_delivery", "buffered"), ("xvla_fast_ablation", "stale_all"),
    ("policy_path", "wrong-checkpoint"),
])
def test_incompatible_run_is_rejected(pool_module, tmp_path, field, value):
    out = tmp_path / "online_d25"
    write_summary(out, pool_module, field=field, value=value)
    with pytest.raises(AssertionError):
        pool_module.collect(out, "libero_spatial", [0], 25, 10)


def test_incomplete_run_is_rejected(pool_module, tmp_path):
    out = tmp_path / "online_d25"
    write_summary(out, pool_module)
    with pytest.raises(AssertionError):
        pool_module.collect(out, "libero_spatial", [0, 5], 25, 10)
