"""Prevent incompatible execution protocols and incomplete episodes entering full results."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def collector(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "research/scripts"))
    return importlib.import_module("run_xvla_accumulated_full_eval").collect


def summary(task):
    return {
        "suite": "libero_spatial",
        "task_id": task,
        "xvla_precomputed_deltas": True,
        "xvla_relative_execution": False,
        "xvla_delta_execution": "accumulated",
        "control_mode": "absolute",
        "control_fps": 20,
        "replan_interval": 25,
        "delay_range": [10, 10],
        "diagnostic_paired_seed": True,
        "episodes": [
            {
                "episode": i,
                "success": True,
                "steps": 100,
                "queue_underruns": 0,
                "delta_clipped_commands": 0,
                "smoke_test_capped": False,
                "initial_action_diagnostics": {"model_shape": [1, 50, 20]},
            }
            for i in range(10)
        ],
    }


def write(out, data):
    p = out / str(data["task_id"])
    p.mkdir()
    (p / "summary.json").write_text(json.dumps(data))


def test_complete_matched_job_is_collected(collector, tmp_path):
    write(tmp_path, summary(0))
    write(tmp_path, summary(5))
    assert len(collector(tmp_path, "libero_spatial", [0, 5], 10, 10)) == 20


@pytest.mark.parametrize(
    "field,value",
    [("replan_interval", 20), ("xvla_delta_execution", "current_pose"), ("xvla_relative_execution", True)],
)
def test_incompatible_job_is_rejected(collector, tmp_path, field, value):
    a = summary(0)
    a[field] = value
    write(tmp_path, a)
    write(tmp_path, summary(5))
    with pytest.raises(AssertionError):
        collector(tmp_path, "libero_spatial", [0, 5], 10, 10)


def test_missing_episode_is_rejected(collector, tmp_path):
    a = summary(0)
    a["episodes"].pop()
    write(tmp_path, a)
    write(tmp_path, summary(5))
    with pytest.raises(AssertionError):
        collector(tmp_path, "libero_spatial", [0, 5], 10, 10)
